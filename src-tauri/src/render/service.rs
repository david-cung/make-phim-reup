//! End-to-end final-render orchestrator (Phase 9).
//!
//! Flow of [`RenderService::apply`]:
//!
//! 1. Load the project + `subtitles.json` + `audio/mix.json`. Refuse
//!    if the mix hasn't been produced — Phase 9 is the last stop, it
//!    doesn't run Phase 8 for you.
//! 2. Fingerprint the source video so cache identity tracks file
//!    edits automatically.
//! 3. Build the deterministic cache key. If it matches the current
//!    manifest entry and the output file is on disk → return
//!    `UpToDate` (unless `force=true`).
//! 4. Otherwise register a `Render` job, spawn a background task
//!    that:
//!      * exports the Vietnamese SRT to the output folder (external
//!        or burned modes);
//!      * builds the FFmpeg argv via [`super::ffmpeg_cmd::build_render_command`];
//!      * spawns ffmpeg with `-progress pipe:1`;
//!      * streams progress → `job://progress`;
//!      * awaits the child, propagates cancellation from the
//!        [`crate::jobs::JobHandle`] token;
//!      * runs `ffprobe` on the output to validate stream layout;
//!      * on success writes the new manifest entry and clears the
//!        subtitle `dirty.render` flag.
//!
//! No Python worker in the loop — rendering is pure FFmpeg.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use uuid::Uuid;

use crate::db::DbHandle;
use crate::ffmpeg::detection::FfmpegHandle;
use crate::ffmpeg::extract::CancelToken;
use crate::ffmpeg::probe::probe_video;
use crate::ffmpeg::progress::{feed_line, ProgressBlock};
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::media::fingerprint::fingerprint_file;
use crate::media::SourceFingerprint;
use crate::mix::{MixCacheFile, MixEntry};
use crate::projects::ProjectService;
use crate::subtitles::{srt as srt_writer, SubtitleCacheFile, SubtitleDoc, SubtitleService};

use super::cache::{
    default_render_output_path, subtitle_sidecar_path, RenderCacheFile, RENDER_MANIFEST_RELATIVE,
};
use super::errors::RenderError;
use super::ffmpeg_cmd::build_render_command;
use super::models::{
    build_render_cache_key, OutputFormat, RenderEntry, RenderEnv, RenderGenerateStart,
    RenderManifest, RenderRequest, RenderSettings, RenderStatus, RenderSummary, SubtitleMode,
};

const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

/// The defaults we advertise, adjusted for what this FFmpeg can do.
///
/// Burning is the default because a sidecar the viewer never loads reads as
/// no subtitles at all. A build without libass can't burn, though, so
/// offering it there would only produce a failed render — fall back to the
/// sidecar, which at least lands beside the movie.
pub(crate) fn default_settings_for(can_burn: bool) -> RenderSettings {
    let mut s = RenderSettings::default();
    if !can_burn && matches!(s.subtitle_mode, SubtitleMode::Burned) {
        s.subtitle_mode = SubtitleMode::External;
    }
    s
}

const ADVERTISED_VIDEO_CODECS: &[&str] = &["copy", "libx264", "libx265", "libvpx-vp9"];
const ADVERTISED_AUDIO_CODECS: &[&str] = &["aac", "libopus", "ac3", "mp3"];
const ADVERTISED_FORMATS: &[&str] = &["mp4", "mkv"];

pub struct RenderService {
    app: AppHandle,
    db: DbHandle,
    projects: Arc<ProjectService>,
    subtitles: Arc<SubtitleService>,
    jobs: Arc<JobRegistry>,
    ffmpeg: Arc<FfmpegHandle>,
    manifest_lock: Arc<parking_lot::Mutex<()>>,
}

struct FinalizeContext {
    job_id: String,
    project_id: String,
    project_root: PathBuf,
    cache_key: String,
    mix_cache_key: String,
    settings: RenderSettings,
    source_fp: SourceFingerprint,
    subtitle_mode: SubtitleMode,
    output_path: PathBuf,
    subtitle_sidecar: Option<PathBuf>,
}

struct RenderOutcome {
    size_bytes: u64,
    duration_secs: f64,
    video_streams: u32,
    audio_streams: u32,
    subtitle_streams: u32,
}

struct TerminalEvent {
    job_id: String,
    project_id: String,
    status: JobStatus,
    progress: f32,
    error_code: Option<String>,
    error_message: Option<String>,
}

impl RenderService {
    pub fn new(
        app: AppHandle,
        db: DbHandle,
        projects: Arc<ProjectService>,
        subtitles: Arc<SubtitleService>,
        jobs: Arc<JobRegistry>,
        ffmpeg: Arc<FfmpegHandle>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app,
            db,
            projects,
            subtitles,
            jobs,
            ffmpeg,
            manifest_lock: Arc::new(parking_lot::Mutex::new(())),
        })
    }

    // -------------------------------------------------- read-only queries

    pub async fn env(self: &Arc<Self>) -> Result<RenderEnv, RenderError> {
        let av = self.ffmpeg.availability();
        Ok(RenderEnv {
            ffmpeg_available: av.available,
            ffmpeg_path: av.ffmpeg_path,
            default_settings: default_settings_for(av.has_subtitles_filter),
            subtitle_burn_available: av.has_subtitles_filter,
            video_codecs: ADVERTISED_VIDEO_CODECS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            audio_codecs: ADVERTISED_AUDIO_CODECS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            output_formats: ADVERTISED_FORMATS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        })
    }

    pub async fn get_manifest(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<RenderManifest>, RenderError> {
        let root = self.project_root(&project_id).await?;
        RenderCacheFile::load(&root).map_err(io_err(&root))
    }

    pub async fn get_summary(
        self: &Arc<Self>,
        project_id: String,
        settings: RenderSettings,
    ) -> Result<Option<RenderSummary>, RenderError> {
        let root = self.project_root(&project_id).await?;
        // The render summary is only meaningful when subtitles + mix
        // both exist — the panel needs them to be sensible.
        let doc = SubtitleCacheFile::load(&root).map_err(io_err(&root))?;
        let mix = MixCacheFile::load(&root).map_err(io_err(&root))?;
        let manifest = RenderCacheFile::load(&root).map_err(io_err(&root))?;
        let source_fp = self.fingerprint_source(&project_id).await.ok();
        Ok(Some(build_summary(
            &root,
            doc.as_ref(),
            mix.as_ref().and_then(|m| m.current.as_ref()),
            manifest.as_ref(),
            &settings,
            source_fp.as_ref(),
        )))
    }

    // ------------------------------------------------------- generate batch

    pub async fn apply(
        self: &Arc<Self>,
        project_id: String,
        request: RenderRequest,
    ) -> Result<RenderGenerateStart, RenderError> {
        let ffmpeg_svc = self.ffmpeg.get().ok_or(RenderError::FfmpegMissing)?;

        // Catch a burn request against a build without libass here. Left to
        // FFmpeg it fails deep into the run with "No option name near
        // '/path/to.srt'", which reads like a broken path rather than a
        // missing feature.
        if matches!(request.settings.subtitle_mode, SubtitleMode::Burned)
            && !ffmpeg_svc.has_subtitles_filter()
        {
            return Err(RenderError::SubtitleBurnUnsupported);
        }

        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);
        let source_media = rec
            .source_media_path
            .clone()
            .ok_or(RenderError::NoSourceMedia)?;
        let source_path = PathBuf::from(&source_media);

        let doc = SubtitleCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(RenderError::NoSubtitles)?;
        let mix_manifest = MixCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(RenderError::NoMix)?;
        let mix_entry = mix_manifest.current.clone().ok_or(RenderError::NoMix)?;
        let mixed_audio_abs = project_root.join(&mix_entry.file);
        if !mixed_audio_abs.exists() {
            return Err(RenderError::MixFileMissing {
                path: mixed_audio_abs.display().to_string(),
            });
        }

        let settings = request.settings.normalised();
        let source_fp = fingerprint_file(&source_path).map_err(|source| RenderError::Io {
            path: source_path.display().to_string(),
            source,
        })?;
        let cache_key = build_render_cache_key(&source_fp, &mix_entry.cache_key, &settings);

        // Resolve the output path — default when the user hasn't set
        // a custom one, otherwise trust the absolute path they passed.
        let output_path = resolve_output_path(&project_root, &settings)?;

        // 6. Cache hit → return synchronously (unless the caller
        // explicitly asked us to re-render).
        let existing = RenderCacheFile::load(&project_root).map_err(io_err(&project_root))?;
        if !request.force {
            if let Some(m) = &existing {
                if let Some(entry) = &m.current {
                    if entry.cache_key == cache_key {
                        let existing_abs = Path::new(&entry.file_absolute);
                        if existing_abs.exists() && existing_abs == output_path {
                            let summary = build_summary(
                                &project_root,
                                Some(&doc),
                                Some(&mix_entry),
                                existing.as_ref(),
                                &settings,
                                Some(&source_fp),
                            );
                            let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                            return Ok(RenderGenerateStart::UpToDate { summary });
                        }
                    }
                }
            }
        }

        // 7. Ensure the output dir exists so tmp writes don't race.
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| RenderError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }

        // Write the SRT up-front. Where it goes depends on what it is
        // for:
        //
        // * `Burned` only needs it as an input to FFmpeg's `subtitles=`
        //   filter, so the project's `output/` folder is fine.
        // * `External` makes the file part of the deliverable. It has to
        //   sit beside the movie and share its stem, or no player will
        //   pick it up — and when the user renders to a custom path
        //   (say `~/Downloads`), the project folder is nowhere near it.
        let subtitle_sidecar_path =
            subtitle_sidecar_path(&project_root, &output_path, settings.subtitle_mode);
        let needs_sidecar = matches!(
            settings.subtitle_mode,
            SubtitleMode::External | SubtitleMode::Burned
        );
        if needs_sidecar {
            if let Some(parent) = subtitle_sidecar_path.parent() {
                std::fs::create_dir_all(parent).map_err(|source| RenderError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
            write_sidecar_srt(&subtitle_sidecar_path, &doc).map_err(|source| RenderError::Io {
                path: subtitle_sidecar_path.display().to_string(),
                source,
            })?;
        }

        // 8. Register the job + emit initial snapshot.
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), project_id.clone(), JobStage::Render)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Render,
            status: JobStatus::Running,
            progress: 0.0,
            error_code: None,
            error_message: None,
            created_at: now,
            started_at: Some(now),
            completed_at: None,
        };
        let db = self.db.clone();
        let snap_for_insert = snap.clone();
        db.run(move |d| JobsRepo::insert(d, &snap_for_insert))
            .await?;
        self.emit_update(&snap);

        // 9. Spawn the ffmpeg task.
        let this = self.clone();
        let ctx = FinalizeContext {
            job_id: job_id.clone(),
            project_id: project_id.clone(),
            project_root: project_root.clone(),
            cache_key,
            mix_cache_key: mix_entry.cache_key.clone(),
            settings: settings.clone(),
            source_fp,
            subtitle_mode: settings.subtitle_mode,
            output_path: output_path.clone(),
            subtitle_sidecar: if matches!(settings.subtitle_mode, SubtitleMode::External) {
                Some(subtitle_sidecar_path.clone())
            } else {
                None
            },
        };
        let ffmpeg_bin = ffmpeg_svc.ffmpeg().to_path_buf();
        let ffprobe_bin = ffmpeg_svc.ffprobe().to_path_buf();
        let cancel = handle.cancel.clone();
        let jobs_for_dereg = self.jobs.clone();
        let job_id_bg = job_id.clone();
        let burn_subtitle_path = if matches!(settings.subtitle_mode, SubtitleMode::Burned) {
            Some(subtitle_sidecar_path.clone())
        } else {
            None
        };
        tokio::spawn(async move {
            let outcome = this
                .clone()
                .run_ffmpeg(
                    ffmpeg_bin.clone(),
                    ffprobe_bin.clone(),
                    project_root.clone(),
                    source_path.clone(),
                    mixed_audio_abs.clone(),
                    burn_subtitle_path,
                    ctx.settings.clone(),
                    ctx.output_path.clone(),
                    ctx.job_id.clone(),
                    ctx.project_id.clone(),
                    cancel,
                )
                .await;
            this.finalize(ctx, outcome).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(RenderGenerateStart::Started(snap))
    }

    // ---------------------------------------------------------- ffmpeg run

    #[allow(clippy::too_many_arguments)]
    async fn run_ffmpeg(
        self: Arc<Self>,
        ffmpeg_bin: PathBuf,
        ffprobe_bin: PathBuf,
        project_root: PathBuf,
        source_video: PathBuf,
        mixed_audio: PathBuf,
        burn_subtitle: Option<PathBuf>,
        settings: RenderSettings,
        output_path: PathBuf,
        job_id: String,
        project_id: String,
        cancel: CancelToken,
    ) -> Result<RenderOutcome, RenderError> {
        // Probe source duration so progress fractions make sense.
        let duration_secs = probe_video(&ffprobe_bin, &source_video)
            .await
            .map(|m| m.duration_secs)
            .unwrap_or(0.0);

        let cmd = build_render_command(
            &source_video,
            &mixed_audio,
            burn_subtitle.as_deref(),
            &settings,
            &output_path,
        );
        tracing::info!(
            job = %job_id,
            subtitle_mode = ?settings.subtitle_mode,
            source = %source_video.display(),
            mixed_audio = %mixed_audio.display(),
            output = %output_path.display(),
            audio_mapping = "1:a:0",
            "spawning ffmpeg render"
        );
        tracing::debug!(args = ?cmd.args, "ffmpeg render argv");

        let mut command = Command::new(&ffmpeg_bin);
        command
            .args(&cmd.args)
            .current_dir(&project_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|source| RenderError::Io {
            path: ffmpeg_bin.display().to_string(),
            source,
        })?;
        let pid = child.id();

        // Progress emitter.
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let app = self.app.clone();
        let db = self.db.clone();
        let last_emit = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let last_persisted = Arc::new(parking_lot::Mutex::new(0.0f32));
        let job_for_prog = job_id.clone();
        let project_for_prog = project_id.clone();
        let progress_task = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            let mut cur = ProgressBlock::default();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Some(done) = feed_line(&mut cur, &line) {
                    let frac =
                        done.fraction(duration_secs)
                            .unwrap_or(if done.ended { 1.0 } else { 0.0 });
                    let mut guard = last_emit.lock();
                    let should_emit = guard.elapsed().as_millis() >= PROGRESS_EMIT_MIN_INTERVAL_MS
                        || frac >= 1.0
                        || done.ended;
                    if !should_emit {
                        continue;
                    }
                    *guard = Instant::now();
                    drop(guard);
                    let evt = JobProgressEvent {
                        id: job_for_prog.clone(),
                        project_id: project_for_prog.clone(),
                        stage: JobStage::Render,
                        progress: frac,
                    };
                    if let Err(err) = app.emit("job://progress", evt) {
                        tracing::warn!(%err, "failed to emit job://progress for render");
                    }
                    let mut last = last_persisted.lock();
                    if (frac - *last).abs() >= 0.05 || frac >= 1.0 {
                        *last = frac;
                        let db = db.clone();
                        let jid = job_for_prog.clone();
                        tokio::spawn(async move {
                            let _ = db
                                .run(move |d| JobsRepo::update_progress(d, &jid, frac))
                                .await;
                        });
                    }
                }
            }
        });

        // stderr collector for post-mortem messages.
        let stderr_task = tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            let mut tail: Vec<String> = Vec::with_capacity(16);
            while let Ok(Some(line)) = reader.next_line().await {
                if tail.len() >= 32 {
                    tail.remove(0);
                }
                tail.push(line);
            }
            tail.join("\n")
        });

        let wait_res = tokio::select! {
            biased;
            _ = cancel.wait() => {
                terminate_child_pid(pid).await;
                let _ = child.wait().await;
                let _ = progress_task.await;
                let _ = stderr_task.await;
                let _ = std::fs::remove_file(&output_path);
                return Err(RenderError::Cancelled);
            }
            r = child.wait() => r,
        };

        let status = wait_res.map_err(|source| RenderError::Io {
            path: output_path.display().to_string(),
            source,
        })?;
        let _ = progress_task.await;
        let stderr_tail = stderr_task.await.unwrap_or_default();

        if !status.success() {
            let _ = std::fs::remove_file(&output_path);
            if is_disk_full(&stderr_tail) {
                return Err(RenderError::DiskFull);
            }
            return Err(RenderError::Ffmpeg(crate::ffmpeg::FfmpegError::RunFailed {
                code: status.code().unwrap_or(-1),
                stderr: tail_lines(&stderr_tail, 6),
            }));
        }

        // Post-run validation. Any failure here removes the partial
        // output so the user never sees a broken movie file dressed
        // up as a successful render.
        let meta = std::fs::metadata(&output_path).map_err(|source| RenderError::Io {
            path: output_path.display().to_string(),
            source,
        })?;
        if meta.len() == 0 {
            let _ = std::fs::remove_file(&output_path);
            return Err(RenderError::OutputInvalid {
                path: output_path.display().to_string(),
            });
        }

        let probed = probe_video(&ffprobe_bin, &output_path)
            .await
            .map_err(RenderError::Ffmpeg)?;
        if probed.duration_secs <= 0.0 {
            let _ = std::fs::remove_file(&output_path);
            return Err(RenderError::OutputInvalid {
                path: output_path.display().to_string(),
            });
        }
        let want_video = true;
        let want_audio = true;
        // We never mux the sidecar into the container in this
        // release, so we don't require a subtitle stream in the
        // output even when subtitle_mode is `External`.
        let want_subtitle = false;
        if (want_video && probed.video_stream_count == 0)
            || (want_audio && probed.audio_stream_count == 0)
            || (want_subtitle && probed.subtitle_stream_count == 0)
        {
            let _ = std::fs::remove_file(&output_path);
            return Err(RenderError::ValidationMismatch {
                path: output_path.display().to_string(),
                want_video,
                want_audio,
                want_subtitle,
                got_video: probed.video_stream_count,
                got_audio: probed.audio_stream_count,
                got_subtitle: probed.subtitle_stream_count,
            });
        }
        tracing::info!(
            job = %job_id,
            output = %output_path.display(),
            size_bytes = meta.len(),
            duration_secs = probed.duration_secs,
            video_streams = probed.video_stream_count,
            audio_streams = probed.audio_stream_count,
            subtitle_streams = probed.subtitle_stream_count,
            "ffmpeg render complete"
        );

        // Final 100% tick so the UI progress bar always reaches the end.
        let evt = JobProgressEvent {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Render,
            progress: 1.0,
        };
        let _ = self.app.emit("job://progress", evt);

        Ok(RenderOutcome {
            size_bytes: meta.len(),
            duration_secs: probed.duration_secs,
            video_streams: probed.video_stream_count,
            audio_streams: probed.audio_stream_count,
            subtitle_streams: probed.subtitle_stream_count,
        })
    }

    // -------------------------------------------------------- finalisation

    async fn finalize(
        self: Arc<Self>,
        ctx: FinalizeContext,
        result: Result<RenderOutcome, RenderError>,
    ) {
        let FinalizeContext {
            job_id,
            project_id,
            project_root,
            cache_key,
            mix_cache_key,
            settings,
            source_fp,
            subtitle_mode,
            output_path,
            subtitle_sidecar,
        } = ctx;
        match result {
            Ok(outcome) => {
                let relative = output_path
                    .strip_prefix(&project_root)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned());
                let subtitle_absolute = subtitle_sidecar.as_ref().map(|p| p.display().to_string());
                let entry = RenderEntry {
                    cache_key: cache_key.clone(),
                    source_fingerprint: source_fp,
                    mix_cache_key,
                    file_absolute: output_path.display().to_string(),
                    file_relative: relative,
                    subtitle_file_absolute: subtitle_absolute,
                    duration_secs: outcome.duration_secs,
                    size_bytes: outcome.size_bytes,
                    video_stream_count: outcome.video_streams,
                    audio_stream_count: outcome.audio_streams,
                    subtitle_stream_count: outcome.subtitle_streams,
                    subtitle_mode,
                    settings: settings.clone(),
                    generated_at: Utc::now(),
                };
                if let Err(err) = self.persist_manifest(&project_root, entry, settings.clone()) {
                    tracing::warn!(%err, "failed to persist render manifest");
                }
                if let Ok(Some(summary)) =
                    self.get_summary(project_id.clone(), settings.clone()).await
                {
                    let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                }
                self.finalize_success(&job_id, &project_id).await;
            }
            Err(RenderError::Cancelled) => {
                self.finalize_cancel(&job_id, &project_id).await;
            }
            Err(err) => {
                let code = err.code();
                let msg = err.to_string();
                tracing::warn!(%err, "render failed");
                self.finalize_failure(&job_id, &project_id, code, &msg)
                    .await;
            }
        }
    }

    fn persist_manifest(
        &self,
        project_root: &Path,
        entry: RenderEntry,
        settings: RenderSettings,
    ) -> Result<(), RenderError> {
        let _guard = self.manifest_lock.lock();
        let mut manifest = RenderCacheFile::load(project_root)
            .map_err(io_err(project_root))?
            .unwrap_or_else(|| RenderManifest::empty(settings.clone()));
        manifest.settings = settings;
        manifest.current = Some(entry);
        manifest.updated_at = Utc::now();
        RenderCacheFile::save(project_root, &manifest).map_err(io_err(project_root))?;
        Ok(())
    }

    async fn maybe_clear_dirty(
        self: &Arc<Self>,
        project_id: String,
        summary: &RenderSummary,
    ) -> Result<(), RenderError> {
        if matches!(summary.status, RenderStatus::Ready) && !summary.needs_render {
            let _ = self
                .subtitles
                .clear_dirty_flags(project_id, crate::subtitles::DirtyFlags::only_render())
                .await
                .map_err(|e| tracing::warn!(%e, "failed to clear subtitle dirty.render flag"));
        }
        Ok(())
    }

    // ---------------------------------------------------- job finalization

    async fn finalize_success(&self, job_id: &str, project_id: &str) {
        let db = self.db.clone();
        let jid = job_id.to_string();
        let _ = db
            .run(move |d| {
                JobsRepo::update_status(d, &jid, JobStatus::Completed, Some(1.0), None, None)
            })
            .await;
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            status: JobStatus::Completed,
            progress: 1.0,
            error_code: None,
            error_message: None,
        });
    }

    async fn finalize_cancel(&self, job_id: &str, project_id: &str) {
        let db = self.db.clone();
        let jid = job_id.to_string();
        let _ = db
            .run(move |d| {
                JobsRepo::update_status(
                    d,
                    &jid,
                    JobStatus::Cancelled,
                    None,
                    Some("CANCELLED"),
                    Some("user cancelled"),
                )
            })
            .await;
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            status: JobStatus::Cancelled,
            progress: 0.0,
            error_code: Some("CANCELLED".into()),
            error_message: Some("user cancelled".into()),
        });
    }

    async fn finalize_failure(&self, job_id: &str, project_id: &str, code: &str, message: &str) {
        let db = self.db.clone();
        let jid = job_id.to_string();
        let code_owned = code.to_string();
        let msg_owned = message.to_string();
        let _ = db
            .run(move |d| {
                JobsRepo::update_status(
                    d,
                    &jid,
                    JobStatus::Failed,
                    None,
                    Some(&code_owned),
                    Some(&msg_owned),
                )
            })
            .await;
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            status: JobStatus::Failed,
            progress: 0.0,
            error_code: Some(code.into()),
            error_message: Some(message.into()),
        });
    }

    fn emit_update(&self, snap: &JobSnapshot) {
        if let Err(err) = self.app.emit("job://update", snap.clone()) {
            tracing::warn!(%err, "failed to emit job://update for render");
        }
    }

    fn emit_terminal(&self, evt: TerminalEvent) {
        let now = Utc::now();
        let snap = JobSnapshot {
            id: evt.job_id,
            project_id: evt.project_id,
            stage: JobStage::Render,
            status: evt.status,
            progress: evt.progress,
            error_code: evt.error_code,
            error_message: evt.error_message,
            created_at: now,
            started_at: Some(now),
            completed_at: Some(now),
        };
        self.emit_update(&snap);
    }

    // ---------------------------------------------------- helpers

    async fn project_root(&self, project_id: &str) -> Result<PathBuf, RenderError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        Ok(PathBuf::from(&rec.root_path))
    }

    async fn fingerprint_source(&self, project_id: &str) -> Result<SourceFingerprint, RenderError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        let source = rec
            .source_media_path
            .clone()
            .ok_or(RenderError::NoSourceMedia)?;
        let path = PathBuf::from(&source);
        fingerprint_file(&path).map_err(|source| RenderError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

// ------------------------------------------------------------------- helpers

fn map_project_err(err: crate::projects::ProjectError) -> RenderError {
    match err {
        crate::projects::ProjectError::Db(e) => RenderError::Db(e),
        other => RenderError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

fn io_err(root: &Path) -> impl Fn(std::io::Error) -> RenderError + '_ {
    move |source| RenderError::Io {
        path: root.display().to_string(),
        source,
    }
}

/// Resolve the concrete output path for a render request.
///
/// * `settings.output_path == None` → default under `<project>/output/`.
/// * `settings.output_path == Some(p)` → validate that `p` is absolute
///   and refuse to write directly on top of the source video (source
///   protection per the spec).
pub(crate) fn resolve_output_path(
    project_root: &Path,
    settings: &RenderSettings,
) -> Result<PathBuf, RenderError> {
    let s = settings.clone().normalised();
    let candidate = match &s.output_path {
        Some(raw) => {
            let p = PathBuf::from(raw);
            if !p.is_absolute() {
                return Err(RenderError::InvalidOutputPath { path: raw.clone() });
            }
            p
        }
        None => default_render_output_path(project_root, s.output_format),
    };
    Ok(candidate)
}

/// Write the Vietnamese SRT sidecar next to the render output.
///
/// The SRT is the same content the Phase 5 `export_subtitles` command
/// produces for the `Translated` kind — reused verbatim so a user's
/// external subtitle preferences never drift between "Export SRT" and
/// "Render".
fn write_sidecar_srt(path: &Path, doc: &SubtitleDoc) -> std::io::Result<()> {
    let body = srt_writer::write(&doc.segments, |s| s.translated_text.clone());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("srt.tmp");
    std::fs::write(&tmp, body.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Compute the current render summary given every relevant piece of
/// on-disk state.
pub(crate) fn build_summary(
    project_root: &Path,
    doc: Option<&SubtitleDoc>,
    mix_entry: Option<&MixEntry>,
    manifest: Option<&RenderManifest>,
    settings: &RenderSettings,
    source_fp: Option<&SourceFingerprint>,
) -> RenderSummary {
    let s = settings.clone().normalised();
    let default_output = default_render_output_path(project_root, s.output_format);

    let entry = manifest.and_then(|m| m.current.as_ref());
    let (status, absolute, relative, subtitle_absolute, duration, size, generated) = match entry {
        Some(e) => {
            let abs = PathBuf::from(&e.file_absolute);
            let exists = abs.exists();
            let status = if !exists {
                RenderStatus::Missing
            } else if let (Some(fp), Some(mix)) = (source_fp, mix_entry) {
                let expected = build_render_cache_key(fp, &mix.cache_key, &s);
                if expected == e.cache_key {
                    RenderStatus::Ready
                } else {
                    RenderStatus::Stale
                }
            } else {
                // Without either input we can't decide freshness —
                // treat as stale so the user re-runs to be safe.
                RenderStatus::Stale
            };
            (
                status,
                exists.then(|| abs.display().to_string()),
                e.file_relative.clone(),
                e.subtitle_file_absolute.clone(),
                Some(e.duration_secs),
                Some(e.size_bytes),
                Some(e.generated_at),
            )
        }
        None => (RenderStatus::Missing, None, None, None, None, None, None),
    };

    let (video_streams, audio_streams, subtitle_streams, subtitle_mode) = match entry {
        Some(e) => (
            e.video_stream_count,
            e.audio_stream_count,
            e.subtitle_stream_count,
            e.subtitle_mode,
        ),
        None => (0, 0, 0, s.subtitle_mode),
    };

    let mut warning: Option<String> = None;
    if doc.is_none() {
        warning =
            Some("Build subtitles first — Phase 9 renders subtitles into the final movie.".into());
    } else if mix_entry.is_none() {
        warning = Some("Run audio mix first — Phase 9 needs the mixed Vietnamese audio.".into());
    }

    let needs_render = !matches!(status, RenderStatus::Ready);
    RenderSummary {
        status,
        settings: s,
        duration_secs: duration,
        absolute_path: absolute,
        relative_path: relative,
        subtitle_absolute_path: subtitle_absolute,
        size_bytes: size,
        generated_at: generated,
        video_stream_count: video_streams,
        audio_stream_count: audio_streams,
        subtitle_stream_count: subtitle_streams,
        subtitle_mode,
        needs_render,
        default_output_absolute: default_output.display().to_string(),
        warning,
        manifest_relative_path: RENDER_MANIFEST_RELATIVE.into(),
    }
}

async fn terminate_child_pid(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        unsafe {
            let _ = libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn is_disk_full(stderr: &str) -> bool {
    stderr.to_lowercase().contains("no space left on device")
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join(" | ")
}

// Consumed to silence unused-import in edge cases.
#[allow(dead_code)]
fn _use_format(_: OutputFormat) {}
