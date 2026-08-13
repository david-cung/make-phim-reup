//! End-to-end audio-mix orchestrator (Phase 8).
//!
//! Flow of [`MixService::apply`]:
//!
//! 1. Load the project + `subtitles.json` + `voices/synced/sync.json`.
//!    Refuse if either is missing (mix requires *some* synced voice).
//! 2. Fingerprint the source video so cache invalidations track file
//!    edits automatically.
//! 3. Build the deterministic cache key. If it matches the current
//!    manifest entry and the WAV is on disk → return `UpToDate`.
//! 4. Otherwise register a `Mix` job, spawn a background task that:
//!      * builds the FFmpeg argv via [`super::ffmpeg_cmd::build_mix_command`];
//!      * spawns ffmpeg with `-progress pipe:1`;
//!      * streams progress → `job://progress`;
//!      * awaits the child, propagates cancellation from the
//!        [`crate::jobs::JobHandle`] token;
//!      * on success writes the new manifest entry and clears the
//!        subtitle `dirty.mix` flag when everything covers.
//!
//! No Python worker in the loop — mixing is pure FFmpeg, so we stay in
//! Rust and reuse the same argv-plus-progress pattern as Phase 2 audio
//! extraction.

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
use crate::ffmpeg::progress::{feed_line, ProgressBlock};
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::media::fingerprint::fingerprint_file;
use crate::media::SourceFingerprint;
use crate::projects::ProjectService;
use crate::subtitles::{SubtitleCacheFile, SubtitleDoc, SubtitleService};
use crate::sync::{SyncCacheFile, SyncManifest, SyncStatus};

use super::cache::{mix_output_path, MixCacheFile, MIX_MANIFEST_RELATIVE, MIX_OUTPUT_RELATIVE};
use super::errors::MixError;
use super::ffmpeg_cmd::build_mix_command;
use super::models::{
    build_mix_cache_key, MixEntry, MixEnv, MixGenerateStart, MixManifest, MixRequest, MixSettings,
    MixStatus, MixSummary, MixVoiceInput, PreviewMixResult,
};

const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

pub struct MixService {
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
    settings: MixSettings,
    source_fp: SourceFingerprint,
    voice_segment_count: u32,
    subtitle_count: u32,
    output_path: PathBuf,
}

struct TerminalEvent {
    job_id: String,
    project_id: String,
    status: JobStatus,
    progress: f32,
    error_code: Option<String>,
    error_message: Option<String>,
}

impl MixService {
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

    pub async fn env(self: &Arc<Self>) -> Result<MixEnv, MixError> {
        let av = self.ffmpeg.availability();
        Ok(MixEnv {
            ffmpeg_available: av.available,
            ffmpeg_path: av.ffmpeg_path,
            default_settings: MixSettings::default(),
        })
    }

    pub async fn get_manifest(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<MixManifest>, MixError> {
        let root = self.project_root(&project_id).await?;
        MixCacheFile::load(&root).map_err(io_err(&root))
    }

    pub async fn get_summary(
        self: &Arc<Self>,
        project_id: String,
        settings: MixSettings,
    ) -> Result<Option<MixSummary>, MixError> {
        let root = self.project_root(&project_id).await?;
        let doc = match SubtitleCacheFile::load(&root).map_err(io_err(&root))? {
            Some(d) => d,
            None => return Ok(None),
        };
        let sync = SyncCacheFile::load(&root).map_err(io_err(&root))?;
        let manifest = MixCacheFile::load(&root).map_err(io_err(&root))?;

        // Fingerprint the source only when needed for the cache-hit check.
        let source_fp = self.fingerprint_source(&project_id).await.ok();
        Ok(Some(build_summary(
            &root,
            &doc,
            sync.as_ref(),
            manifest.as_ref(),
            &settings,
            source_fp.as_ref(),
        )))
    }

    /// Return the last-generated mix (if any) as a preview payload so
    /// the UI can play it back through an `<audio>` element.
    pub async fn get_preview(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<PreviewMixResult>, MixError> {
        let root = self.project_root(&project_id).await?;
        let manifest = match MixCacheFile::load(&root).map_err(io_err(&root))? {
            Some(m) => m,
            None => return Ok(None),
        };
        let Some(entry) = manifest.current else {
            return Ok(None);
        };
        let abs = root.join(&entry.file);
        if !abs.exists() {
            return Ok(None);
        }
        Ok(Some(PreviewMixResult {
            absolute_path: abs.display().to_string(),
            relative_path: entry.file,
            duration_secs: entry.duration_secs,
            sample_rate: entry.sample_rate,
            channels: entry.channels,
            cache_hit: true,
        }))
    }

    // ------------------------------------------------------- generate batch

    pub async fn apply(
        self: &Arc<Self>,
        project_id: String,
        request: MixRequest,
    ) -> Result<MixGenerateStart, MixError> {
        // 1. FFmpeg must be reachable.
        let ffmpeg_svc = self.ffmpeg.get().ok_or(MixError::FfmpegMissing)?;

        // 2. Project + subtitles + sync manifest.
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);
        let source_media = rec
            .source_media_path
            .clone()
            .ok_or(MixError::NoSourceMedia)?;
        let source_path = PathBuf::from(&source_media);

        let doc = SubtitleCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(MixError::NoSubtitles)?;
        let sync_manifest = SyncCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(MixError::NoSyncedVoices)?;

        // 3. Collect voice inputs.
        let settings = request.settings.normalised();
        let voices = collect_voice_inputs(&doc, &sync_manifest);
        if voices.iter().all(|v| v.is_empty) {
            return Err(MixError::NoSyncedVoices);
        }

        // 4. Verify every non-empty voice file exists on disk.
        for v in &voices {
            if v.is_empty {
                continue;
            }
            let abs = project_root.join(&v.relative_file);
            if !abs.exists() {
                return Err(MixError::VoiceMissing {
                    segment_id: v.segment_id,
                    path: abs.display().to_string(),
                });
            }
        }

        // 5. Fingerprint source video for cache identity.
        let source_fp = fingerprint_file(&source_path).map_err(|source| MixError::Io {
            path: source_path.display().to_string(),
            source,
        })?;

        let cache_key = build_mix_cache_key(&source_fp, &voices, &settings);

        // 6. Cache hit → return synchronously.
        let existing = MixCacheFile::load(&project_root).map_err(io_err(&project_root))?;
        if let Some(m) = &existing {
            if let Some(entry) = &m.current {
                if entry.cache_key == cache_key {
                    let abs = project_root.join(&entry.file);
                    if abs.exists() {
                        let summary = build_summary(
                            &project_root,
                            &doc,
                            Some(&sync_manifest),
                            existing.as_ref(),
                            &settings,
                            Some(&source_fp),
                        );
                        let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                        return Ok(MixGenerateStart::UpToDate { summary });
                    }
                }
            }
        }

        // 7. Ensure the output dir exists so tmp writes don't race.
        let output_path = mix_output_path(&project_root);
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| MixError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }

        // 8. Register the job + emit initial snapshot.
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), project_id.clone(), JobStage::Mix)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Mix,
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
            settings,
            source_fp,
            voice_segment_count: voices.iter().filter(|v| !v.is_empty).count() as u32,
            subtitle_count: doc.segments.len() as u32,
            output_path: output_path.clone(),
        };
        let ffmpeg_bin = ffmpeg_svc.ffmpeg().to_path_buf();
        let cancel = handle.cancel.clone();
        let jobs_for_dereg = self.jobs.clone();
        let job_id_bg = job_id.clone();
        let voice_inputs = voices;
        tokio::spawn(async move {
            let outcome = this
                .clone()
                .run_ffmpeg(
                    ffmpeg_bin,
                    ctx.project_root.clone(),
                    source_path,
                    voice_inputs,
                    ctx.settings,
                    ctx.output_path.clone(),
                    ctx.job_id.clone(),
                    ctx.project_id.clone(),
                    cancel,
                )
                .await;
            this.finalize(ctx, outcome).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(MixGenerateStart::Started(snap))
    }

    // ---------------------------------------------------------- ffmpeg run

    #[allow(clippy::too_many_arguments)]
    async fn run_ffmpeg(
        self: Arc<Self>,
        ffmpeg_bin: PathBuf,
        project_root: PathBuf,
        source_video: PathBuf,
        voices: Vec<MixVoiceInput>,
        settings: MixSettings,
        output_path: PathBuf,
        job_id: String,
        project_id: String,
        cancel: CancelToken,
    ) -> Result<u64, MixError> {
        // Probe source duration so progress fractions make sense.
        // We use ffprobe on the source; the JSON reader is already
        // available via ffmpeg::probe.
        let ffprobe = match self.ffmpeg.get() {
            Some(svc) => svc.ffprobe().to_path_buf(),
            None => return Err(MixError::FfmpegMissing),
        };
        let duration_secs = crate::ffmpeg::probe::probe_video(&ffprobe, &source_video)
            .await
            .map(|m| m.duration_secs)
            .unwrap_or(0.0);

        let cmd = build_mix_command(&source_video, &voices, &settings, &output_path);
        tracing::info!(
            job = %job_id,
            voices = voices.iter().filter(|v| !v.is_empty).count(),
            source = %source_video.display(),
            output = %output_path.display(),
            original_volume = settings.original_volume,
            voice_volume = settings.voice_volume,
            ducking = settings.ducking_enabled,
            "spawning ffmpeg mix"
        );
        tracing::debug!(args = ?cmd.args, "ffmpeg mix argv");

        let mut command = Command::new(&ffmpeg_bin);
        command
            .args(&cmd.args)
            // FFmpeg resolves relative input paths against CWD, and our
            // voice paths are stored relative to the project root.
            .current_dir(&project_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|source| MixError::Io {
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
                        stage: JobStage::Mix,
                        progress: frac,
                    };
                    if let Err(err) = app.emit("job://progress", evt) {
                        tracing::warn!(%err, "failed to emit job://progress for mix");
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
                // Best-effort partial cleanup.
                let _ = std::fs::remove_file(&output_path);
                return Err(MixError::Cancelled);
            }
            r = child.wait() => r,
        };

        let status = wait_res.map_err(|source| MixError::Io {
            path: output_path.display().to_string(),
            source,
        })?;
        let _ = progress_task.await;
        let stderr_tail = stderr_task.await.unwrap_or_default();

        if !status.success() {
            let _ = std::fs::remove_file(&output_path);
            if is_disk_full(&stderr_tail) {
                return Err(MixError::DiskFull);
            }
            return Err(MixError::Ffmpeg(crate::ffmpeg::FfmpegError::RunFailed {
                code: status.code().unwrap_or(-1),
                stderr: tail_lines(&stderr_tail, 6),
            }));
        }

        // Basic output sanity check.
        let meta = std::fs::metadata(&output_path).map_err(|source| MixError::Io {
            path: output_path.display().to_string(),
            source,
        })?;
        if meta.len() == 0 {
            let _ = std::fs::remove_file(&output_path);
            return Err(MixError::OutputInvalid {
                path: output_path.display().to_string(),
            });
        }

        // Emit the final 100% tick so the UI progress bar always reaches the end.
        let evt = JobProgressEvent {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Mix,
            progress: 1.0,
        };
        let _ = self.app.emit("job://progress", evt);

        Ok(meta.len())
    }

    // -------------------------------------------------------- finalisation

    async fn finalize(self: Arc<Self>, ctx: FinalizeContext, result: Result<u64, MixError>) {
        let FinalizeContext {
            job_id,
            project_id,
            project_root,
            cache_key,
            settings,
            source_fp,
            voice_segment_count,
            subtitle_count,
            output_path,
        } = ctx;
        match result {
            Ok(size_bytes) => {
                // Probe the produced WAV for sample rate + channels + duration.
                let (duration_secs, sample_rate, channels) =
                    probe_wav_metadata(&output_path).unwrap_or((0.0, 44100, 2));
                tracing::info!(
                    job = %job_id,
                    output = %output_path.display(),
                    size_bytes,
                    duration_secs,
                    sample_rate,
                    channels,
                    voice_segments = voice_segment_count,
                    "ffmpeg mix complete"
                );
                let entry = MixEntry {
                    cache_key: cache_key.clone(),
                    source_fingerprint: source_fp,
                    file: MIX_OUTPUT_RELATIVE.to_string(),
                    duration_secs,
                    sample_rate,
                    channels,
                    size_bytes,
                    voice_segment_count,
                    subtitle_count,
                    settings,
                    generated_at: Utc::now(),
                };
                if let Err(err) = self.persist_manifest(&project_root, entry, settings) {
                    tracing::warn!(%err, "failed to persist mix manifest");
                }
                if let Ok(Some(summary)) = self.get_summary(project_id.clone(), settings).await {
                    let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                }
                self.finalize_success(&job_id, &project_id).await;
            }
            Err(MixError::Cancelled) => {
                self.finalize_cancel(&job_id, &project_id).await;
            }
            Err(err) => {
                let code = err.code();
                let msg = err.to_string();
                tracing::warn!(%err, "mix failed");
                self.finalize_failure(&job_id, &project_id, code, &msg)
                    .await;
            }
        }
    }

    fn persist_manifest(
        &self,
        project_root: &Path,
        entry: MixEntry,
        settings: MixSettings,
    ) -> Result<(), MixError> {
        let _guard = self.manifest_lock.lock();
        let mut manifest = MixCacheFile::load(project_root)
            .map_err(io_err(project_root))?
            .unwrap_or_else(|| MixManifest::empty(settings));
        manifest.settings = settings;
        manifest.current = Some(entry);
        manifest.updated_at = Utc::now();
        MixCacheFile::save(project_root, &manifest).map_err(io_err(project_root))?;
        Ok(())
    }

    async fn maybe_clear_dirty(
        self: &Arc<Self>,
        project_id: String,
        summary: &MixSummary,
    ) -> Result<(), MixError> {
        // Only clear the mix flag when everything is ready.
        if matches!(summary.status, MixStatus::Ready) && !summary.needs_generate {
            let _ = self
                .subtitles
                .clear_dirty_flags(project_id, crate::subtitles::DirtyFlags::only_mix())
                .await
                .map_err(|e| tracing::warn!(%e, "failed to clear subtitle dirty.mix flag"));
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
            tracing::warn!(%err, "failed to emit job://update for mix");
        }
    }

    fn emit_terminal(&self, evt: TerminalEvent) {
        let now = Utc::now();
        let snap = JobSnapshot {
            id: evt.job_id,
            project_id: evt.project_id,
            stage: JobStage::Mix,
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

    async fn project_root(&self, project_id: &str) -> Result<PathBuf, MixError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        Ok(PathBuf::from(&rec.root_path))
    }

    async fn fingerprint_source(&self, project_id: &str) -> Result<SourceFingerprint, MixError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        let source = rec
            .source_media_path
            .clone()
            .ok_or(MixError::NoSourceMedia)?;
        let path = PathBuf::from(&source);
        fingerprint_file(&path).map_err(|source| MixError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

// ------------------------------------------------------------------- helpers

fn map_project_err(err: crate::projects::ProjectError) -> MixError {
    match err {
        crate::projects::ProjectError::Db(e) => MixError::Db(e),
        other => MixError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

fn io_err(root: &Path) -> impl Fn(std::io::Error) -> MixError + '_ {
    move |source| MixError::Io {
        path: root.display().to_string(),
        source,
    }
}

/// Filter and shape the sync manifest into the list the mixer wants.
///
/// * Empty (`SyncStatus::Empty`) segments still produce a silence WAV
///   in Phase 7, but adding them to the mix filter graph would waste
///   FFmpeg time — silence + anything = the same anything. We mark
///   them `is_empty` so the ffmpeg cmd builder skips them.
/// * Segments whose synced WAV entry is missing are also skipped —
///   they'll appear as `stale` in the mix summary until sync is rerun.
pub(crate) fn collect_voice_inputs(doc: &SubtitleDoc, sync: &SyncManifest) -> Vec<MixVoiceInput> {
    let mut out: Vec<MixVoiceInput> = Vec::with_capacity(doc.segments.len());
    for seg in &doc.segments {
        let Some(entry) = sync.find(seg.id) else {
            continue;
        };
        if !entry.file.starts_with("voices/synced/") {
            // Defensive — should always hold, but skip anything under
            // a foreign path rather than let it near FFmpeg.
            continue;
        }
        out.push(MixVoiceInput {
            segment_id: seg.id,
            target_start_secs: entry.target_start.max(0.0),
            relative_file: entry.file.clone(),
            sync_cache_key: entry.cache_key.clone(),
            is_empty: matches!(entry.status, SyncStatus::Empty),
        });
    }
    out
}

/// Compute the current mix summary given every relevant piece of
/// on-disk state.
pub(crate) fn build_summary(
    project_root: &Path,
    doc: &SubtitleDoc,
    sync: Option<&SyncManifest>,
    manifest: Option<&MixManifest>,
    settings: &MixSettings,
    source_fp: Option<&SourceFingerprint>,
) -> MixSummary {
    let subtitle_count = doc.segments.len() as u32;
    let (voice_segment_count, voices) = match sync {
        Some(sync) => {
            let inputs = collect_voice_inputs(doc, sync);
            let non_empty = inputs.iter().filter(|v| !v.is_empty).count() as u32;
            (non_empty, inputs)
        }
        None => (0, Vec::new()),
    };

    let entry = manifest.and_then(|m| m.current.as_ref());
    let (mut status, absolute, relative, duration, size, generated) = match entry {
        Some(e) => {
            let abs = project_root.join(&e.file);
            let exists = abs.exists();
            let status = if !exists {
                MixStatus::Missing
            } else if let Some(fp) = source_fp {
                let expected = build_mix_cache_key(fp, &voices, settings);
                if expected == e.cache_key {
                    MixStatus::Ready
                } else {
                    MixStatus::Stale
                }
            } else {
                // Without a fingerprint we can't decide freshness — trust
                // the file existence and treat as stale so the user re-runs.
                MixStatus::Stale
            };
            (
                status,
                exists.then(|| abs.display().to_string()),
                Some(e.file.clone()),
                Some(e.duration_secs),
                Some(e.size_bytes),
                Some(e.generated_at),
            )
        }
        None => (MixStatus::Missing, None, None, None, None, None),
    };

    // Special-case: sync manifest missing entirely → we can't build a mix.
    let mut warning: Option<String> = None;
    if sync.is_none() || voice_segment_count == 0 {
        status = MixStatus::Missing;
        warning = Some("No synced voice segments yet — run voice sync before mixing.".into());
    }

    let needs_generate = !matches!(status, MixStatus::Ready);
    MixSummary {
        status,
        settings: settings.normalised(),
        duration_secs: duration,
        absolute_path: absolute,
        relative_path: relative,
        voice_segment_count,
        subtitle_count,
        size_bytes: size,
        generated_at: generated,
        needs_generate,
        warning,
        manifest_relative_path: MIX_MANIFEST_RELATIVE.into(),
    }
}

/// Probe a WAV via the `hound` crate would be overkill for one call —
/// we hand-parse the RIFF header just enough to grab the three fields
/// the manifest needs.
fn probe_wav_metadata(path: &Path) -> std::io::Result<(f64, u32, u32)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 44];
    f.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err(std::io::Error::other("not a RIFF/WAVE file"));
    }
    let channels = u16::from_le_bytes([hdr[22], hdr[23]]) as u32;
    let sample_rate = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);
    let byte_rate = u32::from_le_bytes([hdr[28], hdr[29], hdr[30], hdr[31]]) as u64;
    let file_size = std::fs::metadata(path)?.len();
    // 44-byte canonical header → data section length ≈ file - 44.
    let data_bytes = file_size.saturating_sub(44);
    let duration = if byte_rate > 0 {
        data_bytes as f64 / byte_rate as f64
    } else {
        0.0
    };
    Ok((duration, sample_rate, channels))
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
