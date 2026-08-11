//! End-to-end sync orchestrator (Phase 7).
//!
//! Flow of [`SyncService::apply`]:
//!
//! 1. Load the project + `subtitles.json` + `voices/voices.json`.
//!    Refuse if either is missing.
//! 2. Load `voices/synced/sync.json` (if any) as the current cache.
//! 3. For every subtitle segment that has a TTS entry, compute its
//!    expected cache key from `(tts_cache_key, target_duration,
//!    sync_settings)`. Compare against the manifest to build the
//!    "todo" list per the requested [`SyncMode`].
//! 4. If nothing needs syncing → return `UpToDate` synchronously.
//! 5. Otherwise register a `Sync` job, spawn a background task that:
//!      * subscribes to `sync.segment_completed` and persists each
//!        entry into `sync.json` incrementally;
//!      * subscribes to `sync.progress` for coarse UI progress;
//!      * forwards the job cancel token to the worker via
//!        `jsonrpc://cancel`;
//!      * awaits the worker response, clears the subtitle `dirty.sync`
//!        flag when the whole doc is covered, and emits the terminal
//!        `job://update`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::db::DbHandle;
use crate::ffmpeg::detection::FfmpegHandle;
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::projects::ProjectService;
use crate::subtitles::{SubtitleCacheFile, SubtitleDoc, SubtitleService};
use crate::tts::{TtsCacheFile, TtsManifest};
use crate::worker::WorkerSupervisor;

use super::cache::{synced_dir, SyncCacheFile, SYNCED_MANIFEST_RELATIVE};
use super::errors::SyncError;
use super::models::{
    build_sync_cache_key, synced_file_relative, PreviewSyncResult, SyncEnv, SyncGenerateStart,
    SyncManifest, SyncMode, SyncPlan, SyncRequest, SyncSegmentEntry, SyncSettings, SyncStatus,
    SyncSummary,
};

const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

pub struct SyncService {
    app: AppHandle,
    db: DbHandle,
    projects: Arc<ProjectService>,
    subtitles: Arc<SubtitleService>,
    jobs: Arc<JobRegistry>,
    worker: Arc<WorkerSupervisor>,
    ffmpeg: Arc<FfmpegHandle>,
    manifest_lock: Arc<parking_lot::Mutex<()>>,
}

struct FinalizeContext {
    job_id: String,
    project_id: String,
    project_root: PathBuf,
    settings: SyncSettings,
}

struct TerminalEvent {
    job_id: String,
    project_id: String,
    status: JobStatus,
    progress: f32,
    error_code: Option<String>,
    error_message: Option<String>,
}

/// One segment queued for the worker along with the cache identity
/// the host expects the worker to stamp on the returned manifest entry.
pub(crate) struct TodoEntry {
    pub segment_id: u32,
    pub target_start: f64,
    pub target_end: f64,
    pub source_file: String,
    pub output_file: String,
    pub tts_cache_key: String,
    pub source_sample_rate: Option<u32>,
}

impl SyncService {
    pub fn new(
        app: AppHandle,
        db: DbHandle,
        projects: Arc<ProjectService>,
        subtitles: Arc<SubtitleService>,
        jobs: Arc<JobRegistry>,
        worker: Arc<WorkerSupervisor>,
        ffmpeg: Arc<FfmpegHandle>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app,
            db,
            projects,
            subtitles,
            jobs,
            worker,
            ffmpeg,
            manifest_lock: Arc::new(parking_lot::Mutex::new(())),
        })
    }

    // -------------------------------------------------- read-only queries

    pub async fn env(self: &Arc<Self>) -> Result<SyncEnv, SyncError> {
        let av = self.ffmpeg.availability();
        Ok(SyncEnv {
            ffmpeg_available: av.available,
            ffmpeg_path: av.ffmpeg_path,
            default_min_speed: SyncSettings::default().min_speed,
            default_max_speed: SyncSettings::default().max_speed,
        })
    }

    pub async fn get_manifest(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<SyncManifest>, SyncError> {
        let root = self.project_root(&project_id).await?;
        SyncCacheFile::load(&root).map_err(|source| SyncError::Io {
            path: root.display().to_string(),
            source,
        })
    }

    /// Compute a compact summary comparing the manifest with the
    /// current subtitle doc + TTS manifest.
    pub async fn get_summary(
        self: &Arc<Self>,
        project_id: String,
        settings: SyncSettings,
    ) -> Result<Option<SyncSummary>, SyncError> {
        let root = self.project_root(&project_id).await?;
        let Some(doc) = SubtitleCacheFile::load(&root).map_err(io_err(&root))? else {
            return Ok(None);
        };
        let tts = TtsCacheFile::load(&root).map_err(io_err(&root))?;
        let manifest = SyncCacheFile::load(&root).map_err(io_err(&root))?;
        Ok(Some(build_summary(
            &doc,
            tts.as_ref(),
            manifest.as_ref(),
            &settings,
        )))
    }

    // ---------------------------------------------------------- preview

    /// Materialise (or return the cached copy of) the synced WAV for a
    /// single segment. Powers the "Preview Synced Voice" button.
    pub async fn preview_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
        settings: Option<SyncSettings>,
    ) -> Result<PreviewSyncResult, SyncError> {
        let root = self.project_root(&project_id).await?;
        let effective = settings.unwrap_or_default().normalised();

        let doc = SubtitleCacheFile::load(&root)
            .map_err(io_err(&root))?
            .ok_or(SyncError::NoSubtitles)?;
        let seg = doc
            .segments
            .iter()
            .find(|s| s.id == segment_id)
            .ok_or(SyncError::SegmentNotFound { segment_id })?
            .clone();
        let target_duration = (seg.end - seg.start).max(0.0);
        if !seg.start.is_finite() || !seg.end.is_finite() || seg.end <= seg.start {
            return Err(SyncError::InvalidTiming {
                segment_id,
                reason: format!("start={}, end={}", seg.start, seg.end),
            });
        }

        let tts_manifest = TtsCacheFile::load(&root)
            .map_err(io_err(&root))?
            .ok_or(SyncError::NoTts)?;
        let tts_entry = tts_manifest
            .find(segment_id)
            .ok_or(SyncError::SegmentTtsMissing { segment_id })?;

        // Cache check — if the manifest already has a matching entry
        // and the file is present, return it directly.
        if let Some(existing) = SyncCacheFile::load(&root).map_err(io_err(&root))? {
            if let Some(entry) = existing.find(segment_id) {
                let expected =
                    build_sync_cache_key(&tts_entry.cache_key, target_duration, &effective);
                let abs = root.join(&entry.file);
                if entry.cache_key == expected
                    && entry.tts_cache_key == tts_entry.cache_key
                    && abs.exists()
                {
                    return Ok(PreviewSyncResult {
                        segment_id,
                        status: entry.status,
                        target_duration_secs: entry.target_duration_secs,
                        original_duration_secs: entry.original_duration_secs,
                        final_duration_secs: entry.final_duration_secs,
                        speed_factor: entry.speed_factor,
                        absolute_path: abs.display().to_string(),
                        relative_path: entry.file.clone(),
                        cache_hit: true,
                    });
                }
            }
        }

        // Miss → run the worker for just this segment.
        let rel_out = synced_file_relative(segment_id);
        let abs_out = root.join(&rel_out);
        if let Some(parent) = abs_out.parent() {
            std::fs::create_dir_all(parent).map_err(|source| SyncError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }

        let params = json!({
            "projectRoot": root.display().to_string(),
            "ffmpegPath": self.ffmpeg_path_str(),
            "settings": settings_to_wire(&effective),
            "segment": {
                "id": segment_id,
                "targetStart": seg.start,
                "targetEnd": seg.end,
                "sourceFile": tts_entry.file.clone(),
                "outputFile": rel_out.clone(),
                "ttsCacheKey": tts_entry.cache_key.clone(),
                "sourceSampleRate": tts_entry.sample_rate,
            },
        });
        let request_id = self.worker.new_request_id();
        let v = self
            .worker
            .request_no_timeout_with_id(&request_id, "sync.apply_one", params)
            .await
            .map_err(|e| match e {
                crate::worker::WorkerError::Rpc(rpc) => SyncError::from_rpc(rpc),
                other => SyncError::Worker(other),
            })?;

        let entry = entry_from_worker_response(&v, &seg, &tts_entry.cache_key, target_duration)?;
        self.upsert_manifest(&root, entry.clone(), effective)?;
        Ok(PreviewSyncResult {
            segment_id,
            status: entry.status,
            target_duration_secs: entry.target_duration_secs,
            original_duration_secs: entry.original_duration_secs,
            final_duration_secs: entry.final_duration_secs,
            speed_factor: entry.speed_factor,
            absolute_path: abs_out.display().to_string(),
            relative_path: entry.file,
            cache_hit: false,
        })
    }

    // ------------------------------------------------------- generate batch

    pub async fn apply(
        self: &Arc<Self>,
        project_id: String,
        request: SyncRequest,
    ) -> Result<SyncGenerateStart, SyncError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);

        let doc = SubtitleCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(SyncError::NoSubtitles)?;
        let tts_manifest = TtsCacheFile::load(&project_root)
            .map_err(io_err(&project_root))?
            .ok_or(SyncError::NoTts)?;

        // Ensure voices/synced/ exists so tmp writes don't race dir creation.
        let _ = std::fs::create_dir_all(synced_dir(&project_root));

        let settings = request.settings.normalised();
        let manifest = SyncCacheFile::load(&project_root).map_err(io_err(&project_root))?;

        let todo = plan_generation(&doc, &tts_manifest, manifest.as_ref(), &request, &settings);
        if todo.is_empty() {
            let summary = build_summary(&doc, Some(&tts_manifest), manifest.as_ref(), &settings);
            let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
            return Ok(SyncGenerateStart::UpToDate { summary });
        }

        // Seed the manifest so a mid-run crash still leaves a valid file.
        let seed_manifest = manifest.unwrap_or_else(|| SyncManifest::empty(settings));
        {
            let _guard = self.manifest_lock.lock();
            let mut m = seed_manifest.clone();
            m.settings = settings;
            m.updated_at = Utc::now();
            SyncCacheFile::save(&project_root, &m).map_err(io_err(&project_root))?;
        }

        let segments_wire = todo
            .iter()
            .map(|t| {
                json!({
                    "id": t.segment_id,
                    "targetStart": t.target_start,
                    "targetEnd": t.target_end,
                    "sourceFile": t.source_file,
                    "outputFile": t.output_file,
                    "ttsCacheKey": t.tts_cache_key,
                    "sourceSampleRate": t.source_sample_rate,
                })
            })
            .collect::<Vec<_>>();

        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), project_id.clone(), JobStage::Sync)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Sync,
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

        // Wire cancellation → worker.
        let request_id = self.worker.new_request_id();
        let request_id_for_cancel = request_id.clone();
        let worker_for_cancel = self.worker.clone();
        let cancel_token = handle.cancel.clone();
        tokio::spawn(async move {
            cancel_token.wait().await;
            let _ = worker_for_cancel
                .cancel_request(&request_id_for_cancel)
                .await;
        });

        // Wire coarse progress → job://progress.
        let last_emit = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let last_persisted = Arc::new(parking_lot::Mutex::new(0.0f32));
        let app_for_prog = self.app.clone();
        let job_for_prog = job_id.clone();
        let project_for_prog = project_id.clone();
        let db_for_prog = self.db.clone();
        let request_id_for_prog = request_id.clone();
        let progress_sub = self.worker.subscribe(
            "sync.progress",
            Arc::new(move |_, params| {
                let Some(target) = params.get("requestId").and_then(|v| v.as_str()) else {
                    return;
                };
                if target != request_id_for_prog {
                    return;
                }
                let frac = params
                    .get("fraction")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let mut guard = last_emit.lock();
                if guard.elapsed().as_millis() < PROGRESS_EMIT_MIN_INTERVAL_MS && frac < 1.0 {
                    return;
                }
                *guard = Instant::now();
                drop(guard);
                let evt = JobProgressEvent {
                    id: job_for_prog.clone(),
                    project_id: project_for_prog.clone(),
                    stage: JobStage::Sync,
                    progress: frac,
                };
                let _ = app_for_prog.emit("job://progress", evt);
                let mut last = last_persisted.lock();
                if (frac - *last).abs() >= 0.05 || frac >= 1.0 {
                    *last = frac;
                    let db = db_for_prog.clone();
                    let jid = job_for_prog.clone();
                    tokio::spawn(async move {
                        let _ = db
                            .run(move |d| JobsRepo::update_progress(d, &jid, frac))
                            .await;
                    });
                }
            }),
        );

        // Wire per-segment completion → persist into sync.json.
        let manifest_lock = self.manifest_lock.clone();
        let project_root_for_seg = project_root.clone();
        let app_for_seg = self.app.clone();
        let job_for_seg = job_id.clone();
        let project_for_seg = project_id.clone();
        let request_id_for_seg = request_id.clone();
        let settings_for_seg = settings;
        // Snapshot subtitle timing (id → (start, end)) so per-segment
        // completions can rebuild the entry without re-reading the doc.
        let target_lookup: std::collections::HashMap<u32, (f64, f64)> = doc
            .segments
            .iter()
            .map(|s| (s.id, (s.start, s.end)))
            .collect();
        let target_lookup = Arc::new(target_lookup);
        let target_lookup_for_seg = target_lookup.clone();
        let segment_sub = self.worker.subscribe(
            "sync.segment_completed",
            Arc::new(move |_, params| {
                let Some(target) = params.get("requestId").and_then(|v| v.as_str()) else {
                    return;
                };
                if target != request_id_for_seg {
                    return;
                }
                let Some(seg_id) = params
                    .get("segmentId")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                else {
                    tracing::warn!("sync.segment_completed missing segmentId");
                    return;
                };
                let Some(&(start, end)) = target_lookup_for_seg.get(&seg_id) else {
                    tracing::warn!(segment = seg_id, "sync.segment_completed for unknown id");
                    return;
                };
                let target_duration = (end - start).max(0.0);
                let entry = match parse_worker_entry(params, seg_id, start, end, target_duration) {
                    Ok(e) => e,
                    Err(err) => {
                        tracing::warn!(%err, "sync.segment_completed with invalid payload");
                        return;
                    }
                };
                let root = project_root_for_seg.clone();
                let lock = manifest_lock.clone();
                let app = app_for_seg.clone();
                let job_id = job_for_seg.clone();
                let project_id = project_for_seg.clone();
                let settings = settings_for_seg;
                tokio::spawn(async move {
                    let (segment_id, file_rel, synced_count, subtitle_count, status) = {
                        let _guard = lock.lock();
                        let mut manifest = SyncCacheFile::load(&root)
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| SyncManifest::empty(settings));
                        manifest.settings = settings;
                        let segment_id = entry.segment_id;
                        let file = entry.file.clone();
                        let status = entry.status;
                        manifest.upsert(entry);
                        let synced = manifest.segments.len() as u32;
                        if let Err(err) = SyncCacheFile::save(&root, &manifest) {
                            tracing::warn!(%err, "failed to persist sync manifest");
                        }
                        let count = SubtitleCacheFile::load(&root)
                            .ok()
                            .flatten()
                            .map(|d| d.segments.len() as u32)
                            .unwrap_or(0);
                        (segment_id, file, synced, count, status)
                    };
                    let payload = json!({
                        "jobId": job_id,
                        "projectId": project_id,
                        "segmentId": segment_id,
                        "file": file_rel,
                        "status": status.as_str(),
                        "syncedCount": synced_count,
                        "subtitleCount": subtitle_count,
                    });
                    let _ = app.emit("sync://segment_completed", payload);
                });
            }),
        );

        // Kick off the actual RPC on a background task.
        let this = self.clone();
        let jobs_for_dereg = self.jobs.clone();
        let project_root_bg = project_root.clone();
        let project_id_bg = project_id.clone();
        let job_id_bg = job_id.clone();
        let settings_bg = settings;
        let ffmpeg_path = self.ffmpeg_path_str();
        tokio::spawn(async move {
            let params = json!({
                "projectRoot": project_root_bg.display().to_string(),
                "ffmpegPath": ffmpeg_path,
                "settings": settings_to_wire(&settings_bg),
                "segments": segments_wire,
            });
            let result = this
                .worker
                .request_no_timeout_with_id(&request_id, "sync.apply_batch", params)
                .await;
            this.worker.unsubscribe(progress_sub);
            this.worker.unsubscribe(segment_sub);
            let ctx = FinalizeContext {
                job_id: job_id_bg.clone(),
                project_id: project_id_bg,
                project_root: project_root_bg,
                settings: settings_bg,
            };
            this.finalize(ctx, result).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(SyncGenerateStart::Started(snap))
    }

    // -------------------------------------------------------- finalisation

    async fn finalize(
        self: Arc<Self>,
        ctx: FinalizeContext,
        result: Result<Value, crate::worker::WorkerError>,
    ) {
        let FinalizeContext {
            job_id,
            project_id,
            project_root,
            settings,
        } = ctx;
        match result {
            Ok(_) => {
                let _ = self.stamp_completion(&project_root).await;
                if let Ok(Some(summary)) = self.get_summary(project_id.clone(), settings).await {
                    let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                }
                self.finalize_success(&job_id, &project_id).await;
            }
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let se = SyncError::from_rpc(rpc);
                    if matches!(se, SyncError::Cancelled) {
                        self.finalize_cancel(&job_id, &project_id).await;
                    } else {
                        self.finalize_failure(
                            &job_id,
                            &project_id,
                            sync_err_code(&se),
                            &se.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        &job_id,
                        &project_id,
                        "SYNC_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    async fn stamp_completion(&self, project_root: &std::path::Path) -> std::io::Result<()> {
        let _guard = self.manifest_lock.lock();
        if let Some(mut m) = SyncCacheFile::load(project_root)? {
            m.updated_at = Utc::now();
            SyncCacheFile::save(project_root, &m)?;
        }
        Ok(())
    }

    async fn maybe_clear_dirty(
        self: &Arc<Self>,
        project_id: String,
        summary: &SyncSummary,
    ) -> Result<(), SyncError> {
        // Only clear the sync flag when every subtitle actually has an
        // up-to-date synced WAV. `too_long` entries still count as
        // "synced" from a cache standpoint — we ran the pipeline, we
        // just also flagged them for the user.
        if summary.missing_count == 0
            && summary.stale_count == 0
            && summary.subtitle_count > 0
            && summary.synced_count >= summary.subtitle_count
        {
            let _ = self
                .subtitles
                .clear_dirty_flags(project_id, crate::subtitles::DirtyFlags::only_sync())
                .await
                .map_err(|e| tracing::warn!(%e, "failed to clear subtitle dirty.sync flag"));
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
            tracing::warn!(%err, "failed to emit job://update");
        }
    }

    fn emit_terminal(&self, evt: TerminalEvent) {
        let now = Utc::now();
        let snap = JobSnapshot {
            id: evt.job_id,
            project_id: evt.project_id,
            stage: JobStage::Sync,
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

    // ------------------------------------------------------ manifest helpers

    fn upsert_manifest(
        &self,
        project_root: &Path,
        entry: SyncSegmentEntry,
        settings: SyncSettings,
    ) -> Result<(), SyncError> {
        let _guard = self.manifest_lock.lock();
        let mut manifest = SyncCacheFile::load(project_root)
            .map_err(|source| SyncError::Io {
                path: project_root.display().to_string(),
                source,
            })?
            .unwrap_or_else(|| SyncManifest::empty(settings));
        manifest.settings = settings;
        manifest.upsert(entry);
        SyncCacheFile::save(project_root, &manifest).map_err(|source| SyncError::Io {
            path: project_root.display().to_string(),
            source,
        })?;
        Ok(())
    }

    fn ffmpeg_path_str(&self) -> Option<String> {
        self.ffmpeg
            .get()
            .map(|svc| svc.ffmpeg().display().to_string())
    }

    async fn project_root(&self, project_id: &str) -> Result<PathBuf, SyncError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        Ok(PathBuf::from(&rec.root_path))
    }
}

// ------------------------------------------------------------------- helpers

fn map_project_err(err: crate::projects::ProjectError) -> SyncError {
    match err {
        crate::projects::ProjectError::Db(e) => SyncError::Db(e),
        other => SyncError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

fn io_err(root: &Path) -> impl Fn(std::io::Error) -> SyncError + '_ {
    move |source| SyncError::Io {
        path: root.display().to_string(),
        source,
    }
}

fn settings_to_wire(s: &SyncSettings) -> Value {
    let n = s.normalised();
    json!({
        "minSpeed": n.min_speed,
        "maxSpeed": n.max_speed,
        "outputSampleRate": n.output_sample_rate,
        "outputChannels": n.output_channels,
    })
}

/// Rebuild a [`SyncSegmentEntry`] from a `sync.apply_one` result frame.
fn entry_from_worker_response(
    v: &Value,
    subtitle: &crate::subtitles::SubtitleSegment,
    tts_cache_key: &str,
    target_duration: f64,
) -> Result<SyncSegmentEntry, SyncError> {
    parse_worker_entry(
        v,
        subtitle.id,
        subtitle.start,
        subtitle.end,
        target_duration,
    )
    .map(|mut e| {
        if e.tts_cache_key.is_empty() {
            e.tts_cache_key = tts_cache_key.to_string();
        }
        e
    })
}

fn parse_worker_entry(
    v: &Value,
    segment_id: u32,
    target_start: f64,
    target_end: f64,
    target_duration: f64,
) -> Result<SyncSegmentEntry, SyncError> {
    let status_raw = v.get("status").and_then(|x| x.as_str()).unwrap_or("");
    let status = SyncStatus::parse(status_raw).unwrap_or(SyncStatus::Fits);
    let original_duration_secs = v
        .get("originalDurationSecs")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0);
    let final_duration_secs = v
        .get("finalDurationSecs")
        .and_then(|x| x.as_f64())
        .unwrap_or(target_duration);
    let speed_factor = v.get("speedFactor").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
    let cache_key = v
        .get("cacheKey")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let tts_cache_key = v
        .get("ttsCacheKey")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let file = v
        .get("file")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let sample_rate = v.get("sampleRate").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let channels = v.get("channels").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
    let size_bytes = v.get("sizeBytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let generated_at: DateTime<Utc> = Utc::now();
    Ok(SyncSegmentEntry {
        segment_id,
        status,
        target_start,
        target_end,
        target_duration_secs: target_duration,
        original_duration_secs,
        final_duration_secs,
        speed_factor,
        cache_key,
        tts_cache_key,
        file,
        sample_rate,
        channels,
        size_bytes,
        generated_at,
    })
}

/// Build the worker todo list for a given [`SyncRequest`]. This is
/// where cache-hit filtering happens — the worker only synthesises
/// what we ask for, no more.
pub(crate) fn plan_generation(
    doc: &SubtitleDoc,
    tts: &TtsManifest,
    manifest: Option<&SyncManifest>,
    request: &SyncRequest,
    settings: &SyncSettings,
) -> Vec<TodoEntry> {
    let force_ids: Option<std::collections::HashSet<u32>> = match &request.mode {
        SyncMode::Selected { ids } => Some(ids.iter().copied().collect()),
        _ => None,
    };
    let force_all = matches!(request.mode, SyncMode::All);
    let mut todo: Vec<TodoEntry> = Vec::new();
    for seg in &doc.segments {
        let Some(tts_entry) = tts.find(seg.id) else {
            continue;
        };
        if !seg.start.is_finite() || !seg.end.is_finite() || seg.end <= seg.start {
            continue;
        }
        let target_duration = (seg.end - seg.start).max(0.0);
        let expected = build_sync_cache_key(&tts_entry.cache_key, target_duration, settings);
        let should_generate = match &force_ids {
            Some(set) => set.contains(&seg.id),
            None => {
                if force_all {
                    true
                } else {
                    !manifest
                        .and_then(|m| m.find(seg.id))
                        .map(|entry| {
                            entry.cache_key == expected
                                && entry.tts_cache_key == tts_entry.cache_key
                        })
                        .unwrap_or(false)
                }
            }
        };
        if !should_generate {
            continue;
        }
        todo.push(TodoEntry {
            segment_id: seg.id,
            target_start: seg.start,
            target_end: seg.end,
            source_file: tts_entry.file.clone(),
            output_file: synced_file_relative(seg.id),
            tts_cache_key: tts_entry.cache_key.clone(),
            source_sample_rate: Some(tts_entry.sample_rate),
        });
    }
    todo
}

/// Compute a summary comparing every subtitle against the two manifests.
pub(crate) fn build_summary(
    doc: &SubtitleDoc,
    tts: Option<&TtsManifest>,
    manifest: Option<&SyncManifest>,
    settings: &SyncSettings,
) -> SyncSummary {
    let settings = settings.normalised();
    let mut synced = 0u32;
    let mut missing = 0u32;
    let mut stale = 0u32;
    let mut too_long = 0u32;
    let mut adjusted = 0u32;
    let mut fits = 0u32;
    let mut empty = 0u32;
    let subtitle_count = doc.segments.len() as u32;
    for seg in &doc.segments {
        let Some(tts_entry) = tts.and_then(|t| t.find(seg.id)) else {
            missing += 1;
            continue;
        };
        if !seg.start.is_finite() || !seg.end.is_finite() || seg.end <= seg.start {
            missing += 1;
            continue;
        }
        let target_duration = (seg.end - seg.start).max(0.0);
        let expected = build_sync_cache_key(&tts_entry.cache_key, target_duration, &settings);
        match manifest.and_then(|m| m.find(seg.id)) {
            Some(entry) => {
                let matches =
                    entry.cache_key == expected && entry.tts_cache_key == tts_entry.cache_key;
                if matches {
                    synced += 1;
                    match entry.status {
                        SyncStatus::TooLong => too_long += 1,
                        SyncStatus::Adjusted => adjusted += 1,
                        SyncStatus::Fits => fits += 1,
                        SyncStatus::Empty => empty += 1,
                    }
                } else {
                    stale += 1;
                }
            }
            None => missing += 1,
        }
    }
    let updated_at = manifest.map(|m| m.updated_at).unwrap_or_else(Utc::now);
    SyncSummary {
        settings,
        subtitle_count,
        synced_count: synced,
        missing_count: missing,
        stale_count: stale,
        too_long_count: too_long,
        adjusted_count: adjusted,
        fits_count: fits,
        empty_count: empty,
        updated_at,
        relative_path: SYNCED_MANIFEST_RELATIVE.into(),
    }
}

fn sync_err_code(err: &SyncError) -> &'static str {
    match err {
        SyncError::NoSubtitles => "SYNC_NO_SUBTITLES",
        SyncError::NoTts => "SYNC_NO_TTS",
        SyncError::SegmentNotFound { .. } => "SYNC_SEGMENT_NOT_FOUND",
        SyncError::SegmentTtsMissing { .. } => "SYNC_TTS_MISSING",
        SyncError::InvalidTiming { .. } => "SYNC_INVALID_TIMING",
        SyncError::SourceMissing { .. } => "SYNC_SOURCE_MISSING",
        SyncError::SourceInvalid { .. } => "SYNC_SOURCE_INVALID",
        SyncError::FfmpegMissing => "SYNC_FFMPEG_MISSING",
        SyncError::EngineFailure { .. } => "SYNC_ENGINE_FAILURE",
        SyncError::DiskFull => "SYNC_DISK_FULL",
        SyncError::WorkerCrash => "SYNC_WORKER_CRASH",
        SyncError::Cancelled => "SYNC_CANCELLED",
        SyncError::Worker(_) => "SYNC_WORKER_ERROR",
        SyncError::Registry(_) => "SYNC_REGISTRY",
        SyncError::Db(_) => "SYNC_DB",
        SyncError::Io { .. } => "SYNC_IO",
    }
}

/// Pure planner exposed for unit tests + host-side preview classification.
pub fn plan_for(
    target_duration_secs: f64,
    source_duration_secs: f64,
    settings: &SyncSettings,
) -> SyncPlan {
    let s = settings.normalised();
    let target = target_duration_secs.max(0.0);
    let source = source_duration_secs.max(0.0);
    if source <= 0.02 {
        return SyncPlan {
            status: SyncStatus::Empty,
            target_duration_secs: target,
            original_duration_secs: source,
            final_duration_secs: target,
            speed_factor: 1.0,
        };
    }
    if source <= target + 0.01 {
        return SyncPlan {
            status: SyncStatus::Fits,
            target_duration_secs: target,
            original_duration_secs: source,
            final_duration_secs: target,
            speed_factor: 1.0,
        };
    }
    if target <= 0.0 {
        return SyncPlan {
            status: SyncStatus::TooLong,
            target_duration_secs: 0.0,
            original_duration_secs: source,
            final_duration_secs: source / s.max_speed.max(1e-6) as f64,
            speed_factor: s.max_speed,
        };
    }
    let required = (source / target) as f32;
    if required <= s.max_speed + 1e-6 {
        let speed = required.max(s.min_speed);
        return SyncPlan {
            status: SyncStatus::Adjusted,
            target_duration_secs: target,
            original_duration_secs: source,
            final_duration_secs: source / speed as f64,
            speed_factor: speed,
        };
    }
    SyncPlan {
        status: SyncStatus::TooLong,
        target_duration_secs: target,
        original_duration_secs: source,
        final_duration_secs: source / s.max_speed as f64,
        speed_factor: s.max_speed,
    }
}

// Keep the internal helper reachable from tests too.
#[allow(dead_code)]
fn _touch(_: &SyncSegmentEntry) {
    let _ = build_sync_cache_key;
}
