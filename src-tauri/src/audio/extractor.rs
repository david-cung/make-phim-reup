//! End-to-end extraction pipeline:
//!
//! 1. Verify FFmpeg is available.
//! 2. Fingerprint the source; check disk space.
//! 3. Probe the source (get duration + require audio stream).
//! 4. If cache hit → return existing path (no ffmpeg spawn).
//! 5. Otherwise:
//!    * insert a `jobs` row (status = running)
//!    * register a cancel token
//!    * run ffmpeg while emitting `job://progress` events
//!    * on success: write the cache manifest + mark job completed
//!    * on failure/cancel: delete partial output + mark job accordingly

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use thiserror::Error;
use uuid::Uuid;

use crate::audio::cache::{AudioCacheEntry, AudioCacheFile};
use crate::db::{DbError, DbHandle};
use crate::ffmpeg::detection::FfmpegHandle;
use crate::ffmpeg::errors::FfmpegError;
use crate::ffmpeg::extract::{run_extraction, AudioExtractParams, ProgressFn};
use crate::ffmpeg::probe;
use crate::jobs::{
    JobProgressEvent, JobRegistry, JobRegistryError, JobSnapshot, JobStage, JobStatus, JobsRepo,
};
use crate::media::disk;
use crate::media::fingerprint::fingerprint_file;

pub const AUDIO_OUTPUT_REL: &str = "audio/original.wav";
const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("FFmpeg is not available: {reason}")]
    FfmpegUnavailable { reason: String },

    #[error("project has no imported source video")]
    NoSourceMedia,

    #[error(transparent)]
    Ffmpeg(#[from] FfmpegError),

    #[error(transparent)]
    Db(#[from] DbError),

    #[error(transparent)]
    Registry(#[from] JobRegistryError),

    #[error("io error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone)]
pub struct ExtractionRequest {
    pub project_id: String,
    pub project_root: PathBuf,
    pub source_media_path: PathBuf,
    pub params: AudioExtractParams,
}

/// Result of `extract` — either a cache hit (no work done) or a job that
/// has been kicked off in the background.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ExtractionStart {
    /// Nothing to do — audio already exists on disk with a matching fingerprint.
    CacheHit {
        entry: AudioCacheEntry,
        absolute_output: String,
    },
    /// Extraction started. Progress events will follow on `job://progress`.
    Started(JobSnapshot),
}

/// Terminal payload of a completed extraction, cached & fresh alike.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractionResult {
    pub entry: AudioCacheEntry,
    pub absolute_output: String,
    pub cache_hit: bool,
}

pub struct AudioExtractor {
    app: AppHandle,
    db: DbHandle,
    ffmpeg: Arc<FfmpegHandle>,
    jobs: Arc<JobRegistry>,
}

struct RunContext {
    job_id: String,
    project_id: String,
    project_root: PathBuf,
    source_path: PathBuf,
    params: AudioExtractParams,
    src_fp: crate::media::SourceFingerprint,
    duration_secs: f64,
    cancel: crate::ffmpeg::extract::CancelToken,
}

impl AudioExtractor {
    pub fn new(
        app: AppHandle,
        db: DbHandle,
        ffmpeg: Arc<FfmpegHandle>,
        jobs: Arc<JobRegistry>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app,
            db,
            ffmpeg,
            jobs,
        })
    }

    /// Kick off an extraction. Returns `CacheHit` synchronously if the
    /// output already exists; otherwise registers a job and spawns the
    /// ffmpeg run on a background task.
    pub async fn extract(
        self: &Arc<Self>,
        req: ExtractionRequest,
    ) -> Result<ExtractionStart, ExtractError> {
        // 1. FFmpeg must be reachable.
        let svc = self
            .ffmpeg
            .get()
            .ok_or_else(|| ExtractError::FfmpegUnavailable {
                reason: self
                    .ffmpeg
                    .availability()
                    .error
                    .unwrap_or_else(|| "not detected".into()),
            })?;

        // 2. Fingerprint the source (fast, ≤ 128 KiB of IO).
        let src_fp =
            fingerprint_file(&req.source_media_path).map_err(|source| ExtractError::Io {
                path: req.source_media_path.display().to_string(),
                source,
            })?;

        // 3. Cache lookup.
        let cache_file =
            AudioCacheFile::load(&req.project_root).map_err(|source| ExtractError::Io {
                path: req.project_root.display().to_string(),
                source,
            })?;
        if let Some(entry) = cache_file.hit(&req.project_root, &src_fp, &req.params) {
            let abs = entry.absolute_output(&req.project_root);
            return Ok(ExtractionStart::CacheHit {
                absolute_output: abs.display().to_string(),
                entry,
            });
        }

        // 4. Probe the source for duration + stream sanity.
        let meta = probe::probe_video(svc.ffprobe(), &req.source_media_path).await?;
        probe::require_audio_stream(&meta)?;

        // 5. Disk space check.
        let needed = disk::estimate_pcm_wav_size(
            meta.duration_secs,
            req.params.sample_rate,
            req.params.channels,
        );
        let free = disk::free_bytes(&req.project_root).map_err(|source| ExtractError::Io {
            path: req.project_root.display().to_string(),
            source,
        })?;
        if free < needed + (100 << 20) {
            return Err(ExtractError::Ffmpeg(FfmpegError::DiskSpaceLow {
                required_bytes: needed,
                available_bytes: free,
            }));
        }

        // 6. Register the job in memory + DB.
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self.jobs.register(
            job_id.clone(),
            req.project_id.clone(),
            JobStage::ExtractAudio,
        )?;

        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: req.project_id.clone(),
            stage: JobStage::ExtractAudio,
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

        // 7. Spawn the ffmpeg task.
        let this = self.clone();
        let ctx = RunContext {
            job_id,
            project_id: req.project_id.clone(),
            project_root: req.project_root.clone(),
            source_path: req.source_media_path.clone(),
            params: req.params.clone(),
            src_fp,
            duration_secs: meta.duration_secs,
            cancel: handle.cancel.clone(),
        };
        tokio::spawn(async move {
            this.run_ffmpeg_task(ctx).await;
        });

        Ok(ExtractionStart::Started(snap))
    }

    async fn run_ffmpeg_task(self: Arc<Self>, ctx: RunContext) {
        let RunContext {
            job_id,
            project_id,
            project_root,
            source_path,
            params,
            src_fp,
            duration_secs,
            cancel,
        } = ctx;
        let output_abs = project_root.join(AUDIO_OUTPUT_REL);

        // Progress emitter — throttled to avoid flooding the webview.
        let emit_app = self.app.clone();
        let emit_project = project_id.clone();
        let emit_job = job_id.clone();
        let db_for_progress = self.db.clone();
        let db_progress_job = job_id.clone();
        let last_emit = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let last_persisted = Arc::new(parking_lot::Mutex::new(0.0f32));
        let on_progress: ProgressFn = Arc::new(move |frac: f32| {
            let mut guard = last_emit.lock();
            let should_emit =
                guard.elapsed().as_millis() >= PROGRESS_EMIT_MIN_INTERVAL_MS || frac >= 1.0;
            if should_emit {
                *guard = Instant::now();
                drop(guard);
                let evt = JobProgressEvent {
                    id: emit_job.clone(),
                    project_id: emit_project.clone(),
                    stage: JobStage::ExtractAudio,
                    progress: frac,
                };
                if let Err(err) = emit_app.emit("job://progress", evt) {
                    tracing::warn!(%err, "failed to emit job://progress");
                }
                // Persist progress in the DB at coarser intervals — every 5%.
                let mut last = last_persisted.lock();
                if (frac - *last).abs() >= 0.05 || frac >= 1.0 {
                    *last = frac;
                    let db = db_for_progress.clone();
                    let jid = db_progress_job.clone();
                    tokio::spawn(async move {
                        let _ = db
                            .run(move |d| JobsRepo::update_progress(d, &jid, frac))
                            .await;
                    });
                }
            }
        });

        let ffmpeg_svc = match self.ffmpeg.get() {
            Some(s) => s,
            None => {
                self.finalize_failure(
                    &job_id,
                    &project_id,
                    "FFMPEG_UNAVAILABLE",
                    "ffmpeg disappeared mid-run",
                )
                .await;
                self.jobs.deregister(&job_id);
                return;
            }
        };

        let result = run_extraction(
            ffmpeg_svc.ffmpeg(),
            &source_path,
            &output_abs,
            &params,
            duration_secs,
            on_progress,
            cancel.clone(),
        )
        .await;

        match result {
            Ok(outcome) => {
                // Update cache manifest.
                let entry = AudioCacheEntry {
                    source: src_fp,
                    source_path: source_path.display().to_string(),
                    params,
                    output_relative: AUDIO_OUTPUT_REL.into(),
                    output_size_bytes: outcome.output_size_bytes,
                    duration_secs: outcome.duration_secs,
                    created_at: Utc::now(),
                };
                let mut cache = AudioCacheFile::load(&project_root).unwrap_or_default();
                cache.original_wav = Some(entry.clone());
                if let Err(err) = cache.save(&project_root) {
                    tracing::warn!(%err, "failed to write audio cache manifest");
                }
                self.finalize_success(&job_id, &project_id).await;
            }
            Err(FfmpegError::Cancelled) => {
                self.finalize_cancel(&job_id, &project_id).await;
            }
            Err(err) => {
                let code = err.code();
                let msg = err.to_string();
                tracing::warn!(%err, "extraction failed");
                self.finalize_failure(&job_id, &project_id, code, &msg)
                    .await;
            }
        }
        self.jobs.deregister(&job_id);
    }

    async fn finalize_success(&self, job_id: &str, project_id: &str) {
        let db = self.db.clone();
        let jid = job_id.to_string();
        let _ = db
            .run(move |d| {
                JobsRepo::update_status(d, &jid, JobStatus::Completed, Some(1.0), None, None)
            })
            .await;
        self.emit_terminal(job_id, project_id, JobStatus::Completed, 1.0, None, None);
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
        self.emit_terminal(
            job_id,
            project_id,
            JobStatus::Cancelled,
            0.0,
            Some("CANCELLED".into()),
            Some("user cancelled".into()),
        );
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
        self.emit_terminal(
            job_id,
            project_id,
            JobStatus::Failed,
            0.0,
            Some(code.to_string()),
            Some(message.to_string()),
        );
    }

    fn emit_update(&self, snap: &JobSnapshot) {
        if let Err(err) = self.app.emit("job://update", snap.clone()) {
            tracing::warn!(%err, "failed to emit job://update");
        }
    }

    fn emit_terminal(
        &self,
        job_id: &str,
        project_id: &str,
        status: JobStatus,
        progress: f32,
        error_code: Option<String>,
        error_message: Option<String>,
    ) {
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.into(),
            project_id: project_id.into(),
            stage: JobStage::ExtractAudio,
            status,
            progress,
            error_code,
            error_message,
            created_at: now,
            started_at: Some(now),
            completed_at: Some(now),
        };
        self.emit_update(&snap);
    }
}
