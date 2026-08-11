//! End-to-end speech-to-text orchestrator.
//!
//! Flow of :meth:`SttService::transcribe`:
//!
//! 1. Load the project + validate ``audio/original.wav`` exists.
//! 2. Fingerprint the WAV → derive a cache key from
//!    ``(audio_hash, options)`` (Rust side; the Python side does the
//!    same math independently to keep both authoritative).
//! 3. Read ``transcription/transcription.json``. If ``cache_key`` +
//!    ``audio_hash`` still match, return a ``CacheHit`` synchronously.
//! 4. Otherwise register a ``Transcribe`` job, spawn a background
//!    task that:
//!      * subscribes to ``stt.progress`` notifications and pumps
//!        them through as ``job://progress`` events;
//!      * subscribes to the job's cancel token and forwards it to
//!        the worker via ``jsonrpc://cancel``;
//!      * awaits the worker response, writes the transcript to disk,
//!        updates the DB, emits the terminal ``job://update``.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::db::DbHandle;
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::media::fingerprint::fingerprint_file;
use crate::projects::ProjectService;
use crate::worker::WorkerSupervisor;

use super::cache::{TranscriptCacheFile, TRANSCRIPT_RELATIVE};
use super::errors::SttError;
use super::models::{
    ModelInfo, SttEnv, SttOptions, TranscribeSegment, TranscribeStart, Transcript, TranscriptAudio,
    TranscriptSummary, WhisperWord,
};

pub const AUDIO_INPUT_REL: &str = "audio/original.wav";
const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

pub struct SttService {
    app: AppHandle,
    db: DbHandle,
    projects: Arc<ProjectService>,
    jobs: Arc<JobRegistry>,
    worker: Arc<WorkerSupervisor>,
}

/// Bundle every piece of state needed to finalise a transcription
/// job so [`SttService::finalize_transcribe`] doesn't drown in args.
struct TranscribeFinalizeContext {
    job_id: String,
    project_id: String,
    project_root: PathBuf,
    audio_hash: String,
    cache_key: String,
    options: SttOptions,
}

/// Terminal-event payload for [`SttService::emit_terminal`].
struct TerminalEvent {
    job_id: String,
    project_id: String,
    stage: JobStage,
    status: JobStatus,
    progress: f32,
    error_code: Option<String>,
    error_message: Option<String>,
}

impl SttService {
    pub fn new(
        app: AppHandle,
        db: DbHandle,
        projects: Arc<ProjectService>,
        jobs: Arc<JobRegistry>,
        worker: Arc<WorkerSupervisor>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app,
            db,
            projects,
            jobs,
            worker,
        })
    }

    // ------------------------------------------------------ read-only queries

    pub async fn env(self: &Arc<Self>) -> Result<SttEnv, SttError> {
        let value = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "stt.env", json!({}))
            .await
            .map_err(SttError::Worker)?;
        serde_json::from_value(value).map_err(|e| {
            SttError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })
    }

    pub async fn list_models(self: &Arc<Self>) -> Result<Vec<ModelInfo>, SttError> {
        #[derive(serde::Deserialize)]
        struct Response {
            models: Vec<ModelInfo>,
        }
        let value = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "stt.list_models", json!({}))
            .await
            .map_err(SttError::Worker)?;
        let resp: Response = serde_json::from_value(value).map_err(|e| {
            SttError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(resp.models)
    }

    /// Phase 11 — release the resident Whisper model so the multi-GB
    /// RAM footprint is returned to the OS between long idle periods.
    /// Returns whether a model was actually held.
    pub async fn unload(self: &Arc<Self>) -> Result<bool, SttError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            #[serde(default)]
            released: bool,
        }
        let value = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "stt.unload", json!({}))
            .await
            .map_err(SttError::Worker)?;
        let resp: Resp = serde_json::from_value(value).map_err(|e| {
            SttError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(resp.released)
    }

    pub async fn get_transcript_summary(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<TranscriptSummary>, SttError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        match TranscriptCacheFile::load(&root).map_err(|source| SttError::Io {
            path: root.display().to_string(),
            source,
        })? {
            Some(t) => Ok(Some(TranscriptSummary::from_transcript(
                &t,
                TRANSCRIPT_RELATIVE,
            ))),
            None => Ok(None),
        }
    }

    // ---------------------------------------------------------- download model

    pub async fn download_model(self: &Arc<Self>, name: String) -> Result<JobSnapshot, SttError> {
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), String::new(), JobStage::Transcribe)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: String::new(),
            stage: JobStage::Transcribe,
            status: JobStatus::Running,
            progress: 0.0,
            error_code: None,
            error_message: None,
            created_at: now,
            started_at: Some(now),
            completed_at: None,
        };
        // Phase 12 bug-fix — model downloads are NOT project-scoped,
        // so we can't persist them to the `jobs` table (its
        // `project_id` column has a `NOT NULL REFERENCES projects(id)`
        // FK). The download job lives entirely in-memory via
        // `self.jobs.register` above and is surfaced to the UI
        // through `emit_update` / `emit_terminal` events. Attempting
        // an insert here would trip the FK constraint and abort the
        // whole download before it starts.
        self.emit_update(&snap);

        let request_id = self.worker.new_request_id();
        let worker = self.worker.clone();
        let cancel = handle.cancel.clone();
        let request_id_for_cancel = request_id.clone();
        let worker_for_cancel = worker.clone();
        tokio::spawn(async move {
            cancel.wait().await;
            let _ = worker_for_cancel
                .cancel_request(&request_id_for_cancel)
                .await;
        });

        let app_for_prog = self.app.clone();
        let job_for_prog = job_id.clone();
        let request_id_for_prog = request_id.clone();
        let sub = self.worker.subscribe(
            "stt.download_progress",
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
                let evt = JobProgressEvent {
                    id: job_for_prog.clone(),
                    project_id: String::new(),
                    stage: JobStage::Transcribe,
                    progress: frac,
                };
                let _ = app_for_prog.emit("job://progress", evt);
            }),
        );

        let this = self.clone();
        let jobs_for_dereg = self.jobs.clone();
        let job_id_bg = job_id.clone();
        tokio::spawn(async move {
            let result = this
                .worker
                .request_no_timeout_with_id(
                    &request_id,
                    "stt.download_model",
                    json!({ "name": name }),
                )
                .await;
            this.worker.unsubscribe(sub);
            this.finalize_download(&job_id_bg, result).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(snap)
    }

    async fn finalize_download(
        self: Arc<Self>,
        job_id: &str,
        result: Result<Value, crate::worker::WorkerError>,
    ) {
        match result {
            Ok(_) => {
                self.finalize_success(job_id, "", JobStage::Transcribe)
                    .await
            }
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let stt_err = SttError::from_rpc(rpc);
                    if matches!(stt_err, SttError::Cancelled) {
                        self.finalize_cancel(job_id, "", JobStage::Transcribe).await;
                    } else {
                        self.finalize_failure(
                            job_id,
                            "",
                            JobStage::Transcribe,
                            stt_err_code(&stt_err),
                            &stt_err.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        job_id,
                        "",
                        JobStage::Transcribe,
                        "STT_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    // -------------------------------------------------------------- transcribe

    pub async fn transcribe(
        self: &Arc<Self>,
        project_id: String,
        options: SttOptions,
    ) -> Result<TranscribeStart, SttError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);
        let audio_abs = project_root.join(AUDIO_INPUT_REL);
        if !audio_abs.exists() {
            return Err(SttError::AudioNotExtracted);
        }

        // Fingerprint the WAV — reuses the Phase-2 partial hash so
        // trivial re-runs are O(128 KiB) regardless of duration.
        let audio_fp = fingerprint_file(&audio_abs).map_err(|source| SttError::Io {
            path: audio_abs.display().to_string(),
            source,
        })?;
        let audio_hash = audio_fp.hash.clone();
        let cache_key = build_cache_key(&audio_hash, &options);

        // Cache lookup.
        if let Some(existing) =
            TranscriptCacheFile::load(&project_root).map_err(|source| SttError::Io {
                path: project_root.display().to_string(),
                source,
            })?
        {
            if TranscriptCacheFile::hit(&existing, &cache_key, &audio_hash) {
                let summary = TranscriptSummary::from_transcript(&existing, TRANSCRIPT_RELATIVE);
                let abs = project_root.join(TRANSCRIPT_RELATIVE);
                return Ok(TranscribeStart::CacheHit {
                    transcript: summary,
                    absolute_path: abs.display().to_string(),
                });
            }
        }

        // Kick off a job.
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle =
            self.jobs
                .register(job_id.clone(), project_id.clone(), JobStage::Transcribe)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Transcribe,
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

        // Wire cancellation: watch the token, forward to the worker.
        let request_id = self.worker.new_request_id();
        let worker = self.worker.clone();
        let cancel_token = handle.cancel.clone();
        let request_id_for_cancel = request_id.clone();
        let worker_for_cancel = worker.clone();
        tokio::spawn(async move {
            cancel_token.wait().await;
            let _ = worker_for_cancel
                .cancel_request(&request_id_for_cancel)
                .await;
        });

        // Wire progress: forward stt.progress notifications for this id.
        let last_emit = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let last_persisted = Arc::new(parking_lot::Mutex::new(0.0f32));
        let app_for_prog = self.app.clone();
        let job_for_prog = job_id.clone();
        let project_for_prog = project_id.clone();
        let db_for_prog = self.db.clone();
        let request_id_for_prog = request_id.clone();
        let sub = self.worker.subscribe(
            "stt.progress",
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
                    stage: JobStage::Transcribe,
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

        // Fire the request on a background task; the command returns
        // the initial JobSnapshot immediately.
        let this = self.clone();
        let jobs_for_dereg = self.jobs.clone();
        let job_id_bg = job_id.clone();
        let project_id_bg = project_id.clone();
        let project_root_bg = project_root.clone();
        let audio_hash_bg = audio_hash.clone();
        let cache_key_bg = cache_key.clone();
        let options_bg = options.clone();
        tokio::spawn(async move {
            let params = json!({
                "audioPath": audio_abs,
                "audioHash": audio_hash_bg,
                "options": _options_to_wire(&options_bg),
            });
            let result = this
                .worker
                .request_no_timeout_with_id(&request_id, "stt.transcribe", params)
                .await;
            this.worker.unsubscribe(sub);
            let ctx = TranscribeFinalizeContext {
                job_id: job_id_bg.clone(),
                project_id: project_id_bg,
                project_root: project_root_bg,
                audio_hash: audio_hash_bg,
                cache_key: cache_key_bg,
                options: options_bg,
            };
            this.finalize_transcribe(ctx, result).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(TranscribeStart::Started(snap))
    }

    async fn finalize_transcribe(
        self: Arc<Self>,
        ctx: TranscribeFinalizeContext,
        result: Result<Value, crate::worker::WorkerError>,
    ) {
        let TranscribeFinalizeContext {
            job_id,
            project_id,
            project_root,
            audio_hash,
            cache_key,
            options,
        } = ctx;
        match result {
            Ok(value) => {
                match parse_transcribe_response(value, &audio_hash, &cache_key, &options) {
                    Ok(transcript) => match TranscriptCacheFile::save(&project_root, &transcript) {
                        Ok(_) => {
                            self.finalize_success(&job_id, &project_id, JobStage::Transcribe)
                                .await
                        }
                        Err(err) => {
                            self.finalize_failure(
                                &job_id,
                                &project_id,
                                JobStage::Transcribe,
                                "STT_IO",
                                &err.to_string(),
                            )
                            .await
                        }
                    },
                    Err(err) => {
                        self.finalize_failure(
                            &job_id,
                            &project_id,
                            JobStage::Transcribe,
                            "STT_INVALID_RESPONSE",
                            &err.to_string(),
                        )
                        .await
                    }
                }
            }
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let stt_err = SttError::from_rpc(rpc);
                    if matches!(stt_err, SttError::Cancelled) {
                        self.finalize_cancel(&job_id, &project_id, JobStage::Transcribe)
                            .await;
                    } else {
                        self.finalize_failure(
                            &job_id,
                            &project_id,
                            JobStage::Transcribe,
                            stt_err_code(&stt_err),
                            &stt_err.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        &job_id,
                        &project_id,
                        JobStage::Transcribe,
                        "STT_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    // -------------------------------------------------------- job finalization

    async fn finalize_success(&self, job_id: &str, project_id: &str, stage: JobStage) {
        // Phase 12 bug-fix — download jobs are in-memory only (see
        // `download_model`), so there's no row to update. Skip the
        // DB write entirely when we don't have a real project id.
        if !project_id.is_empty() {
            let db = self.db.clone();
            let jid = job_id.to_string();
            let _ = db
                .run(move |d| {
                    JobsRepo::update_status(d, &jid, JobStatus::Completed, Some(1.0), None, None)
                })
                .await;
        }
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            stage,
            status: JobStatus::Completed,
            progress: 1.0,
            error_code: None,
            error_message: None,
        });
    }

    async fn finalize_cancel(&self, job_id: &str, project_id: &str, stage: JobStage) {
        // Phase 12 bug-fix — see `finalize_success`.
        if !project_id.is_empty() {
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
        }
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            stage,
            status: JobStatus::Cancelled,
            progress: 0.0,
            error_code: Some("CANCELLED".into()),
            error_message: Some("user cancelled".into()),
        });
    }

    async fn finalize_failure(
        &self,
        job_id: &str,
        project_id: &str,
        stage: JobStage,
        code: &str,
        message: &str,
    ) {
        // Phase 12 bug-fix — see `finalize_success`.
        if !project_id.is_empty() {
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
        }
        self.emit_terminal(TerminalEvent {
            job_id: job_id.into(),
            project_id: project_id.into(),
            stage,
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
            stage: evt.stage,
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
}

// ------------------------------------------------------------------- helpers

fn map_project_err(err: crate::projects::ProjectError) -> SttError {
    match err {
        crate::projects::ProjectError::Db(e) => SttError::Db(e),
        other => SttError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

pub fn build_cache_key(audio_hash: &str, options: &SttOptions) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let parts = [
        "v1".to_string(),
        audio_hash.to_string(),
        options.model.clone(),
        options
            .language
            .as_deref()
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "auto".into()),
        options.device.clone().unwrap_or_else(|| "".into()),
        options.compute_type.clone().unwrap_or_else(|| "".into()),
        format!("beam={}", options.beam_size),
        format!("words={}", if options.word_timestamps { 1 } else { 0 }),
        format!("vad={}", if options.vad_filter { 1 } else { 0 }),
        format!("temp={:.4}", options.temperature),
        format!("prompt={}", options.initial_prompt.as_deref().unwrap_or("")),
    ];
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn _options_to_wire(options: &SttOptions) -> Value {
    json!({
        "model": options.model,
        "language": options.language,
        "device": options.device,
        "computeType": options.compute_type,
        "beamSize": options.beam_size,
        "wordTimestamps": options.word_timestamps,
        "vadFilter": options.vad_filter,
        "initialPrompt": options.initial_prompt,
        "temperature": options.temperature,
    })
}

fn parse_transcribe_response(
    value: Value,
    audio_hash: &str,
    cache_key: &str,
    options: &SttOptions,
) -> Result<Transcript, serde_json::Error> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WireSegment {
        id: u32,
        start: f64,
        end: f64,
        text: String,
        #[serde(default)]
        avg_logprob: Option<f64>,
        #[serde(default)]
        no_speech_prob: Option<f64>,
        #[serde(default)]
        words: Option<Vec<WireWord>>,
    }
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WireWord {
        word: String,
        start: f64,
        end: f64,
        #[serde(default)]
        probability: Option<f64>,
    }
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WireResp {
        language: String,
        segments: Vec<WireSegment>,
        #[serde(default)]
        cache_key: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        device: Option<String>,
        #[serde(default)]
        compute_type: Option<String>,
        #[serde(default)]
        word_timestamps: Option<bool>,
        #[serde(default)]
        options: Option<Value>,
    }
    let wire: WireResp = serde_json::from_value(value)?;
    let segments = wire
        .segments
        .into_iter()
        .map(|s| TranscribeSegment {
            id: s.id,
            start: s.start,
            end: s.end,
            text: s.text,
            avg_logprob: s.avg_logprob,
            no_speech_prob: s.no_speech_prob,
            words: s.words.map(|words| {
                words
                    .into_iter()
                    .map(|w| WhisperWord {
                        word: w.word,
                        start: w.start,
                        end: w.end,
                        probability: w.probability,
                    })
                    .collect()
            }),
        })
        .collect::<Vec<_>>();

    let duration = segments.last().map(|s| s.end).unwrap_or(0.0);
    Ok(Transcript {
        version: 1,
        language: wire.language,
        segments,
        model: wire.model.unwrap_or_else(|| options.model.clone()),
        device: wire
            .device
            .unwrap_or_else(|| options.device.clone().unwrap_or_default()),
        compute_type: wire
            .compute_type
            .unwrap_or_else(|| options.compute_type.clone().unwrap_or_default()),
        word_timestamps: wire.word_timestamps.unwrap_or(options.word_timestamps),
        audio: TranscriptAudio {
            path: AUDIO_INPUT_REL.into(),
            hash: audio_hash.to_string(),
        },
        duration_secs: duration,
        cache_key: wire.cache_key.unwrap_or_else(|| cache_key.to_string()),
        created_at: Utc::now(),
        provider: "faster-whisper".into(),
        options: wire.options.unwrap_or_else(|| json!({})),
    })
}

fn stt_err_code(err: &SttError) -> &'static str {
    match err {
        SttError::NoSourceMedia => "STT_NO_SOURCE_MEDIA",
        SttError::AudioNotExtracted => "STT_AUDIO_NOT_EXTRACTED",
        SttError::UnknownModel { .. } => "STT_UNKNOWN_MODEL",
        SttError::ModelNotInstalled { .. } => "STT_MODEL_NOT_INSTALLED",
        SttError::WhisperNotInstalled => "STT_WHISPER_NOT_INSTALLED",
        SttError::ModelLoadFailed { .. } => "STT_MODEL_LOAD_FAILED",
        SttError::InvalidAudio { .. } => "STT_INVALID_AUDIO",
        SttError::OutOfMemory => "STT_OUT_OF_MEMORY",
        SttError::WorkerCrash => "STT_WORKER_CRASH",
        SttError::Cancelled => "STT_CANCELLED",
        SttError::DownloadFailed { .. } => "STT_DOWNLOAD_FAILED",
        SttError::Worker(_) => "STT_WORKER_ERROR",
        SttError::Registry(_) => "STT_REGISTRY",
        SttError::Db(_) => "STT_DB",
        SttError::Io { .. } => "STT_IO",
    }
}
