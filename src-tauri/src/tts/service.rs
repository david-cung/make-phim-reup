//! End-to-end TTS orchestrator (Phase 6).
//!
//! Flow of :meth:`TtsService::generate`:
//!
//! 1. Load the project + `subtitles.json`. Refuse if no subtitles.
//! 2. Load `voices/voices.json` (if any) as the current cache.
//! 3. For each subtitle segment, compute its expected cache key from
//!    ``(engine, voice_id, model_name, translated_text, settings)``.
//!    Compare against the manifest to build the "todo" list per the
//!    requested `GenerateMode`.
//! 4. If nothing needs generating → return `UpToDate` synchronously.
//! 5. Otherwise register a `Tts` job, spawn a background task that:
//!      * subscribes to `tts.segment_completed` and persists each
//!        entry into `voices.json` incrementally;
//!      * subscribes to `tts.progress` for coarse UI progress;
//!      * forwards the job cancel token to the worker via
//!        `jsonrpc://cancel`;
//!      * awaits the worker response, clears the subtitle `dirty.tts`
//!        flag when the whole doc is covered, and emits the terminal
//!        `job://update`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::db::DbHandle;
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::projects::ProjectService;
use crate::subtitles::{SubtitleCacheFile, SubtitleDoc, SubtitleService};
use crate::worker::WorkerSupervisor;

use super::cache::{voices_dir, TtsCacheFile, VOICES_RELATIVE, VOICES_SUBDIR};
use super::errors::TtsError;
use super::models::{
    build_segment_cache_key, entry_matches, text_hash, voice_file_relative, GenerateMode,
    GenerateRequest, PreviewResult, TtsEnv, TtsGenerateStart, TtsManifest, TtsSegmentEntry,
    TtsSettings, TtsSummary, VoiceInfo,
};

const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

pub struct TtsService {
    app: AppHandle,
    db: DbHandle,
    projects: Arc<ProjectService>,
    subtitles: Arc<SubtitleService>,
    jobs: Arc<JobRegistry>,
    worker: Arc<WorkerSupervisor>,
    manifest_lock: Arc<parking_lot::Mutex<()>>,
}

struct FinalizeContext {
    job_id: String,
    project_id: String,
    project_root: PathBuf,
}

struct TerminalEvent {
    job_id: String,
    project_id: String,
    stage: JobStage,
    status: JobStatus,
    progress: f32,
    error_code: Option<String>,
    error_message: Option<String>,
}

/// One segment queued for the worker along with the cache identity
/// the host expects the worker to stamp on the returned manifest entry.
struct TodoEntry {
    segment_id: u32,
    text: String,
    voice_id: String,
    settings: TtsSettings,
}

impl TtsService {
    pub fn new(
        app: AppHandle,
        db: DbHandle,
        projects: Arc<ProjectService>,
        subtitles: Arc<SubtitleService>,
        jobs: Arc<JobRegistry>,
        worker: Arc<WorkerSupervisor>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app,
            db,
            projects,
            subtitles,
            jobs,
            worker,
            manifest_lock: Arc::new(parking_lot::Mutex::new(())),
        })
    }

    // -------------------------------------------------- read-only queries

    pub async fn env(self: &Arc<Self>) -> Result<TtsEnv, TtsError> {
        let v = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "tts.env", json!({}))
            .await
            .map_err(TtsError::Worker)?;
        serde_json::from_value(v).map_err(|e| {
            TtsError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })
    }

    /// Phase 10 — release every TTS engine's runtime state so the
    /// worker's resident memory footprint drops between long idle
    /// periods. Returns the list of engine ids that reported a
    /// successful unload.
    pub async fn unload_all(self: &Arc<Self>) -> Result<Vec<String>, TtsError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            #[serde(default)]
            released: Vec<String>,
        }
        let v = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "tts.unload", json!({}))
            .await
            .map_err(TtsError::Worker)?;
        let r: Resp = serde_json::from_value(v).map_err(|e| {
            TtsError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(r.released)
    }

    pub async fn list_voices(self: &Arc<Self>) -> Result<Vec<VoiceInfo>, TtsError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            voices: Vec<VoiceInfo>,
        }
        let v = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "tts.list_voices", json!({}))
            .await
            .map_err(TtsError::Worker)?;
        let r: Resp = serde_json::from_value(v).map_err(|e| {
            TtsError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(r.voices)
    }

    pub async fn get_manifest(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<TtsManifest>, TtsError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        TtsCacheFile::load(&root).map_err(|source| TtsError::Io {
            path: root.display().to_string(),
            source,
        })
    }

    /// Compute a compact summary comparing the manifest with the current
    /// subtitle doc so the UI can display "42/350 generated".
    pub async fn get_summary(
        self: &Arc<Self>,
        project_id: String,
        request: SummaryRequest,
    ) -> Result<Option<TtsSummary>, TtsError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        let Some(doc) = SubtitleCacheFile::load(&root).map_err(|source| TtsError::Io {
            path: root.display().to_string(),
            source,
        })?
        else {
            return Ok(None);
        };
        let manifest = TtsCacheFile::load(&root).map_err(|source| TtsError::Io {
            path: root.display().to_string(),
            source,
        })?;
        Ok(Some(build_summary(&doc, manifest.as_ref(), &request)))
    }

    // ---------------------------------------------------------- preview

    /// Synthesise a single segment on demand for the "Preview Voice"
    /// button. Returns the cached path immediately when the cache
    /// identity already matches — no regeneration.
    pub async fn preview_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
        voice_id: Option<String>,
        settings: Option<TtsSettings>,
        engine: Option<String>,
    ) -> Result<PreviewResult, TtsError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        let doc = SubtitleCacheFile::load(&root)
            .map_err(|source| TtsError::Io {
                path: root.display().to_string(),
                source,
            })?
            .ok_or(TtsError::NoSubtitles)?;
        let seg = doc
            .segments
            .iter()
            .find(|s| s.id == segment_id)
            .ok_or(TtsError::SegmentNotFound { segment_id })?;

        let text = pick_text_for_synthesis(seg);
        if text.trim().is_empty() {
            return Err(TtsError::InvalidText);
        }

        let voice_id =
            voice_id
                .or_else(|| seg.voice_id.clone())
                .ok_or_else(|| TtsError::VoiceMissing {
                    engine: engine.clone().unwrap_or_else(|| "piper".into()),
                    voice_id: String::new(),
                })?;
        let engine_name = engine.unwrap_or_else(|| "piper".into());
        let effective_settings = settings.unwrap_or_default().normalised();

        // Cache check: if the manifest already has a matching entry
        // and the file is present, return it directly.
        let manifest = TtsCacheFile::load(&root).map_err(|source| TtsError::Io {
            path: root.display().to_string(),
            source,
        })?;
        if let Some(m) = manifest.as_ref() {
            if let Some(entry) = m.find(segment_id) {
                let abs = root.join(&entry.file);
                if abs.exists() {
                    // We don't know the voice's model_name up-front on
                    // the Rust side without another worker roundtrip;
                    // just trust the manifest's cache key here.
                    let text_h = text_hash(&text);
                    if entry.text_hash == text_h
                        && entry.voice_id == voice_id
                        && entry.engine == engine_name
                        && (effective_settings.speed - entry.speed).abs() < 1e-4
                        && (effective_settings.pitch - entry.pitch).abs() < 1e-4
                        && (effective_settings.volume - entry.volume).abs() < 1e-4
                    {
                        return Ok(PreviewResult {
                            segment_id,
                            engine: entry.engine.clone(),
                            voice_id: entry.voice_id.clone(),
                            absolute_path: abs.display().to_string(),
                            relative_path: entry.file.clone(),
                            duration_secs: entry.duration_secs,
                            cache_hit: true,
                        });
                    }
                }
            }
        }

        // Miss → synthesise via the worker.
        let rel = voice_file_relative(segment_id);
        let abs = root.join(&rel);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent).map_err(|source| TtsError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }
        let params = json!({
            "segmentId": segment_id,
            "engine": engine_name,
            "voiceId": voice_id,
            "text": text,
            "outputPath": abs.display().to_string(),
            "settings": settings_to_wire(&effective_settings),
        });
        let request_id = self.worker.new_request_id();
        let v = self
            .worker
            .request_no_timeout_with_id(&request_id, "tts.synthesize_one", params)
            .await
            .map_err(|e| match e {
                crate::worker::WorkerError::Rpc(rpc) => TtsError::from_rpc(rpc),
                other => TtsError::Worker(other),
            })?;

        let entry = entry_from_worker_response(&v, segment_id)?;
        // Persist into the manifest so future calls hit the cache.
        self.upsert_manifest(&root, entry.clone(), &engine_name, &voice_id)?;

        Ok(PreviewResult {
            segment_id,
            engine: entry.engine,
            voice_id: entry.voice_id,
            absolute_path: abs.display().to_string(),
            relative_path: entry.file,
            duration_secs: entry.duration_secs,
            cache_hit: false,
        })
    }

    // ------------------------------------------------------- generate batch

    pub async fn generate(
        self: &Arc<Self>,
        project_id: String,
        request: GenerateRequest,
    ) -> Result<TtsGenerateStart, TtsError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);

        let doc = SubtitleCacheFile::load(&project_root)
            .map_err(|source| TtsError::Io {
                path: project_root.display().to_string(),
                source,
            })?
            .ok_or(TtsError::NoSubtitles)?;

        if request.default_voice_id.trim().is_empty() {
            return Err(TtsError::VoiceMissing {
                engine: request.engine.clone(),
                voice_id: String::new(),
            });
        }

        // Ensure voices/ exists so tmp writes don't race directory creation.
        let _ = std::fs::create_dir_all(voices_dir(&project_root));

        let manifest = TtsCacheFile::load(&project_root).map_err(|source| TtsError::Io {
            path: project_root.display().to_string(),
            source,
        })?;

        // Build the todo list per the requested mode.
        let todo = plan_generation(&doc, manifest.as_ref(), &request);

        if todo.is_empty() {
            let summary = build_summary(
                &doc,
                manifest.as_ref(),
                &SummaryRequest {
                    engine: request.engine.clone(),
                    default_voice_id: request.default_voice_id.clone(),
                    settings: request.settings,
                },
            );
            let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
            return Ok(TtsGenerateStart::UpToDate { summary });
        }

        // Seed the manifest so a mid-run crash still leaves a valid file.
        let seed_manifest = manifest.unwrap_or_else(|| {
            TtsManifest::empty(request.engine.clone(), request.default_voice_id.clone())
        });
        {
            let _guard = self.manifest_lock.lock();
            let mut m = seed_manifest.clone();
            m.engine = request.engine.clone();
            m.default_voice_id = request.default_voice_id.clone();
            m.updated_at = Utc::now();
            TtsCacheFile::save(&project_root, &m).map_err(|source| TtsError::Io {
                path: project_root.display().to_string(),
                source,
            })?;
        }

        let segments_wire = todo
            .iter()
            .map(|t| {
                json!({
                    "id": t.segment_id,
                    "text": t.text,
                    "voiceId": t.voice_id,
                    "settings": settings_to_wire(&t.settings),
                })
            })
            .collect::<Vec<_>>();

        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), project_id.clone(), JobStage::Tts)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Tts,
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
            "tts.progress",
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
                    stage: JobStage::Tts,
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

        // Wire per-segment completion → persist into voices.json.
        let manifest_lock = self.manifest_lock.clone();
        let project_root_for_seg = project_root.clone();
        let app_for_seg = self.app.clone();
        let job_for_seg = job_id.clone();
        let project_for_seg = project_id.clone();
        let request_id_for_seg = request_id.clone();
        let engine_for_seg = request.engine.clone();
        let default_voice_for_seg = request.default_voice_id.clone();
        let segment_sub = self.worker.subscribe(
            "tts.segment_completed",
            Arc::new(move |_, params| {
                let Some(target) = params.get("requestId").and_then(|v| v.as_str()) else {
                    return;
                };
                if target != request_id_for_seg {
                    return;
                }
                let Ok(entry) = entry_from_worker_response(params, 0) else {
                    tracing::warn!("tts.segment_completed with invalid payload");
                    return;
                };
                let root = project_root_for_seg.clone();
                let lock = manifest_lock.clone();
                let app = app_for_seg.clone();
                let job_id = job_for_seg.clone();
                let project_id = project_for_seg.clone();
                let engine = engine_for_seg.clone();
                let default_voice = default_voice_for_seg.clone();
                tokio::spawn(async move {
                    let (segment_id, file_rel, generated_count, subtitle_count) = {
                        let _guard = lock.lock();
                        let mut manifest =
                            TtsCacheFile::load(&root).ok().flatten().unwrap_or_else(|| {
                                TtsManifest::empty(engine.clone(), default_voice.clone())
                            });
                        manifest.engine = engine.clone();
                        if manifest.default_voice_id.trim().is_empty() {
                            manifest.default_voice_id = default_voice.clone();
                        }
                        let segment_id = entry.segment_id;
                        let file = entry.file.clone();
                        manifest.upsert(entry);
                        let generated = manifest.segments.len() as u32;
                        if let Err(err) = TtsCacheFile::save(&root, &manifest) {
                            tracing::warn!(%err, "failed to persist tts manifest");
                        }
                        // Compute subtitle count for the event payload;
                        // best-effort — omit if we can't load it.
                        let count = SubtitleCacheFile::load(&root)
                            .ok()
                            .flatten()
                            .map(|d| d.segments.len() as u32)
                            .unwrap_or(0);
                        (segment_id, file, generated, count)
                    };
                    let payload = json!({
                        "jobId": job_id,
                        "projectId": project_id,
                        "segmentId": segment_id,
                        "file": file_rel,
                        "generatedCount": generated_count,
                        "subtitleCount": subtitle_count,
                    });
                    let _ = app.emit("tts://segment_completed", payload);
                });
            }),
        );

        // Kick off the actual RPC on a background task.
        let this = self.clone();
        let jobs_for_dereg = self.jobs.clone();
        let project_root_bg = project_root.clone();
        let project_id_bg = project_id.clone();
        let job_id_bg = job_id.clone();
        let engine_bg = request.engine.clone();
        let default_voice_bg = request.default_voice_id.clone();
        let settings_bg = request.settings;
        tokio::spawn(async move {
            let params = json!({
                "engine": engine_bg,
                "defaultVoiceId": default_voice_bg,
                "settings": settings_to_wire(&settings_bg),
                "projectRoot": project_root_bg.display().to_string(),
                "voicesSubdir": VOICES_SUBDIR,
                "segments": segments_wire,
            });
            let result = this
                .worker
                .request_no_timeout_with_id(&request_id, "tts.synthesize_batch", params)
                .await;
            this.worker.unsubscribe(progress_sub);
            this.worker.unsubscribe(segment_sub);
            let ctx = FinalizeContext {
                job_id: job_id_bg.clone(),
                project_id: project_id_bg,
                project_root: project_root_bg,
            };
            this.finalize(ctx, result, settings_bg, engine_bg, default_voice_bg)
                .await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(TtsGenerateStart::Started(snap))
    }

    // -------------------------------------------------------- finalisation

    async fn finalize(
        self: Arc<Self>,
        ctx: FinalizeContext,
        result: Result<Value, crate::worker::WorkerError>,
        settings: TtsSettings,
        engine: String,
        default_voice_id: String,
    ) {
        let FinalizeContext {
            job_id,
            project_id,
            project_root,
        } = ctx;
        match result {
            Ok(_) => {
                let _ = self.stamp_completion(&project_root).await;
                // If every subtitle has a matching manifest entry, we
                // can safely clear the subtitles' dirty.tts flag.
                if let Ok(Some(summary)) = self
                    .get_summary(
                        project_id.clone(),
                        SummaryRequest {
                            engine,
                            default_voice_id,
                            settings,
                        },
                    )
                    .await
                {
                    let _ = self.maybe_clear_dirty(project_id.clone(), &summary).await;
                }
                self.finalize_success(&job_id, &project_id, JobStage::Tts)
                    .await;
            }
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let te = TtsError::from_rpc(rpc);
                    if matches!(te, TtsError::Cancelled) {
                        self.finalize_cancel(&job_id, &project_id, JobStage::Tts)
                            .await;
                    } else {
                        self.finalize_failure(
                            &job_id,
                            &project_id,
                            JobStage::Tts,
                            tts_err_code(&te),
                            &te.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        &job_id,
                        &project_id,
                        JobStage::Tts,
                        "TTS_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    async fn stamp_completion(&self, project_root: &std::path::Path) -> std::io::Result<()> {
        let _guard = self.manifest_lock.lock();
        if let Some(mut m) = TtsCacheFile::load(project_root)? {
            m.updated_at = Utc::now();
            TtsCacheFile::save(project_root, &m)?;
        }
        Ok(())
    }

    async fn maybe_clear_dirty(
        self: &Arc<Self>,
        project_id: String,
        summary: &TtsSummary,
    ) -> Result<(), TtsError> {
        if summary.missing_count == 0
            && summary.stale_count == 0
            && summary.subtitle_count > 0
            && summary.generated_count >= summary.subtitle_count
        {
            let _ = self
                .subtitles
                .clear_dirty_flags(project_id, crate::subtitles::DirtyFlags::only_tts())
                .await
                .map_err(|e| tracing::warn!(%e, "failed to clear subtitle dirty.tts flag"));
        }
        Ok(())
    }

    // ---------------------------------------------------- job finalization

    async fn finalize_success(&self, job_id: &str, project_id: &str, stage: JobStage) {
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
            stage,
            status: JobStatus::Completed,
            progress: 1.0,
            error_code: None,
            error_message: None,
        });
    }

    async fn finalize_cancel(&self, job_id: &str, project_id: &str, stage: JobStage) {
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

    // ------------------------------------------------------ manifest helpers

    fn upsert_manifest(
        &self,
        project_root: &Path,
        entry: TtsSegmentEntry,
        engine: &str,
        default_voice_id: &str,
    ) -> Result<(), TtsError> {
        let _guard = self.manifest_lock.lock();
        let mut manifest = TtsCacheFile::load(project_root)
            .map_err(|source| TtsError::Io {
                path: project_root.display().to_string(),
                source,
            })?
            .unwrap_or_else(|| {
                TtsManifest::empty(engine.to_string(), default_voice_id.to_string())
            });
        manifest.engine = engine.to_string();
        if manifest.default_voice_id.trim().is_empty() {
            manifest.default_voice_id = default_voice_id.to_string();
        }
        manifest.upsert(entry);
        TtsCacheFile::save(project_root, &manifest).map_err(|source| TtsError::Io {
            path: project_root.display().to_string(),
            source,
        })?;
        Ok(())
    }
}

// ------------------------------------------------------------------- helpers

/// Passed into `get_summary`/`generate` — every field the summary needs
/// so we don't have to re-derive it from the manifest alone.
#[derive(Debug, Clone)]
pub struct SummaryRequest {
    pub engine: String,
    pub default_voice_id: String,
    pub settings: TtsSettings,
}

fn map_project_err(err: crate::projects::ProjectError) -> TtsError {
    match err {
        crate::projects::ProjectError::Db(e) => TtsError::Db(e),
        other => TtsError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

/// Extract the text we should feed the TTS engine for one subtitle.
/// We prefer `translatedText`; if it's empty, fall back to `sourceText`
/// so the user can preview without a translation yet.
fn pick_text_for_synthesis(seg: &crate::subtitles::SubtitleSegment) -> String {
    let translated = seg.translated_text.trim();
    if !translated.is_empty() {
        translated.to_string()
    } else {
        seg.source_text.trim().to_string()
    }
}

fn settings_to_wire(s: &TtsSettings) -> Value {
    let n = s.normalised();
    json!({
        "speed": n.speed,
        "pitch": n.pitch,
        "volume": n.volume,
    })
}

/// Build a `TtsSegmentEntry` from either a `tts.synthesize_one` reply
/// or a `tts.segment_completed` notification. The two shapes are
/// intentionally identical.
fn entry_from_worker_response(v: &Value, fallback_id: u32) -> Result<TtsSegmentEntry, TtsError> {
    let segment_id = v
        .get("segmentId")
        .and_then(|x| x.as_u64())
        .map(|x| x as u32)
        .unwrap_or(fallback_id);
    let engine = v
        .get("engine")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let voice_id = v
        .get("voiceId")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let model_name = v
        .get("modelName")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let cache_key = v
        .get("cacheKey")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let text_h = v
        .get("textHash")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let text = v
        .get("text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let file = v
        .get("file")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let duration = v
        .get("durationSecs")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0);
    let sample_rate = v.get("sampleRate").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let channels = v.get("channels").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
    let size = v.get("sizeBytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let settings = v.get("settings");
    let speed = settings
        .and_then(|s| s.get("speed"))
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0) as f32;
    let pitch = settings
        .and_then(|s| s.get("pitch"))
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0) as f32;
    let volume = settings
        .and_then(|s| s.get("volume"))
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0) as f32;
    Ok(TtsSegmentEntry {
        segment_id,
        engine,
        voice_id,
        model_name,
        cache_key,
        text_hash: text_h,
        text,
        speed,
        pitch,
        volume,
        file,
        duration_secs: duration,
        sample_rate,
        channels,
        size_bytes: size,
        generated_at: Utc::now(),
    })
}

/// Plan out which segments need synthesising for a given request.
///
/// The plan is authoritative — the worker only synthesises what we ask
/// for, no more. This is where cache-hit filtering happens.
fn plan_generation(
    doc: &SubtitleDoc,
    manifest: Option<&TtsManifest>,
    request: &GenerateRequest,
) -> Vec<TodoEntry> {
    let mut todo: Vec<TodoEntry> = Vec::new();
    let force_ids: Option<std::collections::HashSet<u32>> = match &request.mode {
        GenerateMode::Selected { ids } => Some(ids.iter().copied().collect()),
        _ => None,
    };
    let force_all = matches!(request.mode, GenerateMode::All);
    let settings_default = request.settings.normalised();
    for seg in &doc.segments {
        let text = pick_text_for_synthesis(seg);
        if text.trim().is_empty() {
            continue;
        }
        let voice_id = seg
            .voice_id
            .clone()
            .unwrap_or_else(|| request.default_voice_id.clone());
        if voice_id.trim().is_empty() {
            continue;
        }
        let effective_settings = settings_default; // per-segment overrides are Phase 7+ material.

        let should_generate = match &force_ids {
            Some(set) => set.contains(&seg.id),
            None => {
                if force_all {
                    true
                } else {
                    !manifest
                        .and_then(|m| m.find(seg.id))
                        .map(|entry| {
                            entry.text_hash == text_hash(&text)
                                && entry.engine == request.engine
                                && entry.voice_id == voice_id
                                && (effective_settings.speed - entry.speed).abs() < 1e-4
                                && (effective_settings.pitch - entry.pitch).abs() < 1e-4
                                && (effective_settings.volume - entry.volume).abs() < 1e-4
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
            text,
            voice_id,
            settings: effective_settings,
        });
    }
    todo
}

/// Compute a summary comparing every subtitle to the manifest's entry.
pub fn build_summary(
    doc: &SubtitleDoc,
    manifest: Option<&TtsManifest>,
    request: &SummaryRequest,
) -> TtsSummary {
    let settings = request.settings.normalised();
    let mut generated = 0u32;
    let mut missing = 0u32;
    let mut stale = 0u32;
    let subtitle_count = doc.segments.len() as u32;
    for seg in &doc.segments {
        let text = pick_text_for_synthesis(seg);
        if text.trim().is_empty() {
            missing += 1;
            continue;
        }
        let voice_id = seg
            .voice_id
            .clone()
            .unwrap_or_else(|| request.default_voice_id.clone());
        if voice_id.trim().is_empty() {
            missing += 1;
            continue;
        }
        match manifest.and_then(|m| m.find(seg.id)) {
            Some(entry) => {
                let matches = entry.text_hash == text_hash(&text)
                    && entry.engine == request.engine
                    && entry.voice_id == voice_id
                    && (settings.speed - entry.speed).abs() < 1e-4
                    && (settings.pitch - entry.pitch).abs() < 1e-4
                    && (settings.volume - entry.volume).abs() < 1e-4;
                if matches {
                    generated += 1;
                } else {
                    stale += 1;
                }
            }
            None => missing += 1,
        }
    }
    let updated_at = manifest.map(|m| m.updated_at).unwrap_or_else(Utc::now);
    TtsSummary {
        engine: request.engine.clone(),
        default_voice_id: request.default_voice_id.clone(),
        subtitle_count,
        generated_count: generated,
        missing_count: missing,
        stale_count: stale,
        updated_at,
        relative_path: VOICES_RELATIVE.into(),
    }
}

fn tts_err_code(err: &TtsError) -> &'static str {
    match err {
        TtsError::NoSubtitles => "TTS_NO_SUBTITLES",
        TtsError::EngineUnavailable { .. } => "TTS_ENGINE_UNAVAILABLE",
        TtsError::VoiceMissing { .. } => "TTS_VOICE_MISSING",
        TtsError::ModelInvalid { .. } => "TTS_MODEL_INVALID",
        TtsError::InvalidText => "TTS_INVALID_TEXT",
        TtsError::EngineFailure { .. } => "TTS_ENGINE_FAILURE",
        TtsError::OutOfMemory => "TTS_OUT_OF_MEMORY",
        TtsError::DiskFull => "TTS_DISK_FULL",
        TtsError::WorkerCrash => "TTS_WORKER_CRASH",
        TtsError::Cancelled => "TTS_CANCELLED",
        TtsError::SegmentNotFound { .. } => "TTS_SEGMENT_NOT_FOUND",
        TtsError::Worker(_) => "TTS_WORKER_ERROR",
        TtsError::Registry(_) => "TTS_REGISTRY",
        TtsError::Db(_) => "TTS_DB",
        TtsError::Io { .. } => "TTS_IO",
    }
}

#[allow(dead_code)]
fn _touch(_a: &TtsSegmentEntry, _b: bool) {
    let _ = entry_matches;
    let _ = build_segment_cache_key;
}
