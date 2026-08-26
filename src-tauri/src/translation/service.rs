//! End-to-end local-LLM translation orchestrator.
//!
//! Flow of :meth:`TranslationService::translate`:
//!
//! 1. Load the project + transcript. Refuse if no transcript exists.
//! 2. Derive a cache key from
//!    ``(transcript_cache_key, audio_hash, options)``. The Python
//!    worker does the same math independently to keep both authoritative.
//! 3. Load ``translation/translation.json`` if present. If the cache
//!    key matches AND every segment already has a translation → return
//!    a ``CacheHit`` synchronously.
//! 4. Otherwise register a ``Translate`` job, spawn a background task
//!    that:
//!      * seeds ``translation.json`` on disk with empty translations
//!        (or reuses the resumable partial) so a crash leaves a valid
//!        deliverable behind;
//!      * subscribes to ``translate.chunk_completed`` notifications and
//!        persists each chunk as it arrives;
//!      * subscribes to ``translate.progress`` for coarse job progress;
//!      * forwards the job's cancel token to the worker via
//!        ``jsonrpc://cancel``;
//!      * awaits the worker response, finalises the doc, and emits the
//!        terminal ``job://update``.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::db::DbHandle;
use crate::jobs::{JobProgressEvent, JobRegistry, JobSnapshot, JobStage, JobStatus, JobsRepo};
use crate::projects::ProjectService;
use crate::pronouns::{
    context_to_wire, obvious_pronoun_flags, segment_contexts, upsert_review_flag, PronounCacheFile,
};
use crate::stt::TranscriptCacheFile;
use crate::subtitles::SubtitleCacheFile;
use crate::worker::WorkerSupervisor;

use super::cache::{
    apply_chunk, empty_doc_from, EmptyDocParams, TranslationCacheFile, TRANSLATION_RELATIVE,
};
use super::errors::TranslationError;
use super::models::{
    RecommendedPreset, TranslateOptions, TranslateStart, TranslatedSegment, TranslationDoc,
    TranslationEnv, TranslationModelInfo, TranslationSummary,
};

const PROGRESS_EMIT_MIN_INTERVAL_MS: u128 = 100;

pub struct TranslationService {
    app: AppHandle,
    db: DbHandle,
    projects: Arc<ProjectService>,
    jobs: Arc<JobRegistry>,
    worker: Arc<WorkerSupervisor>,
    doc_lock: Arc<parking_lot::Mutex<()>>,
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

impl TranslationService {
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
            doc_lock: Arc::new(parking_lot::Mutex::new(())),
        })
    }

    // ---------------------------------------------------- read-only queries

    pub async fn env(self: &Arc<Self>) -> Result<TranslationEnv, TranslationError> {
        let v = self
            .worker
            .request_no_timeout_with_id(&self.worker.new_request_id(), "translate.env", json!({}))
            .await
            .map_err(TranslationError::Worker)?;
        serde_json::from_value(v).map_err(|e| {
            TranslationError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })
    }

    pub async fn list_models(
        self: &Arc<Self>,
    ) -> Result<Vec<TranslationModelInfo>, TranslationError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            models: Vec<TranslationModelInfo>,
        }
        let v = self
            .worker
            .request_no_timeout_with_id(
                &self.worker.new_request_id(),
                "translate.list_models",
                json!({}),
            )
            .await
            .map_err(TranslationError::Worker)?;
        let r: Resp = serde_json::from_value(v).map_err(|e| {
            TranslationError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(r.models)
    }

    /// Phase 12 — curated list of GGUF models the app knows how to
    /// auto-download when the user has nothing installed. Sourced
    /// from the Python worker so both sides agree on what's
    /// available.
    pub async fn list_recommended(
        self: &Arc<Self>,
    ) -> Result<Vec<RecommendedPreset>, TranslationError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            presets: Vec<RecommendedPreset>,
        }
        let v = self
            .worker
            .request_no_timeout_with_id(
                &self.worker.new_request_id(),
                "translate.list_recommended",
                json!({}),
            )
            .await
            .map_err(TranslationError::Worker)?;
        let r: Resp = serde_json::from_value(v).map_err(|e| {
            TranslationError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(r.presets)
    }

    // ---------------------------------------------------------- download model

    /// Phase 12 — pull a curated GGUF from HuggingFace into the
    /// translation directory and surface progress via
    /// `job://update` / `job://progress` events. Mirrors
    /// `stt::service::download_model`; both channels are in-memory
    /// only (see FK bug-fix note there).
    pub async fn download_model(
        self: &Arc<Self>,
        preset: String,
    ) -> Result<JobSnapshot, TranslationError> {
        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), String::new(), JobStage::Translate)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: String::new(),
            stage: JobStage::Translate,
            status: JobStatus::Running,
            progress: 0.0,
            error_code: None,
            error_message: None,
            created_at: now,
            started_at: Some(now),
            completed_at: None,
        };
        // In-memory only — cannot persist because `jobs.project_id`
        // has a NOT-NULL FK. See stt::service::download_model.
        self.emit_update(&snap);

        let request_id = self.worker.new_request_id();
        let cancel = handle.cancel.clone();
        let request_id_for_cancel = request_id.clone();
        let worker_for_cancel = self.worker.clone();
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
            "translate.download_progress",
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
                    stage: JobStage::Translate,
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
                    "translate.download_model",
                    json!({ "preset": preset }),
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
            Ok(_) => self.finalize_success(job_id, "", JobStage::Translate).await,
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let t_err = TranslationError::from_rpc(rpc);
                    if matches!(t_err, TranslationError::Cancelled) {
                        self.finalize_cancel(job_id, "", JobStage::Translate).await;
                    } else {
                        let code = translation_err_code(&t_err);
                        self.finalize_failure(
                            job_id,
                            "",
                            JobStage::Translate,
                            code,
                            &t_err.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        job_id,
                        "",
                        JobStage::Translate,
                        "TRANSLATE_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    /// Phase 11 — release the resident GGUF context inside the Python
    /// worker so multi-GB of RAM is returned to the OS between long
    /// idle periods. Returns whether a model was actually released.
    pub async fn unload(self: &Arc<Self>) -> Result<bool, TranslationError> {
        #[derive(serde::Deserialize)]
        struct Resp {
            #[serde(default)]
            released: bool,
        }
        let v = self
            .worker
            .request_no_timeout_with_id(
                &self.worker.new_request_id(),
                "translate.unload",
                json!({}),
            )
            .await
            .map_err(TranslationError::Worker)?;
        let r: Resp = serde_json::from_value(v).map_err(|e| {
            TranslationError::Worker(crate::worker::WorkerError::Protocol { msg: e.to_string() })
        })?;
        Ok(r.released)
    }

    pub async fn get_translation_summary(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<TranslationSummary>, TranslationError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        match TranslationCacheFile::load(&root).map_err(|source| TranslationError::Io {
            path: root.display().to_string(),
            source,
        })? {
            Some(doc) => Ok(Some(TranslationSummary::from_doc(
                &doc,
                TRANSLATION_RELATIVE,
            ))),
            None => Ok(None),
        }
    }

    pub async fn get_translation_doc(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<TranslationDoc>, TranslationError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        TranslationCacheFile::load(&root).map_err(|source| TranslationError::Io {
            path: root.display().to_string(),
            source,
        })
    }

    /// Overwrite a single segment's translation from the manual editor.
    /// Marks the segment ``edited = true`` so the LLM won't clobber it
    /// on the next translate run.
    pub async fn update_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
        translation: String,
    ) -> Result<TranslationSummary, TranslationError> {
        let rec = self
            .projects
            .open(project_id)
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        let doc_lock = self.doc_lock.clone();
        let _guard = doc_lock.lock();
        let mut doc = TranslationCacheFile::load(&root)
            .map_err(|source| TranslationError::Io {
                path: root.display().to_string(),
                source,
            })?
            .ok_or(TranslationError::NoTranscript)?;
        let found = doc.segments.iter_mut().find(|s| s.id == segment_id);
        let Some(seg) = found else {
            return Err(TranslationError::Io {
                path: root.display().to_string(),
                source: std::io::Error::other(format!(
                    "segment id {segment_id} not found in translation.json"
                )),
            });
        };
        seg.translation = translation;
        seg.edited = true;
        doc.updated_at = Utc::now();
        TranslationCacheFile::save(&root, &doc).map_err(|source| TranslationError::Io {
            path: root.display().to_string(),
            source,
        })?;
        Ok(TranslationSummary::from_doc(&doc, TRANSLATION_RELATIVE))
    }

    // -------------------------------------------------------------- translate

    pub async fn translate(
        self: &Arc<Self>,
        project_id: String,
        options: TranslateOptions,
    ) -> Result<TranslateStart, TranslationError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let project_root = PathBuf::from(&rec.root_path);

        // Need a transcript to translate anything.
        let transcript = TranscriptCacheFile::load(&project_root)
            .map_err(|source| TranslationError::Io {
                path: project_root.display().to_string(),
                source,
            })?
            .ok_or(TranslationError::NoTranscript)?;

        if options.model.trim().is_empty() {
            return Err(TranslationError::ModelNotInstalled {
                name: options.model.clone(),
            });
        }

        let transcript_key = transcript.cache_key.clone();
        let audio_hash = transcript.audio.hash.clone();
        let cache_key = build_cache_key(&transcript_key, &audio_hash, &options);

        // Cache lookup — only a *complete* file counts as a hit; a
        // partial one is a resumable state we'll pick up below.
        let existing_doc =
            TranslationCacheFile::load(&project_root).map_err(|source| TranslationError::Io {
                path: project_root.display().to_string(),
                source,
            })?;
        if let Some(doc) = existing_doc.as_ref() {
            if TranslationCacheFile::is_complete_hit(doc, &cache_key, &transcript_key) {
                let summary = TranslationSummary::from_doc(doc, TRANSLATION_RELATIVE);
                let abs = project_root.join(TRANSLATION_RELATIVE);
                return Ok(TranslateStart::CacheHit {
                    summary,
                    absolute_path: abs.display().to_string(),
                });
            }
        }

        // Build the on-disk doc we'll incrementally persist to. Reuse
        // the existing one if its cache_key matches; otherwise start
        // fresh (preserving hand-edits carried over via the frontend
        // is a Phase 5+ concern — for now, changing options resets
        // untouched segments but keeps the edited ones).
        let seed_doc = build_or_reuse_doc(
            &project_root,
            existing_doc.as_ref(),
            &transcript,
            &options,
            &cache_key,
            &transcript_key,
            &audio_hash,
        );
        TranslationCacheFile::save(&project_root, &seed_doc).map_err(|source| {
            TranslationError::Io {
                path: project_root.display().to_string(),
                source,
            }
        })?;

        let pronoun_doc = PronounCacheFile::load(&project_root).unwrap_or_else(|err| {
            tracing::warn!(%err, "failed to load pronoun context; translating without it");
            crate::pronouns::PronounContextDoc::empty()
        });
        let subtitle_doc = SubtitleCacheFile::load(&project_root).ok().flatten();
        let pronoun_contexts = segment_contexts(&pronoun_doc, subtitle_doc.as_ref());

        // Build the request payload for the worker.
        let segments_wire = transcript
            .segments
            .iter()
            .map(|s| {
                let pronoun_context = pronoun_contexts
                    .get(&s.id)
                    .map(|ctx| context_to_wire(ctx, &pronoun_doc));
                json!({
                    "id": s.id,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "speakerId": s.speaker_id,
                    "speakerConfidence": s.speaker_confidence,
                    "pronounContext": pronoun_context,
                })
            })
            .collect::<Vec<_>>();
        let existing_wire = seed_doc
            .segments
            .iter()
            .filter(|s| !s.translation.trim().is_empty())
            .map(|s| json!({ "id": s.id, "translation": s.translation, "edited": s.edited }))
            .collect::<Vec<_>>();

        let job_id = format!("job_{}", Uuid::new_v4().simple());
        let handle = self
            .jobs
            .register(job_id.clone(), project_id.clone(), JobStage::Translate)?;
        let now = Utc::now();
        let snap = JobSnapshot {
            id: job_id.clone(),
            project_id: project_id.clone(),
            stage: JobStage::Translate,
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
        let request_id_for_cancel = request_id.clone();
        let worker_for_cancel = self.worker.clone();
        let cancel_token = handle.cancel.clone();
        tokio::spawn(async move {
            cancel_token.wait().await;
            let _ = worker_for_cancel
                .cancel_request(&request_id_for_cancel)
                .await;
        });

        // Wire progress: pump coarse fractions into `job://progress`.
        let last_emit = Arc::new(parking_lot::Mutex::new(Instant::now()));
        let last_persisted = Arc::new(parking_lot::Mutex::new(0.0f32));
        let app_for_prog = self.app.clone();
        let job_for_prog = job_id.clone();
        let project_for_prog = project_id.clone();
        let db_for_prog = self.db.clone();
        let request_id_for_prog = request_id.clone();
        let progress_sub = self.worker.subscribe(
            "translate.progress",
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
                    stage: JobStage::Translate,
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

        // Wire chunk_completed: persist each chunk to disk and refresh
        // the currentMedia snapshot on the frontend so the editor
        // renders live.
        let doc_lock = self.doc_lock.clone();
        let project_root_for_chunks = project_root.clone();
        let pronoun_doc_for_chunks = pronoun_doc.clone();
        let pronoun_contexts_for_chunks = pronoun_contexts.clone();
        let app_for_chunks = self.app.clone();
        let job_for_chunks = job_id.clone();
        let project_for_chunks = project_id.clone();
        let request_id_for_chunks = request_id.clone();
        let chunk_sub = self.worker.subscribe(
            "translate.chunk_completed",
            Arc::new(move |_, params| {
                let Some(target) = params.get("requestId").and_then(|v| v.as_str()) else {
                    return;
                };
                if target != request_id_for_chunks {
                    return;
                }
                let updates: Vec<(u32, String, serde_json::Value)> = params
                    .get("translations")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| {
                                let id = item.get("id").and_then(|v| v.as_u64())? as u32;
                                let t =
                                    item.get("translation").and_then(|v| v.as_str())?.to_string();
                                let metadata = item
                                    .get("metadata")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!({}));
                                Some((id, t, metadata))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if updates.is_empty() {
                    return;
                }
                let root = project_root_for_chunks.clone();
                let app = app_for_chunks.clone();
                let job_id = job_for_chunks.clone();
                let project_id = project_for_chunks.clone();
                let doc_lock = doc_lock.clone();
                let pronoun_doc = pronoun_doc_for_chunks.clone();
                let pronoun_contexts = pronoun_contexts_for_chunks.clone();
                tokio::spawn(async move {
                    let _guard = doc_lock.lock();
                    let doc = match TranslationCacheFile::load(&root) {
                        Ok(Some(d)) => Some(d),
                        _ => None,
                    };
                    let Some(mut doc) = doc else {
                        tracing::warn!(
                            "chunk_completed arrived but translation.json missing"
                        );
                        return;
                    };
                    let hits = apply_chunk(&mut doc, &updates);
                    if hits == 0 {
                        return;
                    }
                    if let Err(err) = TranslationCacheFile::save(&root, &doc) {
                        tracing::warn!(%err, "failed to persist chunk update");
                        return;
                    }
                    let pronoun_updates = updates
                        .iter()
                        .map(|(id, text, _)| (*id, text.clone()))
                        .collect::<Vec<_>>();
                    persist_pronoun_review_flags(
                        &root,
                        &pronoun_doc,
                        &pronoun_contexts,
                        &pronoun_updates,
                    );
                    // Nudge the frontend so the editor can re-fetch.
                    let payload = json!({
                        "jobId": job_id,
                        "projectId": project_id,
                        "translatedCount": doc.segments.iter().filter(|s| !s.translation.trim().is_empty()).count(),
                        "segmentCount": doc.segments.len(),
                    });
                    let _ = app.emit("translation://chunk_completed", payload);
                });
            }),
        );

        // Kick off the actual RPC on a background task; return the
        // JobSnapshot immediately so the UI has something to render.
        let this = self.clone();
        let jobs_for_dereg = self.jobs.clone();
        let project_root_bg = project_root.clone();
        let project_id_bg = project_id.clone();
        let job_id_bg = job_id.clone();
        let options_bg = options.clone();
        let transcript_key_bg = transcript_key.clone();
        let audio_hash_bg = audio_hash.clone();
        tokio::spawn(async move {
            let params = json!({
                "transcriptCacheKey": transcript_key_bg,
                "audioHash": audio_hash_bg,
                "segments": segments_wire,
                "existingTranslations": existing_wire,
                "options": options_to_wire(&options_bg),
            });
            let result = this
                .worker
                .request_no_timeout_with_id(&request_id, "translate.translate", params)
                .await;
            this.worker.unsubscribe(progress_sub);
            this.worker.unsubscribe(chunk_sub);
            let ctx = FinalizeContext {
                job_id: job_id_bg.clone(),
                project_id: project_id_bg,
                project_root: project_root_bg,
            };
            this.finalize(ctx, result).await;
            jobs_for_dereg.deregister(&job_id_bg);
        });

        Ok(TranslateStart::Started(snap))
    }

    async fn finalize(
        self: Arc<Self>,
        ctx: FinalizeContext,
        result: Result<Value, crate::worker::WorkerError>,
    ) {
        let FinalizeContext {
            job_id,
            project_id,
            project_root,
        } = ctx;
        match result {
            Ok(_) => {
                // Refresh updated_at + one final atomic write so the
                // manifest reflects the completion moment even if the
                // last chunk arrived just before us.
                let _ = self.stamp_completion(&project_root).await;
                self.finalize_success(&job_id, &project_id, JobStage::Translate)
                    .await;
            }
            Err(err) => match err {
                crate::worker::WorkerError::Rpc(rpc) => {
                    let te = TranslationError::from_rpc(rpc);
                    if matches!(te, TranslationError::Cancelled) {
                        self.finalize_cancel(&job_id, &project_id, JobStage::Translate)
                            .await;
                    } else {
                        self.finalize_failure(
                            &job_id,
                            &project_id,
                            JobStage::Translate,
                            translation_err_code(&te),
                            &te.to_string(),
                        )
                        .await;
                    }
                }
                _ => {
                    self.finalize_failure(
                        &job_id,
                        &project_id,
                        JobStage::Translate,
                        "TRANSLATE_WORKER_ERROR",
                        &err.to_string(),
                    )
                    .await;
                }
            },
        }
    }

    async fn stamp_completion(&self, project_root: &std::path::Path) -> std::io::Result<()> {
        let _guard = self.doc_lock.lock();
        if let Some(mut doc) = TranslationCacheFile::load(project_root)? {
            doc.updated_at = Utc::now();
            TranslationCacheFile::save(project_root, &doc)?;
        }
        Ok(())
    }

    // ---------------------------------------------------- job finalization

    async fn finalize_success(&self, job_id: &str, project_id: &str, stage: JobStage) {
        // Phase 12 bug-fix (mirrors stt::service) — download jobs
        // live in-memory only because `jobs.project_id` has a
        // NOT-NULL FK to `projects(id)`. Skip the DB update when we
        // don't have a real project id, otherwise `update_status`
        // returns `NotFound` for a row that was never inserted.
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

fn map_project_err(err: crate::projects::ProjectError) -> TranslationError {
    match err {
        crate::projects::ProjectError::Db(e) => TranslationError::Db(e),
        other => TranslationError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

pub fn build_cache_key(
    transcript_cache_key: &str,
    audio_hash: &str,
    options: &TranslateOptions,
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let parts = [
        "translation_v1".to_string(),
        transcript_cache_key.to_string(),
        audio_hash.to_string(),
        options.source_language.to_ascii_lowercase(),
        options.target_language.to_ascii_lowercase(),
        options.model.clone(),
        options.prompt_version.clone(),
        format!("chunk={}", options.chunk_size),
        format!("before={}", options.context_before),
        format!("after={}", options.context_after),
        format!("retry_before={}", options.retry_context_before),
        format!("retry_after={}", options.retry_context_after),
        format!("max_retries={}", options.max_translation_retries),
        format!("low_conf={:.4}", options.low_confidence_threshold),
        format!("temp={:.4}", options.temperature),
        format!("top_p={:.4}", options.top_p),
        format!("max_tokens={}", options.max_tokens),
    ];
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn options_to_wire(options: &TranslateOptions) -> Value {
    json!({
        "model": options.model,
        "sourceLanguage": options.source_language,
        "targetLanguage": options.target_language,
        "promptVersion": options.prompt_version,
        "chunkSize": options.chunk_size,
        "contextBefore": options.context_before,
        "contextAfter": options.context_after,
        "retryContextBefore": options.retry_context_before,
        "retryContextAfter": options.retry_context_after,
        "maxTranslationRetries": options.max_translation_retries,
        "lowConfidenceThreshold": options.low_confidence_threshold,
        "temperature": options.temperature,
        "topP": options.top_p,
        "maxTokens": options.max_tokens,
    })
}

fn persist_pronoun_review_flags(
    project_root: &std::path::Path,
    base_doc: &crate::pronouns::PronounContextDoc,
    contexts: &std::collections::BTreeMap<u32, crate::pronouns::SegmentPronounContext>,
    updates: &[(u32, String)],
) {
    let mut doc = match PronounCacheFile::load(project_root) {
        Ok(existing) => existing,
        Err(_) => base_doc.clone(),
    };
    let mut changed = false;
    for (segment_id, text) in updates {
        let Some(ctx) = contexts.get(segment_id) else {
            continue;
        };
        let mut flags = ctx.flags.clone();
        if let Some(rule) = ctx.rule.as_ref() {
            flags.extend(obvious_pronoun_flags(text, rule));
        }
        if flags.is_empty() {
            continue;
        }
        upsert_review_flag(&mut doc, *segment_id, flags, Some(ctx));
        changed = true;
    }
    if changed {
        if let Err(err) = PronounCacheFile::save(project_root, &doc) {
            tracing::warn!(%err, "failed to persist pronoun review flags");
        }
    }
}

fn build_or_reuse_doc(
    _project_root: &std::path::Path,
    existing: Option<&TranslationDoc>,
    transcript: &crate::stt::Transcript,
    options: &TranslateOptions,
    cache_key: &str,
    transcript_key: &str,
    audio_hash: &str,
) -> TranslationDoc {
    let segments = transcript
        .segments
        .iter()
        .map(|s| (s.id, s.text.clone(), s.start, s.end))
        .collect::<Vec<_>>();

    // If the existing doc matches our cache identity, preserve its
    // translations (partial resume). If it doesn't match but has
    // user-edited segments, we still keep those to avoid losing manual
    // work — the cache_key stamped on disk becomes the *new* one.
    let mut base = empty_doc_from(
        &segments,
        EmptyDocParams {
            source_language: options.source_language.clone(),
            target_language: options.target_language.clone(),
            model: options.model.clone(),
            prompt_version: options.prompt_version.clone(),
            cache_key: cache_key.to_string(),
            transcript_cache_key: transcript_key.to_string(),
            audio_hash: audio_hash.to_string(),
            options_value: options_to_wire(options),
        },
    );

    if let Some(prev) = existing {
        let resumable = TranslationCacheFile::is_resumable(prev, cache_key, transcript_key);
        for seg in base.segments.iter_mut() {
            let matching = prev.segments.iter().find(|s| s.id == seg.id);
            if let Some(m) = matching {
                if m.edited && !m.translation.trim().is_empty() {
                    // Always preserve user edits regardless of cache identity.
                    seg.translation = m.translation.clone();
                    seg.edited = true;
                } else if resumable && !m.translation.trim().is_empty() {
                    seg.translation = m.translation.clone();
                    seg.edited = m.edited;
                }
            }
        }
        // Preserve the earliest created_at if we're resuming — the
        // "when this translation first started" is more useful than
        // "the moment we hit resume".
        if resumable {
            base.created_at = prev.created_at;
        }
    }
    base
}

fn translation_err_code(err: &TranslationError) -> &'static str {
    match err {
        TranslationError::NoTranscript => "TRANSLATE_NO_TRANSCRIPT",
        TranslationError::ModelNotInstalled { .. } => "TRANSLATE_MODEL_NOT_INSTALLED",
        TranslationError::LlamaNotInstalled => "TRANSLATE_LLAMA_NOT_INSTALLED",
        TranslationError::ModelLoadFailed { .. } => "TRANSLATE_MODEL_LOAD_FAILED",
        TranslationError::InvalidJson { .. } => "TRANSLATE_INVALID_JSON",
        TranslationError::IncompleteResponse { .. } => "TRANSLATE_INCOMPLETE_RESPONSE",
        TranslationError::OutOfMemory => "TRANSLATE_OUT_OF_MEMORY",
        TranslationError::LlmFailure { .. } => "TRANSLATE_LLM_FAILURE",
        TranslationError::WorkerCrash => "TRANSLATE_WORKER_CRASH",
        TranslationError::Cancelled => "TRANSLATE_CANCELLED",
        TranslationError::UnknownPromptVersion { .. } => "TRANSLATE_UNKNOWN_PROMPT",
        TranslationError::UnknownPreset { .. } => "TRANSLATE_UNKNOWN_PRESET",
        TranslationError::DownloadFailed { .. } => "TRANSLATE_DOWNLOAD_FAILED",
        TranslationError::Worker(_) => "TRANSLATE_WORKER_ERROR",
        TranslationError::Registry(_) => "TRANSLATE_REGISTRY",
        TranslationError::Db(_) => "TRANSLATE_DB",
        TranslationError::Io { .. } => "TRANSLATE_IO",
    }
}

// TranslatedSegment intentionally re-exported here for downstream tests
#[allow(dead_code)]
fn _touch(_: &TranslatedSegment) {}
