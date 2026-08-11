//! Phase 10 — unified local-model registry.
//!
//! The registry is the single source of truth the frontend Settings ›
//! AI Models pane talks to. It does NOT re-implement the per-stage
//! scanners — those live in the Python worker (Phases 3/4/6) and stay
//! authoritative. This module *aggregates* what those scanners report
//! into one flat `Vec<LocalModel>` so a user can see every installed
//! model in one place, regardless of stage.
//!
//! Design decisions:
//!
//! * The registry stores **only metadata** (id, path, size, status).
//!   Model binaries are never loaded here — that stays with the
//!   worker providers.
//! * A short-lived in-memory cache absorbs repeated UI reads
//!   (`getLocalModels()` on every focus). A rescan is only performed
//!   when the user asks (`rescan_local_models`) or the models
//!   directory changes.
//! * Validation is intentionally lightweight: existence + required
//!   file names + non-zero size. Full model loading is left to the
//!   inference providers.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tauri::async_runtime::spawn_blocking;

use crate::stt::SttService;
use crate::translation::TranslationService;
use crate::tts::TtsService;

/// The kind of a locally-installed model. Kept flat so the wire
/// format stays a simple enum and the UI can filter by column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Whisper,
    Translation,
    Tts,
    Voice,
}

impl ModelKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Translation => "translation",
            Self::Tts => "tts",
            Self::Voice => "voice",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "whisper" => Self::Whisper,
            "translation" => Self::Translation,
            "tts" => Self::Tts,
            "voice" => Self::Voice,
            _ => return None,
        })
    }
}

/// A single row shown in the Model Manager. Never carries the model
/// binary — only the metadata the UI needs to render + the absolute
/// path so the user can locate it in Finder / Explorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalModel {
    /// Stable per-kind identifier (e.g. `whisper:small`,
    /// `translation:qwen-7b.gguf`, `tts:piper:vi_VN-01`).
    pub id: String,
    /// Human-readable name (for Whisper: `small`; for GGUF:
    /// filename; for voices: registry name).
    pub name: String,
    pub kind: ModelKind,
    /// Engine identifier when relevant (`faster-whisper`, `llama.cpp`,
    /// `piper`, ...). Optional so future engines can slot in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Absolute path to the model file or directory. `None` when the
    /// model is *known* (e.g. Whisper 'small') but not installed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Free-form version tag (Whisper `params_m`, voice `quality`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub status: ModelStatus,
    /// Optional short hint for the "invalid" or "missing" cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// Lightweight validation outcome. Missing = the app knows about the
/// model but doesn't have the files. Invalid = files are there but
/// don't look like a working model (wrong extension, empty file, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Available,
    Missing,
    Invalid,
}

/// Snapshot of the model directory the app is currently using and
/// whether it is user-overridden. Rendered above the model table in
/// Settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDirectoryInfo {
    pub path: String,
    pub is_default: bool,
    pub default_path: String,
    pub whisper_subdir: String,
    pub translation_subdir: String,
    pub tts_subdir: String,
    /// `true` when the directory does not exist yet (e.g. right after
    /// a fresh install). The Registry still reports zero models in
    /// that case — nothing is created implicitly.
    pub exists: bool,
}

/// Handle held by `AppState`. Cheap to clone. Every read is served
/// from the in-memory cache, mutation goes through the worker for the
/// underlying registries.
#[derive(Clone)]
pub struct ModelRegistry {
    inner: Arc<ModelRegistryInner>,
}

struct ModelRegistryInner {
    stt: Arc<SttService>,
    translation: Arc<TranslationService>,
    tts: Arc<TtsService>,
    cached: RwLock<Option<Vec<LocalModel>>>,
}

impl ModelRegistry {
    pub fn new(
        stt: Arc<SttService>,
        translation: Arc<TranslationService>,
        tts: Arc<TtsService>,
    ) -> Self {
        Self {
            inner: Arc::new(ModelRegistryInner {
                stt,
                translation,
                tts,
                cached: RwLock::new(None),
            }),
        }
    }

    /// Return the cached model list, running a scan on first access
    /// or when the cache has been invalidated by `rescan()` /
    /// `invalidate()`.
    pub async fn list(&self) -> Vec<LocalModel> {
        if let Some(cached) = self.inner.cached.read().clone() {
            return cached;
        }
        self.rescan().await
    }

    /// Force-refresh the registry. Called by the "Scan Models" button
    /// and after `import_local_model` / `set_models_dir` succeed.
    pub async fn rescan(&self) -> Vec<LocalModel> {
        let stt = self.inner.stt.clone();
        let translation = self.inner.translation.clone();
        let tts = self.inner.tts.clone();

        // Each of these is a JSON-RPC call to the worker (fast and
        // already asynchronous) — we run them concurrently.
        let (whisper, gguf, voices) = tokio::join!(
            async {
                stt.list_models().await.unwrap_or_else(|err| {
                    tracing::warn!(%err, "stt list_models failed during registry scan");
                    Vec::new()
                })
            },
            async {
                translation.list_models().await.unwrap_or_else(|err| {
                    tracing::warn!(
                        %err,
                        "translation list_models failed during registry scan",
                    );
                    Vec::new()
                })
            },
            async {
                tts.list_voices().await.unwrap_or_else(|err| {
                    tracing::warn!(%err, "tts list_voices failed during registry scan");
                    Vec::new()
                })
            },
        );

        let mut out: Vec<LocalModel> = Vec::new();

        for m in whisper {
            let status = if m.installed {
                validate_whisper_dir(m.path.as_deref())
            } else {
                ModelStatus::Missing
            };
            let hint = match status {
                ModelStatus::Invalid => {
                    Some("Directory exists but is missing model.bin or config.json.".to_string())
                }
                _ => None,
            };
            out.push(LocalModel {
                id: format!("whisper:{}", m.name),
                name: m.name,
                kind: ModelKind::Whisper,
                engine: Some("faster-whisper".into()),
                language: None,
                path: m.path,
                size_bytes: m.size_bytes,
                version: Some(format!("{}M params", m.params_m)),
                status,
                hint,
            });
        }

        for m in gguf {
            // The translation registry only surfaces installed models,
            // so anything we see here is expected to be Available. We
            // still guard against the file having been removed between
            // the worker's scan and this call.
            let status = validate_gguf_file(&m.path);
            let hint = match status {
                ModelStatus::Missing => Some("File was removed after scan.".into()),
                ModelStatus::Invalid => Some("File is empty or unreadable.".into()),
                _ => None,
            };
            out.push(LocalModel {
                id: format!("translation:{}", m.name),
                name: m.name,
                kind: ModelKind::Translation,
                engine: Some("llama.cpp".into()),
                language: None,
                path: Some(m.path),
                size_bytes: Some(m.size_bytes),
                version: None,
                status,
                hint,
            });
        }

        for v in voices {
            let status = validate_voice_file(&v.model_path);
            let hint = match status {
                ModelStatus::Missing => Some("ONNX file was removed after scan.".into()),
                ModelStatus::Invalid => Some("ONNX file is empty or unreadable.".into()),
                _ => None,
            };
            out.push(LocalModel {
                id: format!("tts:{}:{}", v.engine, v.id),
                name: v.name.clone(),
                kind: ModelKind::Voice,
                engine: Some(v.engine),
                language: Some(v.language),
                path: Some(v.model_path),
                size_bytes: file_size_or_none(v.config_path.as_deref().unwrap_or("")),
                version: v.quality,
                status,
                hint,
            });
        }

        out.sort_by(|a, b| {
            a.kind
                .as_str()
                .cmp(b.kind.as_str())
                .then(a.name.cmp(&b.name))
        });

        *self.inner.cached.write() = Some(out.clone());
        out
    }

    /// Drop the cached list so the next `list()` re-scans. Called
    /// after `import_local_model` and `set_models_dir` change the
    /// on-disk layout.
    pub fn invalidate(&self) {
        *self.inner.cached.write() = None;
    }
}

// -------------------------------------------------------------- validators

fn validate_whisper_dir(path: Option<&str>) -> ModelStatus {
    let Some(p) = path else {
        return ModelStatus::Missing;
    };
    let dir = std::path::Path::new(p);
    if !dir.is_dir() {
        return ModelStatus::Missing;
    }
    for required in ["model.bin", "config.json"] {
        let f = dir.join(required);
        if !f.is_file() {
            return ModelStatus::Invalid;
        }
        if let Ok(md) = std::fs::metadata(&f) {
            if md.len() == 0 {
                return ModelStatus::Invalid;
            }
        }
    }
    ModelStatus::Available
}

fn validate_gguf_file(path: &str) -> ModelStatus {
    let f = std::path::Path::new(path);
    if !f.is_file() {
        return ModelStatus::Missing;
    }
    match std::fs::metadata(f) {
        Ok(md) if md.len() > 0 => ModelStatus::Available,
        _ => ModelStatus::Invalid,
    }
}

fn validate_voice_file(path: &str) -> ModelStatus {
    let f = std::path::Path::new(path);
    if !f.is_file() {
        return ModelStatus::Missing;
    }
    match std::fs::metadata(f) {
        Ok(md) if md.len() > 0 => ModelStatus::Available,
        _ => ModelStatus::Invalid,
    }
}

fn file_size_or_none(path: &str) -> Option<u64> {
    if path.is_empty() {
        return None;
    }
    std::fs::metadata(path).ok().map(|md| md.len())
}

/// Probe whether the currently-configured models directory is
/// writable so the UI can grey out "Add Local Model" if not. The
/// probe is intentionally cheap: it does a `mkdir -p` and drops a
/// zero-byte sentinel that we then delete.
pub async fn probe_writable(dir: std::path::PathBuf) -> Result<(), String> {
    spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let sentinel = dir.join(".lmt-write-probe");
        std::fs::write(&sentinel, b"").map_err(|e| e.to_string())?;
        std::fs::remove_file(&sentinel).map_err(|e| e.to_string())?;
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}
