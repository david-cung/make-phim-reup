//! Wire types for the translation subsystem, mirroring Python-side JSON.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::jobs::JobSnapshot;

/// Client-supplied translation parameters. Everything that materially
/// affects the produced translation lives here so cache invalidation
/// stays trivial (a hash of the options + transcript key).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranslateOptions {
    /// GGUF filename under ``<models>/translation/`` (e.g. `qwen2-7b-q4_k_m.gguf`).
    pub model: String,
    #[serde(default = "en")]
    pub source_language: String,
    #[serde(default = "vi")]
    pub target_language: String,
    #[serde(default = "default_prompt_version")]
    pub prompt_version: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: u32,
    #[serde(default = "default_context_before")]
    pub context_before: u32,
    #[serde(default = "default_context_after")]
    pub context_after: u32,
    #[serde(default = "default_retry_context_before")]
    pub retry_context_before: u32,
    #[serde(default = "default_retry_context_after")]
    pub retry_context_after: u32,
    #[serde(default = "default_max_translation_retries")]
    pub max_translation_retries: u32,
    #[serde(default = "default_low_confidence_threshold")]
    pub low_confidence_threshold: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn en() -> String {
    "en".into()
}
fn vi() -> String {
    "vi".into()
}
fn default_prompt_version() -> String {
    "translation_prompt_v5".into()
}
fn default_chunk_size() -> u32 {
    15
}
fn default_context_before() -> u32 {
    5
}
fn default_context_after() -> u32 {
    5
}
fn default_retry_context_before() -> u32 {
    12
}
fn default_retry_context_after() -> u32 {
    12
}
fn default_max_translation_retries() -> u32 {
    2
}
fn default_low_confidence_threshold() -> f32 {
    0.80
}
fn default_temperature() -> f32 {
    0.2
}
fn default_top_p() -> f32 {
    0.95
}
fn default_max_tokens() -> u32 {
    2048
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self {
            model: String::new(),
            source_language: en(),
            target_language: vi(),
            prompt_version: default_prompt_version(),
            chunk_size: default_chunk_size(),
            context_before: default_context_before(),
            context_after: default_context_after(),
            retry_context_before: default_retry_context_before(),
            retry_context_after: default_retry_context_after(),
            max_translation_retries: default_max_translation_retries(),
            low_confidence_threshold: default_low_confidence_threshold(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// A single GGUF model discovered under ``<models>/translation/``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslationModelInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub is_default: bool,
}

/// Phase 12 — one entry from the curated list of translation
/// presets the app can auto-download. The Python worker owns the
/// canonical list (see ``translation/registry.py::_RECOMMENDED_MODELS``).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendedPreset {
    pub preset: String,
    pub repo: String,
    pub filename: String,
    pub approx_size_bytes: u64,
    pub label: String,
    pub is_default: bool,
}

/// Snapshot returned by ``get_translation_env``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslationEnv {
    pub llama_installed: bool,
    pub models_root: String,
    pub translation_root: String,
    pub default_model: Option<String>,
    pub prompt_versions: Vec<String>,
}

/// A single translated subtitle segment as stored in ``translation.json``.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranslatedSegment {
    pub id: u32,
    pub source_text: String,
    pub translation: String,
    #[serde(default)]
    pub dubbing: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub edited: bool,
    #[serde(default)]
    pub translation_metadata: serde_json::Value,
}

/// The full JSON persisted at ``translation/translation.json``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslationDoc {
    pub version: u32,
    pub source_language: String,
    pub target_language: String,
    pub segments: Vec<TranslatedSegment>,
    pub model: String,
    pub prompt_version: String,
    pub cache_key: String,
    pub transcript_cache_key: String,
    pub audio_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub options: serde_json::Value,
}

fn default_provider() -> String {
    "llama.cpp".into()
}

/// Compact summary the frontend uses to render "42/350 translated"
/// without shipping the whole segment array on every refresh.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslationSummary {
    pub source_language: String,
    pub target_language: String,
    pub model: String,
    pub prompt_version: String,
    pub segment_count: u32,
    pub translated_count: u32,
    pub edited_count: u32,
    pub cache_key: String,
    pub transcript_cache_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub relative_path: String,
}

impl TranslationSummary {
    pub fn from_doc(doc: &TranslationDoc, relative_path: &str) -> Self {
        let translated = doc
            .segments
            .iter()
            .filter(|s| !s.translation.trim().is_empty())
            .count() as u32;
        let edited = doc.segments.iter().filter(|s| s.edited).count() as u32;
        Self {
            source_language: doc.source_language.clone(),
            target_language: doc.target_language.clone(),
            model: doc.model.clone(),
            prompt_version: doc.prompt_version.clone(),
            segment_count: doc.segments.len() as u32,
            translated_count: translated,
            edited_count: edited,
            cache_key: doc.cache_key.clone(),
            transcript_cache_key: doc.transcript_cache_key.clone(),
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            relative_path: relative_path.into(),
        }
    }
}

/// What ``translate`` returns to the frontend — either an inline cache
/// hit (nothing to do) or a JobSnapshot for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TranslateStart {
    CacheHit {
        summary: TranslationSummary,
        absolute_path: String,
    },
    Started(JobSnapshot),
}
