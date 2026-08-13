//! Wire types for the STT subsystem, mirroring the Python-side JSON.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::jobs::JobSnapshot;

/// Client-supplied transcription parameters. Everything that
/// materially affects the produced transcript belongs here so cache
/// invalidation stays trivial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SttOptions {
    pub model: String,
    /// ``None`` (or ``"auto"`` from the UI) => language auto-detect.
    pub language: Option<String>,
    /// ``None`` => pick a safe default based on detected devices.
    pub device: Option<String>,
    /// ``None`` => derived from device (``float16`` on CUDA, ``int8`` on CPU).
    pub compute_type: Option<String>,
    pub beam_size: u32,
    pub word_timestamps: bool,
    pub vad_filter: bool,
    pub initial_prompt: Option<String>,
    pub temperature: f32,
    #[serde(default)]
    pub quality_profile: Option<String>,
    #[serde(default = "default_resegment")]
    pub resegment: bool,
}

fn default_resegment() -> bool {
    true
}

impl Default for SttOptions {
    fn default() -> Self {
        Self {
            model: "medium".into(),
            language: None,
            device: None,
            compute_type: None,
            beam_size: 5,
            word_timestamps: true,
            vad_filter: true,
            initial_prompt: None,
            temperature: 0.0,
            quality_profile: Some("balanced".into()),
            resegment: true,
        }
    }
}

/// A device the worker can run inference on.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttDeviceInfo {
    pub kind: String,
    pub label: String,
    pub supported: bool,
    #[serde(default = "one")]
    pub count: u32,
    pub detail: Option<String>,
}

fn one() -> u32 {
    1
}

/// Snapshot returned by ``get_stt_env``: which backends are available
/// and where the model cache lives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SttEnv {
    pub devices: Vec<SttDeviceInfo>,
    pub default_device: String,
    pub whisper_installed: bool,
    pub models_root: String,
    #[serde(default)]
    pub hardware: Option<serde_json::Value>,
    #[serde(default)]
    pub large_v3: Option<serde_json::Value>,
    #[serde(default)]
    pub profiles: Option<serde_json::Value>,
}

/// Catalogue entry — one row of the "Model:" dropdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub name: String,
    pub repo: String,
    pub params_m: u32,
    pub installed: bool,
    pub size_bytes: Option<u64>,
    pub path: Option<String>,
}

/// A single word timestamp emitted when
/// ``options.word_timestamps == true``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhisperWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
}

/// A single Whisper segment — the canonical subtitle timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeSegment {
    pub id: u32,
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_speech_prob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WhisperWord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptAudio {
    pub path: String,
    pub hash: String,
}

/// The full JSON persisted at ``transcription/transcription.json``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcript {
    pub version: u32,
    pub language: String,
    pub segments: Vec<TranscribeSegment>,
    pub model: String,
    pub device: String,
    pub compute_type: String,
    pub word_timestamps: bool,
    pub audio: TranscriptAudio,
    pub duration_secs: f64,
    pub cache_key: String,
    pub created_at: DateTime<Utc>,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub options: serde_json::Value,
}

fn default_provider() -> String {
    "faster-whisper".into()
}

/// What ``transcribe`` returns to the frontend — either an inline cache
/// hit or a JobSnapshot for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TranscribeStart {
    CacheHit {
        transcript: TranscriptSummary,
        absolute_path: String,
    },
    Started(JobSnapshot),
}

/// Compact summary used in list responses so the frontend can render
/// "42 segments detected" without shipping the whole segment array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSummary {
    pub language: String,
    pub model: String,
    pub device: String,
    pub compute_type: String,
    pub word_timestamps: bool,
    pub segment_count: u32,
    pub duration_secs: f64,
    pub cache_key: String,
    pub created_at: DateTime<Utc>,
    pub audio_hash: String,
    pub relative_path: String,
}

impl TranscriptSummary {
    pub fn from_transcript(t: &Transcript, relative_path: &str) -> Self {
        Self {
            language: t.language.clone(),
            model: t.model.clone(),
            device: t.device.clone(),
            compute_type: t.compute_type.clone(),
            word_timestamps: t.word_timestamps,
            segment_count: t.segments.len() as u32,
            duration_secs: t.duration_secs,
            cache_key: t.cache_key.clone(),
            created_at: t.created_at,
            audio_hash: t.audio.hash.clone(),
            relative_path: relative_path.into(),
        }
    }
}
