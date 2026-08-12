//! Wire types for the TTS subsystem, mirroring Python-side JSON.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::jobs::JobSnapshot;

pub const TTS_CACHE_SCHEMA_VERSION: u32 = 1;

// -------------------------------------------------------------- settings

/// Per-segment synthesis knobs.
///
/// Every field is stored on disk / hashed into the cache key regardless
/// of whether the current engine honours it — that way a user who
/// later switches to an engine that *does* respect e.g. `pitch` gets a
/// clean cache miss and a re-render at that moment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TtsSettings {
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub pitch: f32,
    #[serde(default = "default_volume")]
    pub volume: f32,
}

fn default_speed() -> f32 {
    1.0
}

fn default_volume() -> f32 {
    1.0
}

impl Default for TtsSettings {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 0.0,
            volume: 1.0,
        }
    }
}

impl TtsSettings {
    pub fn normalised(self) -> Self {
        Self {
            speed: clamp_f32(self.speed, 0.25, 4.0),
            pitch: clamp_f32(self.pitch, -12.0, 12.0),
            volume: clamp_f32(self.volume, 0.0, 4.0),
        }
    }
}

fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() {
        return lo;
    }
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

// -------------------------------------------------------------- env

/// Description of one TTS engine the worker knows about.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsEngineInfo {
    pub id: String,
    pub name: String,
    pub available: bool,
    #[serde(default)]
    pub supported_settings: Vec<String>,
}

/// Snapshot returned by ``get_tts_env``.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsEnv {
    pub engines: Vec<TtsEngineInfo>,
    pub models_root: String,
    pub tts_root: String,
    pub piper_installed: bool,
    pub default_engine: String,
}

// -------------------------------------------------------------- voices

/// Phase 12 — one entry from the curated list of TTS voice
/// presets the app can auto-download. Owned by the Python worker
/// (`tts/registry.py::_RECOMMENDED_VOICES`); the wire shape here
/// is fixed by `#[serde(rename_all = "camelCase")]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendedVoicePreset {
    pub preset: String,
    pub engine: String,
    pub voice_id: String,
    pub language: String,
    pub target_languages: Vec<String>,
    pub quality: String,
    pub approx_size_bytes: u64,
    pub label: String,
    pub is_default: bool,
}

/// One installed voice model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: String,
    pub engine: String,
    pub model_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    pub sample_rate: u32,
    pub installed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(default)]
    pub supported_settings: Vec<String>,
}

// -------------------------------------------------------------- manifest

/// One entry in `<project>/voices/voices.json` — the per-segment TTS
/// cache identity + generated-file metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TtsSegmentEntry {
    pub segment_id: u32,
    pub engine: String,
    pub voice_id: String,
    pub model_name: String,
    pub cache_key: String,
    pub text_hash: String,
    /// The literal (trimmed) text that was fed to the engine. Stored
    /// alongside `text_hash` purely so the UI can do cheap
    /// string-equality staleness checks without pulling in a
    /// cryptographic hash on the render path.
    #[serde(default)]
    pub text: String,
    pub speed: f32,
    pub pitch: f32,
    pub volume: f32,
    /// Path relative to the project root (e.g. ``voices/000012.wav``).
    pub file: String,
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub size_bytes: u64,
    pub generated_at: DateTime<Utc>,
}

/// The full JSON persisted at ``voices/voices.json``.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TtsManifest {
    pub version: u32,
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub default_voice_id: String,
    pub segments: Vec<TtsSegmentEntry>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TtsManifest {
    pub fn empty(engine: String, default_voice_id: String) -> Self {
        let now = Utc::now();
        Self {
            version: TTS_CACHE_SCHEMA_VERSION,
            engine,
            default_voice_id,
            segments: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn find(&self, segment_id: u32) -> Option<&TtsSegmentEntry> {
        self.segments.iter().find(|s| s.segment_id == segment_id)
    }

    pub fn upsert(&mut self, entry: TtsSegmentEntry) {
        match self
            .segments
            .iter()
            .position(|s| s.segment_id == entry.segment_id)
        {
            Some(idx) => self.segments[idx] = entry,
            None => self.segments.push(entry),
        }
        self.segments.sort_by_key(|s| s.segment_id);
        self.updated_at = Utc::now();
    }

    pub fn remove(&mut self, segment_id: u32) -> bool {
        let before = self.segments.len();
        self.segments.retain(|s| s.segment_id != segment_id);
        let removed = self.segments.len() < before;
        if removed {
            self.updated_at = Utc::now();
        }
        removed
    }
}

// -------------------------------------------------------------- summary

/// Compact summary for the frontend TTS panel — never ships the full
/// per-segment manifest so re-renders stay cheap.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsSummary {
    pub engine: String,
    pub default_voice_id: String,
    /// Total subtitle segments in `subtitles.json`.
    pub subtitle_count: u32,
    /// Segments whose manifest entry matches the current cache identity.
    pub generated_count: u32,
    /// Segments that have no manifest entry at all.
    pub missing_count: u32,
    /// Segments whose manifest entry exists but is stale (text/voice/
    /// settings changed since generation).
    pub stale_count: u32,
    pub updated_at: DateTime<Utc>,
    pub relative_path: String,
}

// -------------------------------------------------------------- generation

/// Which subset of segments to generate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum GenerateMode {
    /// Generate every subtitle whose cache identity currently misses.
    #[default]
    Missing,
    /// Force-generate every subtitle regardless of cache identity.
    All,
    /// Force-generate just this list of segment ids.
    Selected { ids: Vec<u32> },
}

/// Full payload for the ``generate_tts`` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub engine: String,
    pub default_voice_id: String,
    #[serde(default)]
    pub settings: TtsSettings,
    #[serde(default)]
    pub mode: GenerateMode,
}

/// What `generate_tts` returns — either an inline no-op (nothing to
/// do) or a JobSnapshot for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TtsGenerateStart {
    UpToDate { summary: TtsSummary },
    Started(JobSnapshot),
}

/// What `preview_tts_segment` returns — the file that will be played.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewResult {
    pub segment_id: u32,
    pub engine: String,
    pub voice_id: String,
    pub absolute_path: String,
    pub relative_path: String,
    pub duration_secs: f64,
    pub cache_hit: bool,
}

// -------------------------------------------------------------- cache key

/// Deterministic hash over every input that materially changes the
/// generated WAV. Must match the Python-side ``build_segment_cache_key``.
pub fn build_segment_cache_key(
    engine: &str,
    voice_id: &str,
    model_name: &str,
    text: &str,
    settings: &TtsSettings,
) -> String {
    let s = settings.normalised();
    let parts = [
        format!("tts_v{}", TTS_CACHE_SCHEMA_VERSION),
        engine.to_string(),
        voice_id.to_string(),
        model_name.to_string(),
        format!("speed={:.4}", s.speed),
        format!("pitch={:.4}", s.pitch),
        format!("volume={:.4}", s.volume),
        format!("text={}", text),
    ];
    let mut hasher = Sha256::new();
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

pub fn text_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// Convenience: returns `true` when the entry's cache identity still
/// matches the given expected fingerprint.
pub fn entry_matches(entry: &TtsSegmentEntry, expected_cache_key: &str) -> bool {
    entry.cache_key == expected_cache_key
}

/// Relative path template used for per-segment WAV files.
pub fn voice_file_relative(segment_id: u32) -> String {
    format!("voices/{:06}.wav", segment_id)
}
