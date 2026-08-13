//! Wire types for the Phase 8 audio-mix subsystem.
//!
//! Mirrors the Phase 7 shape (`SyncSettings`, `SyncManifest`, …) so the
//! frontend can consume both with a matching mental model. Everything
//! stays camelCase on the wire and gets serialised straight to the
//! project's `audio/mix.json` manifest.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::jobs::JobSnapshot;
use crate::media::SourceFingerprint;

// v3 invalidates mixdowns produced by the old ducking graph, where the
// voice sidechain output was consumed twice without `asplit`. Those WAVs
// contain the original soundtrack but no audible generated voice.
pub const MIX_CACHE_SCHEMA_VERSION: u32 = 3;

/// v1's defaults for the two knobs v2 changed. Kept only so
/// [`MixManifest::migrated`] can tell an untouched knob from one the user
/// deliberately set to the same value.
const V1_ORIGINAL_VOLUME: f32 = 0.70;
const V1_DUCKING_DEPTH_DB: f32 = 12.0;

// -------------------------------------------------------------- settings

/// Per-project audio mix knobs.
///
/// The defaults treat the Vietnamese voice as the point of the film and
/// the original as background: the source track sits at 25% and drops a
/// further 20 dB while the voice speaks. Mixing it much louder than that
/// leaves two people talking over each other, and the foreign dialogue
/// wins because it's the one lip-synced to the picture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MixSettings {
    /// Linear gain multiplier for the original movie soundtrack
    /// (music, SFX, dialogue). Range 0.0..=2.0.
    #[serde(default = "default_original_volume")]
    pub original_volume: f32,
    /// Linear gain multiplier for the Vietnamese voice-over track.
    /// Range 0.0..=2.0.
    #[serde(default = "default_voice_volume")]
    pub voice_volume: f32,
    /// Enable side-chain compression of the original track keyed by the
    /// voice track. When on, the original ducks while the voice speaks.
    #[serde(default = "default_ducking_enabled")]
    pub ducking_enabled: bool,
    /// How much to reduce the original by when ducking, in dB below the
    /// baseline. Range 0.0..=30.0. Only used when `ducking_enabled`.
    #[serde(default = "default_ducking_depth_db")]
    pub ducking_depth_db: f32,
    /// Compressor threshold in dB (below zero). Voice louder than this
    /// triggers the ducker. Range -60.0..=0.0.
    #[serde(default = "default_ducking_threshold_db")]
    pub ducking_threshold_db: f32,
    /// Compressor attack time in ms. How fast the ducker clamps down.
    #[serde(default = "default_ducking_attack_ms")]
    pub ducking_attack_ms: f32,
    /// Compressor release time in ms. How fast the original recovers.
    #[serde(default = "default_ducking_release_ms")]
    pub ducking_release_ms: f32,
    /// Output sample rate. `None` keeps the source video's rate.
    #[serde(default)]
    pub output_sample_rate: Option<u32>,
    /// Output channels (1 = mono, 2 = stereo).
    #[serde(default = "default_output_channels")]
    pub output_channels: u32,
}

fn default_original_volume() -> f32 {
    0.25
}
fn default_voice_volume() -> f32 {
    1.00
}
fn default_ducking_enabled() -> bool {
    true
}
fn default_ducking_depth_db() -> f32 {
    20.0
}
fn default_ducking_threshold_db() -> f32 {
    -24.0
}
fn default_ducking_attack_ms() -> f32 {
    20.0
}
fn default_ducking_release_ms() -> f32 {
    300.0
}
fn default_output_channels() -> u32 {
    2
}

impl Default for MixSettings {
    fn default() -> Self {
        Self {
            original_volume: default_original_volume(),
            voice_volume: default_voice_volume(),
            ducking_enabled: default_ducking_enabled(),
            ducking_depth_db: default_ducking_depth_db(),
            ducking_threshold_db: default_ducking_threshold_db(),
            ducking_attack_ms: default_ducking_attack_ms(),
            ducking_release_ms: default_ducking_release_ms(),
            output_sample_rate: None,
            output_channels: default_output_channels(),
        }
    }
}

impl MixSettings {
    /// Clamp every numeric field into the range the UI actually offers.
    /// Called on every entry point so a hand-edited `mix.json` cannot
    /// hand FFmpeg absurd values.
    pub fn normalised(self) -> Self {
        Self {
            original_volume: clamp_f32(self.original_volume, 0.0, 2.0),
            voice_volume: clamp_f32(self.voice_volume, 0.0, 2.0),
            ducking_enabled: self.ducking_enabled,
            ducking_depth_db: clamp_f32(self.ducking_depth_db, 0.0, 30.0),
            ducking_threshold_db: clamp_f32(self.ducking_threshold_db, -60.0, 0.0),
            ducking_attack_ms: clamp_f32(self.ducking_attack_ms, 1.0, 500.0),
            ducking_release_ms: clamp_f32(self.ducking_release_ms, 10.0, 5000.0),
            output_sample_rate: self.output_sample_rate.filter(|&s| s >= 8000),
            output_channels: self.output_channels.clamp(1, 2),
        }
    }
}

fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() {
        return lo;
    }
    v.clamp(lo, hi)
}

// -------------------------------------------------------------- summary/status

/// The lifecycle of `audio/mixed_vi.wav` as far as the UI cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixStatus {
    /// Nothing on disk yet.
    Missing,
    /// File on disk but cache key drifted (settings or upstream changed).
    Stale,
    /// File matches the current cache identity.
    Ready,
}

// -------------------------------------------------------------- manifest

/// The full JSON persisted at `<project>/audio/mix.json`. There is
/// exactly one active mixdown per project — unlike Phase 6/7 which
/// stamp one entry per segment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MixManifest {
    pub version: u32,
    #[serde(default)]
    pub settings: MixSettings,
    /// The one and only mixdown entry.
    #[serde(default)]
    pub current: Option<MixEntry>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl MixManifest {
    pub fn empty(settings: MixSettings) -> Self {
        let now = Utc::now();
        Self {
            version: MIX_CACHE_SCHEMA_VERSION,
            settings,
            current: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Bring a manifest written by an older build up to the current
    /// schema. Applied on every load, and version-gated so each step
    /// runs at most once per project.
    ///
    /// v1 held the original at 70% and ducked it by 12 dB, which left the
    /// source dialogue competing with the voice-over instead of sitting
    /// behind it. The cached WAV regenerates on its own because the cache
    /// key carries the schema version, but the stored settings would
    /// otherwise pin the project to the old balance forever. Adopt the new
    /// value only where the knob still holds the v1 default, so a
    /// deliberate choice survives.
    pub fn migrated(mut self) -> Self {
        if self.version < 2 {
            if approx_eq(self.settings.original_volume, V1_ORIGINAL_VOLUME) {
                self.settings.original_volume = default_original_volume();
            }
            if approx_eq(self.settings.ducking_depth_db, V1_DUCKING_DEPTH_DB) {
                self.settings.ducking_depth_db = default_ducking_depth_db();
            }
        }
        self.version = MIX_CACHE_SCHEMA_VERSION;
        self
    }
}

/// Compare two knob values for "the user never moved this". Slider values
/// make a round-trip through JSON, so an exact `==` is the wrong tool.
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// One entry in the mix manifest (there's only ever zero or one at a
/// time). Contains everything a downstream stage needs to know without
/// re-probing FFmpeg.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MixEntry {
    /// Deterministic hash — see [`build_mix_cache_key`].
    pub cache_key: String,
    /// Hash of the source video (Phase 2 fingerprint) — decouples the
    /// mix from the raw file bytes so re-imports invalidate the mix.
    pub source_fingerprint: SourceFingerprint,
    /// Path relative to the project root — e.g. `audio/mixed_vi.wav`.
    pub file: String,
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub size_bytes: u64,
    /// Number of voice segments folded into the mix (excludes silence).
    pub voice_segment_count: u32,
    /// Total number of subtitle segments considered (whether voiced or not).
    pub subtitle_count: u32,
    /// Settings used to produce this file — the frontend uses this to
    /// show the "generated with" summary next to the current sliders.
    pub settings: MixSettings,
    pub generated_at: DateTime<Utc>,
}

/// Compact summary the frontend consumes in the mix panel and pipeline
/// row — never ships the whole manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MixSummary {
    pub status: MixStatus,
    pub settings: MixSettings,
    /// Duration of the current mix if any.
    pub duration_secs: Option<f64>,
    /// Absolute path on disk. Handy for the `<audio>` preview.
    pub absolute_path: Option<String>,
    pub relative_path: Option<String>,
    pub voice_segment_count: u32,
    pub subtitle_count: u32,
    pub size_bytes: Option<u64>,
    pub generated_at: Option<DateTime<Utc>>,
    /// True iff a fresh run is necessary before the mix is up to date.
    pub needs_generate: bool,
    /// Any warning text worth surfacing (e.g. `synced/ is empty`).
    pub warning: Option<String>,
    pub manifest_relative_path: String,
}

// -------------------------------------------------------------- env

/// Snapshot returned by `get_mix_env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MixEnv {
    pub ffmpeg_available: bool,
    pub ffmpeg_path: Option<String>,
    pub default_settings: MixSettings,
}

// -------------------------------------------------------------- request

/// Which subset of segments to (re)include in the mix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum MixMode {
    /// Include every synced voice segment.
    #[default]
    All,
}

/// Full payload for the `apply_mix` command.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MixRequest {
    #[serde(default)]
    pub settings: MixSettings,
    #[serde(default)]
    pub mode: MixMode,
}

/// What `apply_mix` returns — either an inline no-op (nothing changed)
/// or a `JobSnapshot` for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum MixGenerateStart {
    UpToDate { summary: MixSummary },
    Started(JobSnapshot),
}

/// What `get_project_mix_preview` returns for the preview `<audio>` tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewMixResult {
    pub absolute_path: String,
    pub relative_path: String,
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub cache_hit: bool,
}

// -------------------------------------------------------------- cache key

/// Deterministic hash over every input that materially changes
/// `mixed_vi.wav`. Anything not folded in here is either handled
/// verbatim by FFmpeg (encoder version) or intentionally out of scope
/// (log level, tmp dir, etc.).
pub fn build_mix_cache_key(
    source_fp: &SourceFingerprint,
    voice_segments: &[MixVoiceInput],
    settings: &MixSettings,
) -> String {
    let s = settings.normalised();
    let mut parts: Vec<String> = Vec::with_capacity(voice_segments.len() + 16);
    parts.push(format!("mix_v{}", MIX_CACHE_SCHEMA_VERSION));
    parts.push(format!("src_size={}", source_fp.size_bytes));
    parts.push(format!("src_hash={}", source_fp.hash));
    parts.push(format!("orig_vol={:.4}", s.original_volume));
    parts.push(format!("voice_vol={:.4}", s.voice_volume));
    parts.push(format!("duck={}", s.ducking_enabled as u8));
    if s.ducking_enabled {
        parts.push(format!("duck_depth={:.2}", s.ducking_depth_db));
        parts.push(format!("duck_thresh={:.2}", s.ducking_threshold_db));
        parts.push(format!("duck_attack={:.2}", s.ducking_attack_ms));
        parts.push(format!("duck_release={:.2}", s.ducking_release_ms));
    }
    parts.push(format!(
        "out_sample_rate={}",
        s.output_sample_rate
            .map(|v| v.to_string())
            .unwrap_or_default()
    ));
    parts.push(format!("out_channels={}", s.output_channels));
    parts.push(format!("segments={}", voice_segments.len()));
    // Segments are sorted by segment id so the hash is order-stable.
    let mut sorted: Vec<&MixVoiceInput> = voice_segments.iter().collect();
    sorted.sort_by_key(|s| s.segment_id);
    for v in sorted {
        parts.push(format!(
            "seg:{}|start={:.3}|cache={}",
            v.segment_id, v.target_start_secs, v.sync_cache_key
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// One voice input fed to the mixer.
///
/// Kept intentionally minimal — the file path plus its subtitle-window
/// start-time plus the sync cache key it came from. Duration comes
/// from the file itself, which the FFmpeg command discovers at run
/// time via `adelay`.
#[derive(Debug, Clone, PartialEq)]
pub struct MixVoiceInput {
    pub segment_id: u32,
    pub target_start_secs: f64,
    /// Path relative to the project root, e.g. `voices/synced/000001.wav`.
    pub relative_file: String,
    pub sync_cache_key: String,
    /// Status of the source sync entry. `Empty` segments are skipped by
    /// the mixer even though sync produced a silence WAV — mixing pure
    /// silence adds nothing.
    pub is_empty: bool,
}
