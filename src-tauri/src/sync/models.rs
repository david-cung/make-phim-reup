//! Wire types for the Phase 7 sync subsystem, mirroring Python-side
//! JSON. Kept camelCase on the wire so the frontend can consume them
//! without a translation layer.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::jobs::JobSnapshot;

pub const SYNC_CACHE_SCHEMA_VERSION: u32 = 1;

// -------------------------------------------------------------- settings

/// Per-project sync knobs. The default `[0.85, 1.20]` range comes
/// straight from the phase spec — anything outside it is classified
/// as `too_long` rather than silently distorted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SyncSettings {
    #[serde(default = "default_min_speed")]
    pub min_speed: f32,
    #[serde(default = "default_max_speed")]
    pub max_speed: f32,
    /// `None` = keep the source TTS WAV's sample rate.
    #[serde(default)]
    pub output_sample_rate: Option<u32>,
    #[serde(default = "default_output_channels")]
    pub output_channels: u32,
}

fn default_min_speed() -> f32 {
    0.85
}
fn default_max_speed() -> f32 {
    1.20
}
fn default_output_channels() -> u32 {
    1
}

impl Default for SyncSettings {
    fn default() -> Self {
        Self {
            min_speed: default_min_speed(),
            max_speed: default_max_speed(),
            output_sample_rate: None,
            output_channels: default_output_channels(),
        }
    }
}

impl SyncSettings {
    pub fn normalised(self) -> Self {
        let mut lo = clamp_f32(self.min_speed, 0.5, 1.0);
        let mut hi = clamp_f32(self.max_speed, 1.0, 2.0);
        if hi < lo {
            hi = lo;
        }
        if lo.is_nan() {
            lo = default_min_speed();
        }
        if hi.is_nan() {
            hi = default_max_speed();
        }
        let sr = self.output_sample_rate.filter(|&s| s >= 8000);
        let channels = self.output_channels.max(1);
        Self {
            min_speed: lo,
            max_speed: hi,
            output_sample_rate: sr,
            output_channels: channels,
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

// ----------------------------------------------------------------- status

/// The per-segment classification the planner produces. Mirrors the
/// Python constants of the same names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncStatus {
    /// No usable TTS input; output is pure silence of the target length.
    Empty,
    /// Voice already fits inside the target window (padded with silence).
    Fits,
    /// Voice was stretched via `atempo` inside the allowed range.
    Adjusted,
    /// Even at `max_speed` the voice overflows the window — flagged
    /// for the user to shorten the translation.
    TooLong,
}

impl SyncStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Fits => "fits",
            Self::Adjusted => "adjusted",
            Self::TooLong => "too_long",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "empty" => Self::Empty,
            "fits" => Self::Fits,
            "adjusted" => Self::Adjusted,
            "too_long" => Self::TooLong,
            _ => return None,
        })
    }
}

// ----------------------------------------------------------------- planner

/// Pure planner output — no FFmpeg yet. Used by both the frontend
/// preview and internal cache-hit checks so both sides agree on
/// classification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SyncPlan {
    pub status: SyncStatus,
    pub target_duration_secs: f64,
    pub original_duration_secs: f64,
    pub final_duration_secs: f64,
    pub speed_factor: f32,
}

// -------------------------------------------------------------- manifest

/// One entry in `voices/synced/sync.json` — the per-segment cache
/// identity plus everything needed by the UI badge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SyncSegmentEntry {
    pub segment_id: u32,
    pub status: SyncStatus,
    pub target_start: f64,
    pub target_end: f64,
    pub target_duration_secs: f64,
    pub original_duration_secs: f64,
    pub final_duration_secs: f64,
    pub speed_factor: f32,
    pub cache_key: String,
    pub tts_cache_key: String,
    /// Path relative to the project root (e.g. `voices/synced/000012.wav`).
    pub file: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub size_bytes: u64,
    pub generated_at: DateTime<Utc>,
}

/// The full JSON persisted at `voices/synced/sync.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SyncManifest {
    pub version: u32,
    #[serde(default)]
    pub settings: SyncSettings,
    pub segments: Vec<SyncSegmentEntry>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SyncManifest {
    pub fn empty(settings: SyncSettings) -> Self {
        let now = Utc::now();
        Self {
            version: SYNC_CACHE_SCHEMA_VERSION,
            settings,
            segments: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn find(&self, segment_id: u32) -> Option<&SyncSegmentEntry> {
        self.segments.iter().find(|s| s.segment_id == segment_id)
    }

    pub fn upsert(&mut self, entry: SyncSegmentEntry) {
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

/// Compact summary for the Sync panel — never ships the whole
/// per-segment manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncSummary {
    pub settings: SyncSettings,
    /// Total subtitle segments in `subtitles.json`.
    pub subtitle_count: u32,
    /// Segments whose synced entry matches the current cache identity.
    pub synced_count: u32,
    /// Segments with no synced entry at all.
    pub missing_count: u32,
    /// Segments whose synced entry is out of date (TTS regenerated
    /// or timing/settings changed).
    pub stale_count: u32,
    /// Segments flagged as `too_long` even after max-speed stretch.
    pub too_long_count: u32,
    /// Segments that were stretched inside the allowed range.
    pub adjusted_count: u32,
    /// Segments that fit inside their window without stretching.
    pub fits_count: u32,
    /// Segments with no usable TTS (still produce silence).
    pub empty_count: u32,
    pub updated_at: DateTime<Utc>,
    pub relative_path: String,
}

// -------------------------------------------------------------- env

/// Snapshot returned by `get_sync_env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncEnv {
    pub ffmpeg_available: bool,
    pub ffmpeg_path: Option<String>,
    pub default_min_speed: f32,
    pub default_max_speed: f32,
}

// -------------------------------------------------------------- generation

/// Which subset of segments to sync.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum SyncMode {
    /// Sync every subtitle whose cache identity currently misses.
    #[default]
    Missing,
    /// Force-sync every subtitle regardless of cache identity.
    All,
    /// Force-sync just this list of segment ids.
    Selected { ids: Vec<u32> },
}

/// Full payload for the `apply_sync` command.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncRequest {
    #[serde(default)]
    pub settings: SyncSettings,
    #[serde(default)]
    pub mode: SyncMode,
}

/// What `apply_sync` returns — either an inline no-op (nothing to
/// do) or a JobSnapshot for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum SyncGenerateStart {
    UpToDate { summary: SyncSummary },
    Started(JobSnapshot),
}

/// What `preview_sync_segment` returns — the file that will be played
/// plus the classification for the badge.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewSyncResult {
    pub segment_id: u32,
    pub status: SyncStatus,
    pub target_duration_secs: f64,
    pub original_duration_secs: f64,
    pub final_duration_secs: f64,
    pub speed_factor: f32,
    pub absolute_path: String,
    pub relative_path: String,
    pub cache_hit: bool,
}

// -------------------------------------------------------------- cache key

/// Deterministic hash over every input that materially changes the
/// synced WAV. MUST match the Python-side `build_sync_cache_key`.
pub fn build_sync_cache_key(
    tts_cache_key: &str,
    target_duration_secs: f64,
    settings: &SyncSettings,
) -> String {
    let s = settings.normalised();
    let sr_part = s
        .output_sample_rate
        .map(|v| v.to_string())
        .unwrap_or_default();
    let parts = [
        format!("sync_v{}", SYNC_CACHE_SCHEMA_VERSION),
        format!("tts={}", tts_cache_key),
        format!("target={:.3}", target_duration_secs),
        format!("min_speed={:.4}", s.min_speed),
        format!("max_speed={:.4}", s.max_speed),
        format!("out_sample_rate={}", sr_part),
        format!("out_channels={}", s.output_channels),
    ];
    let mut hasher = Sha256::new();
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// Relative path for a synced per-segment WAV.
pub fn synced_file_relative(segment_id: u32) -> String {
    format!("voices/synced/{:06}.wav", segment_id)
}
