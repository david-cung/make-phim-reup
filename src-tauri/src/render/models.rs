//! Wire types for the Phase 9 final-render subsystem.
//!
//! Mirrors the Phase 8 (`MixSettings`, `MixManifest`, …) shape so the
//! frontend can consume both with the same mental model. Everything
//! stays camelCase on the wire and gets serialised straight to the
//! project's `output/render.json` manifest.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::jobs::JobSnapshot;
use crate::media::SourceFingerprint;

pub const RENDER_CACHE_SCHEMA_VERSION: u32 = 1;

// -------------------------------------------------------------- settings

/// How the Vietnamese subtitle track should show up in the final file.
///
/// * `External` writes `movie_vi.srt` next to the muxed movie. Video is
///   copied (`-c:v copy`) — no re-encode.
/// * `Burned` renders the subtitle text into the pixels using FFmpeg's
///   `subtitles=` filter. This forces a video re-encode.
/// * `None` skips subtitles entirely — the file is just video + the
///   mixed Vietnamese audio. Video is copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubtitleMode {
    None,
    #[default]
    External,
    Burned,
}

impl SubtitleMode {
    /// Burning subtitles into pixels is the only mode that forces a
    /// video re-encode; the other two can stream-copy.
    pub const fn requires_reencode(self) -> bool {
        matches!(self, Self::Burned)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::External => "external",
            Self::Burned => "burned",
        }
    }
}

/// Which output container to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Mp4,
    Mkv,
}

impl OutputFormat {
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
        }
    }
}

/// Video codec knob. `Copy` = `-c:v copy` (fast, lossless, container
/// permitting). `Reencode(name)` = re-encode to the given codec.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum VideoCodec {
    #[default]
    Copy,
    Reencode {
        codec: String,
    },
}

impl VideoCodec {
    pub fn ffmpeg_name(&self) -> String {
        match self {
            Self::Copy => "copy".into(),
            Self::Reencode { codec } => codec.clone(),
        }
    }

    /// Cache-key fragment — deterministic string form.
    pub fn cache_repr(&self) -> String {
        match self {
            Self::Copy => "copy".into(),
            Self::Reencode { codec } => format!("re:{codec}"),
        }
    }
}

/// Audio codec knob. Always re-encoded from the mixed WAV, so this is
/// the FFmpeg codec name (`aac`, `libopus`, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioCodec {
    pub codec: String,
    /// Optional bitrate (e.g. `"192k"`). `None` = FFmpeg default.
    #[serde(default)]
    pub bitrate: Option<String>,
}

impl Default for AudioCodec {
    fn default() -> Self {
        Self {
            codec: "aac".into(),
            bitrate: Some("192k".into()),
        }
    }
}

/// Per-project render knobs. Deliberately minimal — the spec asks us
/// to hide low-level FFmpeg options from the main UI.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RenderSettings {
    #[serde(default)]
    pub output_format: OutputFormat,
    #[serde(default)]
    pub video_codec: VideoCodec,
    #[serde(default)]
    pub audio_codec: AudioCodec,
    #[serde(default)]
    pub subtitle_mode: SubtitleMode,
    /// Absolute path where the final file should be written. `None`
    /// means "use the default under `<project>/output/`".
    #[serde(default)]
    pub output_path: Option<String>,
}

impl RenderSettings {
    /// Normalise the settings: clamp fields the UI cannot legitimately
    /// produce and coerce the video codec to `Reencode` when the
    /// subtitle mode requires it.
    pub fn normalised(mut self) -> Self {
        // Trim bogus strings so cache keys stay stable across
        // whitespace-only edits.
        if let Some(b) = &self.audio_codec.bitrate {
            let t = b.trim().to_string();
            self.audio_codec.bitrate = if t.is_empty() { None } else { Some(t) };
        }
        self.audio_codec.codec = self.audio_codec.codec.trim().to_string();
        if self.audio_codec.codec.is_empty() {
            self.audio_codec.codec = "aac".into();
        }
        if let VideoCodec::Reencode { codec } = &self.video_codec {
            let t = codec.trim().to_string();
            if t.is_empty() {
                self.video_codec = VideoCodec::Copy;
            } else {
                self.video_codec = VideoCodec::Reencode { codec: t };
            }
        }
        // Burned subtitles force a video re-encode — silently upgrade
        // `Copy` to `Reencode { codec: "libx264" }` so the cache key
        // reflects what FFmpeg will actually do.
        if self.subtitle_mode.requires_reencode() && matches!(self.video_codec, VideoCodec::Copy) {
            self.video_codec = VideoCodec::Reencode {
                codec: "libx264".into(),
            };
        }
        if let Some(p) = &self.output_path {
            let t = p.trim().to_string();
            self.output_path = if t.is_empty() { None } else { Some(t) };
        }
        self
    }
}

// -------------------------------------------------------------- summary/status

/// The lifecycle of `output/movie_vi.mp4` as far as the UI cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderStatus {
    /// No render on disk yet.
    Missing,
    /// File on disk but cache identity drifted (upstream changed, or
    /// the user tweaked render settings).
    Stale,
    /// File matches the current cache identity.
    Ready,
}

// -------------------------------------------------------------- manifest

/// The full JSON persisted at `<project>/output/render.json`. There is
/// exactly one active render entry per project.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RenderManifest {
    pub version: u32,
    #[serde(default)]
    pub settings: RenderSettings,
    #[serde(default)]
    pub current: Option<RenderEntry>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl RenderManifest {
    pub fn empty(settings: RenderSettings) -> Self {
        let now = Utc::now();
        Self {
            version: RENDER_CACHE_SCHEMA_VERSION,
            settings,
            current: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// One render entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RenderEntry {
    pub cache_key: String,
    pub source_fingerprint: SourceFingerprint,
    /// Cache key of the mix that was folded into this render — lets
    /// the summary detect stale renders after a mix rerun without
    /// re-probing the source.
    pub mix_cache_key: String,
    /// Absolute path to the produced file.
    pub file_absolute: String,
    /// Path relative to the project root when the file lives inside
    /// the project. `None` when a custom `output_path` outside the
    /// project was used.
    #[serde(default)]
    pub file_relative: Option<String>,
    /// Absolute path to the external SRT sidecar, if any.
    #[serde(default)]
    pub subtitle_file_absolute: Option<String>,
    pub duration_secs: f64,
    pub size_bytes: u64,
    pub video_stream_count: u32,
    pub audio_stream_count: u32,
    pub subtitle_stream_count: u32,
    pub subtitle_mode: SubtitleMode,
    pub settings: RenderSettings,
    pub generated_at: DateTime<Utc>,
}

/// Compact summary the frontend consumes in the render panel — never
/// ships the whole manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderSummary {
    pub status: RenderStatus,
    pub settings: RenderSettings,
    pub duration_secs: Option<f64>,
    pub absolute_path: Option<String>,
    pub relative_path: Option<String>,
    pub subtitle_absolute_path: Option<String>,
    pub size_bytes: Option<u64>,
    pub generated_at: Option<DateTime<Utc>>,
    pub video_stream_count: u32,
    pub audio_stream_count: u32,
    pub subtitle_stream_count: u32,
    pub subtitle_mode: SubtitleMode,
    /// True iff a fresh render is required before the output is up to
    /// date.
    pub needs_render: bool,
    /// The default output path we'd suggest if the user hasn't picked
    /// a custom one — always inside the project's `output/` folder.
    pub default_output_absolute: String,
    /// Any warning text worth surfacing (e.g. `mix is stale`).
    pub warning: Option<String>,
    pub manifest_relative_path: String,
}

// -------------------------------------------------------------- env

/// Snapshot returned by `get_render_env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderEnv {
    pub ffmpeg_available: bool,
    pub ffmpeg_path: Option<String>,
    pub default_settings: RenderSettings,
    /// FFmpeg codec names we advertise as "just works" in the UI.
    pub video_codecs: Vec<String>,
    pub audio_codecs: Vec<String>,
    pub output_formats: Vec<String>,
}

// -------------------------------------------------------------- request

/// Full payload for the `apply_render` command.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderRequest {
    #[serde(default)]
    pub settings: RenderSettings,
    /// When true, ignore the on-disk cache and re-render even if the
    /// cache key matches. Handy for the "Regenerate" button.
    #[serde(default)]
    pub force: bool,
}

/// What `apply_render` returns — either an inline no-op (nothing
/// changed) or a `JobSnapshot` for a fresh run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum RenderGenerateStart {
    UpToDate { summary: RenderSummary },
    Started(JobSnapshot),
}

// -------------------------------------------------------------- cache key

/// Deterministic hash over every input that materially changes
/// `movie_vi.mp4`.
pub fn build_render_cache_key(
    source_fp: &SourceFingerprint,
    mix_cache_key: &str,
    settings: &RenderSettings,
) -> String {
    let s = settings.clone().normalised();
    let mut parts: Vec<String> = Vec::with_capacity(16);
    parts.push(format!("render_v{}", RENDER_CACHE_SCHEMA_VERSION));
    parts.push(format!("src_size={}", source_fp.size_bytes));
    parts.push(format!("src_hash={}", source_fp.hash));
    parts.push(format!("mix={mix_cache_key}"));
    parts.push(format!("fmt={}", s.output_format.extension()));
    parts.push(format!("vcodec={}", s.video_codec.cache_repr()));
    parts.push(format!("acodec={}", s.audio_codec.codec));
    parts.push(format!(
        "abitrate={}",
        s.audio_codec.bitrate.clone().unwrap_or_default()
    ));
    parts.push(format!("submode={}", s.subtitle_mode.as_str()));
    let mut hasher = Sha256::new();
    hasher.update(parts.join("\x1f").as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}
