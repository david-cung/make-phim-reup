//! Phase 8 — Audio mixing.
//!
//! Host-side orchestration only. Unlike Phases 6/7 there is no Python
//! worker involvement — audio mixing is pure FFmpeg, so [`MixService`]
//! spawns a single ffmpeg process, parses its `-progress pipe:1`
//! output, emits `job://progress` events, and honours cancellation via
//! the shared [`crate::jobs::JobHandle`] cancel token.
//!
//! Public shape mirrors Phase 7 (`sync`): a service owns the
//! `<project>/audio/mix.json` manifest, the frontend consumes a compact
//! [`MixSummary`], and dirty-flag clearing goes through
//! [`crate::subtitles::SubtitleService::clear_dirty_flags`] with the
//! single-bit [`crate::subtitles::DirtyFlags::only_mix`] mask.

pub mod cache;
pub mod errors;
pub mod ffmpeg_cmd;
pub mod models;
pub mod service;
#[cfg(test)]
mod tests;

pub use cache::{
    manifest_path, mix_output_path, MixCacheFile, MIX_MANIFEST_FILENAME, MIX_MANIFEST_RELATIVE,
    MIX_OUTPUT_RELATIVE, MIX_SUBDIR,
};
pub use errors::MixError;
pub use ffmpeg_cmd::{build_filter_graph, build_mix_command, MixCommand};
pub use models::{
    build_mix_cache_key, MixEntry, MixEnv, MixGenerateStart, MixManifest, MixMode, MixRequest,
    MixSettings, MixStatus, MixSummary, MixVoiceInput, PreviewMixResult,
};
pub use service::MixService;
