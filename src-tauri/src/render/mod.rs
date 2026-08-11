//! Phase 9 — Final video rendering.
//!
//! Host-side orchestration only. Like Phase 8 (`mix`) there is no
//! Python worker involvement — final rendering is pure FFmpeg, so
//! [`RenderService`] spawns a single ffmpeg process, parses its
//! `-progress pipe:1` output, emits `job://progress` events, and
//! honours cancellation via the shared [`crate::jobs::JobHandle`]
//! cancel token.
//!
//! Public shape mirrors Phase 8: a service owns the
//! `<project>/output/render.json` manifest, the frontend consumes a
//! compact [`RenderSummary`], and dirty-flag clearing goes through
//! [`crate::subtitles::SubtitleService::clear_dirty_flags`] with the
//! single-bit [`crate::subtitles::DirtyFlags::only_render`] mask.
//!
//! Post-render validation uses `ffprobe` to confirm the output file
//! exists, has non-zero size, has a positive duration, and contains
//! the expected video / audio streams (subtitle stream tracking is
//! done via the sidecar SRT since we don't mux subtitles into the
//! container in this release).

pub mod cache;
pub mod errors;
pub mod ffmpeg_cmd;
pub mod models;
pub mod service;
#[cfg(test)]
mod tests;

pub use cache::{
    default_render_output_path, default_subtitle_output_path, manifest_path, RenderCacheFile,
    RENDER_MANIFEST_FILENAME, RENDER_MANIFEST_RELATIVE, RENDER_OUTPUT_BASENAME, RENDER_SUBDIR,
    RENDER_SUBTITLE_BASENAME,
};
pub use errors::RenderError;
pub use ffmpeg_cmd::{build_render_command, build_subtitles_filter, RenderCommand};
pub use models::{
    build_render_cache_key, AudioCodec, OutputFormat, RenderEntry, RenderEnv, RenderGenerateStart,
    RenderManifest, RenderRequest, RenderSettings, RenderStatus, RenderSummary, SubtitleMode,
    VideoCodec, RENDER_CACHE_SCHEMA_VERSION,
};
pub use service::RenderService;
