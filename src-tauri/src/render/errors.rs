//! Render-specific error type mirroring the Phase 9 spec's failure modes.

use thiserror::Error;

use crate::db::DbError;
use crate::ffmpeg::FfmpegError;
use crate::jobs::JobRegistryError;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("project has no imported source video")]
    NoSourceMedia,

    #[error("project has no subtitles yet; build subtitles first")]
    NoSubtitles,

    #[error("project has no mixed Vietnamese audio yet; run audio mix before rendering")]
    NoMix,

    #[error("mixed audio file is missing on disk at `{path}`")]
    MixFileMissing { path: String },

    #[error("ffmpeg is not available; install it or set a custom path in Settings")]
    FfmpegMissing,

    #[error(
        "this FFmpeg build cannot burn subtitles — it has no `subtitles` filter, \
         which requires libass. Install a full FFmpeg build (on macOS: \
         `brew install ffmpeg`) or point Settings at one, or switch subtitle \
         mode to External to get a .srt file beside the movie instead."
    )]
    SubtitleBurnUnsupported,

    #[error("render cancelled")]
    Cancelled,

    #[error("disk full while writing the final render")]
    DiskFull,

    #[error("output path `{path}` is invalid; provide an absolute file path")]
    InvalidOutputPath { path: String },

    #[error("render output is missing, empty, or corrupt at `{path}`")]
    OutputInvalid { path: String },

    #[error(
        "render validation failed for `{path}`: expected video={want_video}, \
         audio={want_audio}, subtitle={want_subtitle}; got video={got_video}, \
         audio={got_audio}, subtitle={got_subtitle}"
    )]
    ValidationMismatch {
        path: String,
        want_video: bool,
        want_audio: bool,
        want_subtitle: bool,
        got_video: u32,
        got_audio: u32,
        got_subtitle: u32,
    },

    #[error(transparent)]
    Ffmpeg(#[from] FfmpegError),

    #[error(transparent)]
    Registry(#[from] JobRegistryError),

    #[error(transparent)]
    Db(#[from] DbError),

    #[error("io error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl RenderError {
    /// Stable error code surfaced to the frontend.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoSourceMedia => "RENDER_NO_SOURCE_MEDIA",
            Self::NoSubtitles => "RENDER_NO_SUBTITLES",
            Self::NoMix => "RENDER_NO_MIX",
            Self::MixFileMissing { .. } => "RENDER_MIX_MISSING",
            Self::FfmpegMissing => "RENDER_FFMPEG_MISSING",
            Self::SubtitleBurnUnsupported => "RENDER_SUBTITLE_BURN_UNSUPPORTED",
            Self::Cancelled => "RENDER_CANCELLED",
            Self::DiskFull => "RENDER_DISK_FULL",
            Self::InvalidOutputPath { .. } => "RENDER_INVALID_OUTPUT_PATH",
            Self::OutputInvalid { .. } => "RENDER_OUTPUT_INVALID",
            Self::ValidationMismatch { .. } => "RENDER_VALIDATION_FAILED",
            Self::Ffmpeg(_) => "RENDER_FFMPEG",
            Self::Registry(_) => "RENDER_REGISTRY",
            Self::Db(_) => "RENDER_DB",
            Self::Io { .. } => "RENDER_IO",
        }
    }

    pub fn is_recoverable(&self) -> bool {
        !matches!(self, Self::Ffmpeg(FfmpegError::NotFound { .. }))
    }
}
