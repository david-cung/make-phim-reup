//! Mix-specific error type mirroring the Phase 8 spec's failure modes.

use thiserror::Error;

use crate::db::DbError;
use crate::ffmpeg::FfmpegError;
use crate::jobs::JobRegistryError;

#[derive(Debug, Error)]
pub enum MixError {
    #[error("project has no imported source video")]
    NoSourceMedia,

    #[error("project has no subtitles yet; build subtitles first")]
    NoSubtitles,

    #[error("project has no synced voice segments yet; run voice sync before mixing")]
    NoSyncedVoices,

    #[error("ffmpeg is not available; install it or set a custom path in Settings")]
    FfmpegMissing,

    #[error("mix cancelled")]
    Cancelled,

    #[error("disk full while writing the mix")]
    DiskFull,

    #[error("invalid voice segment {segment_id}: {reason}")]
    InvalidSegment { segment_id: u32, reason: String },

    #[error("voice source wav is missing for segment {segment_id}: {path}")]
    VoiceMissing { segment_id: u32, path: String },

    #[error("mix output wav is empty or unreadable at `{path}`")]
    OutputInvalid { path: String },

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

impl MixError {
    /// Stable error code surfaced to the frontend.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoSourceMedia => "MIX_NO_SOURCE_MEDIA",
            Self::NoSubtitles => "MIX_NO_SUBTITLES",
            Self::NoSyncedVoices => "MIX_NO_SYNCED_VOICES",
            Self::FfmpegMissing => "MIX_FFMPEG_MISSING",
            Self::Cancelled => "MIX_CANCELLED",
            Self::DiskFull => "MIX_DISK_FULL",
            Self::InvalidSegment { .. } => "MIX_INVALID_SEGMENT",
            Self::VoiceMissing { .. } => "MIX_VOICE_MISSING",
            Self::OutputInvalid { .. } => "MIX_OUTPUT_INVALID",
            Self::Ffmpeg(_) => "MIX_FFMPEG",
            Self::Registry(_) => "MIX_REGISTRY",
            Self::Db(_) => "MIX_DB",
            Self::Io { .. } => "MIX_IO",
        }
    }

    pub fn is_recoverable(&self) -> bool {
        !matches!(self, Self::Ffmpeg(FfmpegError::NotFound { .. }))
    }
}
