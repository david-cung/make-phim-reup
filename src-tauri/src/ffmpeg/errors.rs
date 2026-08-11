//! FFmpeg-specific error taxonomy. Each variant maps 1:1 to a stable
//! `AppError.code` so the UI can show useful, localisable messages.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FfmpegError {
    #[error("FFmpeg is not installed or not reachable at `{path}`")]
    NotFound { path: PathBuf },

    #[error("FFprobe is not installed or not reachable at `{path}`")]
    ProbeNotFound { path: PathBuf },

    #[error("FFmpeg version could not be determined: {details}")]
    VersionUnknown { details: String },

    #[error("input file does not exist: `{path}`")]
    InputMissing { path: PathBuf },

    #[error("unsupported source extension: `{ext}`")]
    UnsupportedExtension { ext: String },

    #[error("ffprobe returned invalid JSON: {details}")]
    ProbeParse { details: String },

    #[error("ffprobe failed (exit {code}): {stderr}")]
    ProbeFailed { code: i32, stderr: String },

    #[error("input has no audio stream")]
    NoAudioStream,

    #[error("input has no video stream")]
    NoVideoStream,

    #[error("input appears corrupted or is an unsupported codec: {details}")]
    UnsupportedCodec { details: String },

    #[error("ffmpeg failed (exit {code}): {stderr}")]
    RunFailed { code: i32, stderr: String },

    #[error("ffmpeg produced no output at `{path}`")]
    NoOutput { path: PathBuf },

    #[error("io error during {ctx}: {source}")]
    Io {
        ctx: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("permission denied writing to `{path}`")]
    PermissionDenied { path: PathBuf },

    #[error(
        "insufficient disk space: {required_bytes} bytes needed, {available_bytes} bytes free"
    )]
    DiskSpaceLow {
        required_bytes: u64,
        available_bytes: u64,
    },

    #[error("operation was cancelled")]
    Cancelled,
}

impl FfmpegError {
    pub fn io(ctx: &'static str, source: std::io::Error) -> Self {
        Self::Io { ctx, source }
    }

    /// Stable code string used by AppError. Keep in sync with `errors.rs`.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "FFMPEG_NOT_FOUND",
            Self::ProbeNotFound { .. } => "FFPROBE_NOT_FOUND",
            Self::VersionUnknown { .. } => "FFMPEG_VERSION_UNKNOWN",
            Self::InputMissing { .. } => "INPUT_MISSING",
            Self::UnsupportedExtension { .. } => "UNSUPPORTED_EXTENSION",
            Self::ProbeParse { .. } => "PROBE_PARSE",
            Self::ProbeFailed { .. } => "PROBE_FAILED",
            Self::NoAudioStream => "NO_AUDIO_STREAM",
            Self::NoVideoStream => "NO_VIDEO_STREAM",
            Self::UnsupportedCodec { .. } => "UNSUPPORTED_CODEC",
            Self::RunFailed { .. } => "FFMPEG_RUN_FAILED",
            Self::NoOutput { .. } => "FFMPEG_NO_OUTPUT",
            Self::Io { .. } => "FFMPEG_IO",
            Self::PermissionDenied { .. } => "PERMISSION_DENIED",
            Self::DiskSpaceLow { .. } => "DISK_SPACE_LOW",
            Self::Cancelled => "CANCELLED",
        }
    }

    /// A recoverable error is one the user can potentially fix themselves
    /// (install ffmpeg, free disk, pick another file). Fatal errors are
    /// truly unexpected — most of the io/unknown-codec category.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NotFound { .. }
                | Self::ProbeNotFound { .. }
                | Self::InputMissing { .. }
                | Self::UnsupportedExtension { .. }
                | Self::NoAudioStream
                | Self::NoVideoStream
                | Self::UnsupportedCodec { .. }
                | Self::PermissionDenied { .. }
                | Self::DiskSpaceLow { .. }
                | Self::Cancelled
        )
    }
}
