//! Structured errors from the subtitle subsystem.
//!
//! Every UI-visible failure lands here; `From<SubtitleError> for
//! AppError` maps each variant to a stable code + hint pair so the
//! frontend can react intelligently.

use thiserror::Error;

use crate::db::DbError;

#[derive(Debug, Error)]
pub enum SubtitleError {
    #[error("project has no transcript yet; run speech recognition first")]
    NoTranscript,

    #[error("no subtitle document exists yet for this project")]
    NoSubtitles,

    #[error("subtitle segment id `{id}` not found")]
    SegmentNotFound { id: u32 },

    #[error("invalid timing: {reason}")]
    InvalidTiming { reason: String },

    #[error(
        "cannot split segment `{id}` at time `{time}`: must lie strictly between start and end"
    )]
    InvalidSplit { id: u32, time: f64 },

    #[error("cannot merge segment `{id}`: no following segment exists")]
    NoMergeTarget { id: u32 },

    #[error("invalid subtitle file at `{path}`: {reason}")]
    InvalidSubtitleFile { path: String, reason: String },

    #[error("unsupported subtitle format for `{path}` (expected .srt or .ass/.ssa)")]
    UnsupportedFormat { path: String },

    #[error("export path is empty or not absolute")]
    InvalidExportPath,

    #[error("db: {0}")]
    Db(#[from] DbError),

    #[error("io error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}
