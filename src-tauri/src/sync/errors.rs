//! Sync-specific error type mirroring the Phase 7 spec's failure modes.

use thiserror::Error;

use crate::db::DbError;
use crate::jobs::JobRegistryError;
use crate::worker::protocol::RpcError;
use crate::worker::WorkerError;

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("project has no subtitles yet; build subtitles first")]
    NoSubtitles,

    #[error("project has no TTS voices yet; generate TTS before syncing")]
    NoTts,

    #[error("segment {segment_id} not found in subtitles")]
    SegmentNotFound { segment_id: u32 },

    #[error("segment {segment_id} has no TTS voice yet")]
    SegmentTtsMissing { segment_id: u32 },

    #[error("segment {segment_id} has invalid subtitle timing: {reason}")]
    InvalidTiming { segment_id: u32, reason: String },

    #[error("source TTS wav is missing for segment {segment_id}: {path}")]
    SourceMissing { segment_id: u32, path: String },

    #[error("source TTS wav is unreadable for segment {segment_id}: {reason}")]
    SourceInvalid { segment_id: u32, reason: String },

    #[error("ffmpeg is not available; install it or set a custom path in Settings")]
    FfmpegMissing,

    #[error("ffmpeg failed while syncing: {reason}")]
    EngineFailure { reason: String },

    #[error("disk full while writing synced voice files")]
    DiskFull,

    #[error("worker crashed during sync")]
    WorkerCrash,

    #[error("sync cancelled")]
    Cancelled,

    #[error("worker error: {0}")]
    Worker(#[from] WorkerError),

    #[error("job registry: {0}")]
    Registry(#[from] JobRegistryError),

    #[error("db: {0}")]
    Db(#[from] DbError),

    #[error("io error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl SyncError {
    /// Best-effort mapping from a Python-side RPC error code back to
    /// our Rust variants. Everything unknown becomes `Worker(...)` so
    /// the UI still sees a structured error.
    pub fn from_rpc(err: RpcError) -> Self {
        match err.code.as_str() {
            "E_CANCELLED" => Self::Cancelled,
            "SYNC_FFMPEG_MISSING" => Self::FfmpegMissing,
            "SYNC_SOURCE_MISSING" => Self::SourceMissing {
                segment_id: extract_u32(&err, "segmentId").unwrap_or(0),
                path: extract_str(&err, "sourceFile").unwrap_or_default(),
            },
            "SYNC_SOURCE_INVALID" => Self::SourceInvalid {
                segment_id: extract_u32(&err, "segmentId").unwrap_or(0),
                reason: err.message.clone(),
            },
            "SYNC_INVALID_TIMING" => Self::InvalidTiming {
                segment_id: extract_u32(&err, "segmentId").unwrap_or(0),
                reason: err.message.clone(),
            },
            "SYNC_ENGINE_FAILURE" => Self::EngineFailure {
                reason: err.message.clone(),
            },
            "SYNC_DISK_FULL" => Self::DiskFull,
            "SYNC_WORKER_CRASH" => Self::WorkerCrash,
            _ => Self::Worker(WorkerError::Rpc(err)),
        }
    }
}

fn extract_str(err: &RpcError, key: &str) -> Option<String> {
    err.data
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn extract_u32(err: &RpcError, key: &str) -> Option<u32> {
    err.data
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
}
