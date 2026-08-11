//! STT-specific error type.
//!
//! Every failure mode listed in the Phase 3 spec has a matching
//! variant so :mod:`errors::From<SttError> for AppError` can produce
//! a stable code + hint pair for the frontend.

use thiserror::Error;

use crate::db::DbError;
use crate::jobs::JobRegistryError;
use crate::worker::protocol::RpcError;
use crate::worker::WorkerError;

#[derive(Debug, Error)]
pub enum SttError {
    #[error("project has no imported source video")]
    NoSourceMedia,

    #[error("audio/original.wav is missing; extract audio first")]
    AudioNotExtracted,

    #[error("unknown whisper model: {name}")]
    UnknownModel { name: String },

    #[error("whisper model {name:?} is not installed; download it first")]
    ModelNotInstalled { name: String },

    #[error("faster-whisper is not installed in the worker environment")]
    WhisperNotInstalled,

    #[error("failed to load whisper model {name:?}: {reason}")]
    ModelLoadFailed { name: String, reason: String },

    #[error("invalid or unsupported audio: {reason}")]
    InvalidAudio { reason: String },

    #[error("out of memory during transcription")]
    OutOfMemory,

    #[error("worker crashed during transcription")]
    WorkerCrash,

    #[error("transcription cancelled")]
    Cancelled,

    #[error("model download failed: {reason}")]
    DownloadFailed { reason: String },

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

impl SttError {
    /// Best-effort mapping from a Python-side RPC error code to our
    /// Rust variants. Everything unknown becomes ``WorkerCrash`` so
    /// the UI still sees a structured error.
    pub fn from_rpc(err: RpcError) -> Self {
        match err.code.as_str() {
            "E_CANCELLED" => Self::Cancelled,
            "STT_MODEL_NOT_INSTALLED" => Self::ModelNotInstalled {
                name: extract_str(&err, "model").unwrap_or_default(),
            },
            "STT_WHISPER_NOT_INSTALLED" => Self::WhisperNotInstalled,
            "STT_MODEL_LOAD_FAILED" => Self::ModelLoadFailed {
                name: extract_str(&err, "model").unwrap_or_default(),
                reason: err.message.clone(),
            },
            "STT_INVALID_AUDIO" | "STT_UNSUPPORTED_AUDIO" => Self::InvalidAudio {
                reason: err.message.clone(),
            },
            "STT_OUT_OF_MEMORY" => Self::OutOfMemory,
            "STT_UNKNOWN_MODEL" => Self::UnknownModel {
                name: extract_str(&err, "name").unwrap_or_default(),
            },
            "STT_DOWNLOAD_FAILED" => Self::DownloadFailed {
                reason: err.message.clone(),
            },
            "STT_WORKER_CRASH" => Self::WorkerCrash,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rpc(code: &str, msg: &str, data: Option<serde_json::Value>) -> RpcError {
        RpcError {
            code: code.into(),
            message: msg.into(),
            data,
        }
    }

    #[test]
    fn maps_cancelled() {
        let err = SttError::from_rpc(rpc("E_CANCELLED", "cancelled", None));
        assert!(matches!(err, SttError::Cancelled));
    }

    #[test]
    fn maps_out_of_memory() {
        let err = SttError::from_rpc(rpc("STT_OUT_OF_MEMORY", "oom", None));
        assert!(matches!(err, SttError::OutOfMemory));
    }

    #[test]
    fn maps_model_not_installed_with_data() {
        let err = SttError::from_rpc(rpc(
            "STT_MODEL_NOT_INSTALLED",
            "download it",
            Some(json!({"model": "medium"})),
        ));
        match err {
            SttError::ModelNotInstalled { name } => assert_eq!(name, "medium"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn maps_whisper_not_installed() {
        let err = SttError::from_rpc(rpc("STT_WHISPER_NOT_INSTALLED", "no fw", None));
        assert!(matches!(err, SttError::WhisperNotInstalled));
    }

    #[test]
    fn unknown_code_falls_back_to_worker() {
        let err = SttError::from_rpc(rpc("SOMETHING_ELSE", "?", None));
        assert!(matches!(err, SttError::Worker(_)));
    }
}
