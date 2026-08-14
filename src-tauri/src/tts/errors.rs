//! TTS-specific error type.
//!
//! Every failure mode listed in the Phase 6 spec has a matching
//! variant so :mod:`errors::From<TtsError> for AppError` can produce a
//! stable code + hint pair for the frontend.

use thiserror::Error;

use crate::db::DbError;
use crate::jobs::JobRegistryError;
use crate::worker::protocol::RpcError;
use crate::worker::WorkerError;

#[derive(Debug, Error)]
pub enum TtsError {
    #[error("project has no subtitles yet; build subtitles first")]
    NoSubtitles,

    #[error("TTS engine {engine:?} is not available in the worker environment")]
    EngineUnavailable { engine: String },

    /// The worker installed a Python runtime into its own environment and
    /// is being respawned, so the request has to be issued again.
    #[error("{reason}")]
    EngineRestarting { reason: String },

    #[error("voice {voice_id:?} is not installed for engine {engine:?}")]
    VoiceMissing { engine: String, voice_id: String },

    #[error("voice model is invalid or unreadable: {reason}")]
    ModelInvalid { reason: String },

    #[error("cannot synthesise empty or blank text")]
    InvalidText,

    #[error("TTS engine failed: {reason}")]
    EngineFailure { reason: String },

    #[error("out of memory during TTS synthesis")]
    OutOfMemory,

    #[error("disk full while writing generated voice files")]
    DiskFull,

    #[error("worker crashed during TTS synthesis")]
    WorkerCrash,

    #[error("TTS cancelled")]
    Cancelled,

    #[error("segment {segment_id} not found in subtitles")]
    SegmentNotFound { segment_id: u32 },

    #[error("unknown recommended TTS voice preset: {preset:?}")]
    UnknownPreset { preset: String },

    #[error("failed to download TTS voice: {reason}")]
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

impl TtsError {
    /// Best-effort mapping from a Python-side RPC error code back to
    /// our Rust variants. Everything unknown becomes `Worker(...)` so
    /// the UI still sees a structured error.
    pub fn from_rpc(err: RpcError) -> Self {
        match err.code.as_str() {
            "E_CANCELLED" => Self::Cancelled,
            "TTS_ENGINE_UNAVAILABLE" if extract_bool(&err, "restarting") => {
                Self::EngineRestarting {
                    reason: err.message.clone(),
                }
            }
            "TTS_ENGINE_UNAVAILABLE" => Self::EngineUnavailable {
                engine: extract_str(&err, "engine").unwrap_or_default(),
            },
            "TTS_VOICE_MISSING" => Self::VoiceMissing {
                engine: extract_str(&err, "engine").unwrap_or_default(),
                voice_id: extract_str(&err, "voiceId").unwrap_or_default(),
            },
            "TTS_MODEL_INVALID" => Self::ModelInvalid {
                reason: err.message.clone(),
            },
            "TTS_INVALID_TEXT" => Self::InvalidText,
            "TTS_ENGINE_FAILURE" => Self::EngineFailure {
                reason: err.message.clone(),
            },
            "TTS_OUT_OF_MEMORY" => Self::OutOfMemory,
            "TTS_DISK_FULL" => Self::DiskFull,
            "TTS_WORKER_CRASH" => Self::WorkerCrash,
            "TTS_UNKNOWN_PRESET" => Self::UnknownPreset {
                preset: extract_str(&err, "preset").unwrap_or_default(),
            },
            "TTS_DOWNLOAD_FAILED" => Self::DownloadFailed {
                reason: err.message.clone(),
            },
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

fn extract_bool(err: &RpcError, key: &str) -> bool {
    err.data
        .as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
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
    fn maps_unknown_preset_with_data() {
        let err = TtsError::from_rpc(rpc(
            "TTS_UNKNOWN_PRESET",
            "no such voice",
            Some(json!({"preset": "vi_VN-nope-low"})),
        ));
        match err {
            TtsError::UnknownPreset { preset } => assert_eq!(preset, "vi_VN-nope-low"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn maps_restarting_engine_to_its_own_variant() {
        let err = TtsError::from_rpc(rpc(
            "TTS_ENGINE_UNAVAILABLE",
            "F5-TTS runtime installed. Restarting the local engine.",
            Some(json!({"restarting": true, "runtimeInstalled": true})),
        ));
        match err {
            TtsError::EngineRestarting { reason } => {
                assert_eq!(reason, "F5-TTS runtime installed. Restarting the local engine.")
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn maps_download_failed_preserves_message() {
        let err = TtsError::from_rpc(rpc(
            "TTS_DOWNLOAD_FAILED",
            "connection reset by peer",
            None,
        ));
        match err {
            TtsError::DownloadFailed { reason } => {
                assert_eq!(reason, "connection reset by peer")
            }
            _ => panic!("wrong variant"),
        }
    }
}
