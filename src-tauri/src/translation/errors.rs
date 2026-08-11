//! Translation-specific error type.
//!
//! Every failure mode listed in the Phase 4 spec has a matching
//! variant so :mod:`errors::From<TranslationError> for AppError` can
//! produce a stable code + hint pair for the frontend.

use thiserror::Error;

use crate::db::DbError;
use crate::jobs::JobRegistryError;
use crate::worker::protocol::RpcError;
use crate::worker::WorkerError;

#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("project has no transcript yet; run speech recognition first")]
    NoTranscript,

    #[error("translation model {name:?} is not installed at <models>/translation/")]
    ModelNotInstalled { name: String },

    #[error("llama-cpp-python is not installed in the worker environment")]
    LlamaNotInstalled,

    #[error("failed to load GGUF model {name:?}: {reason}")]
    ModelLoadFailed { name: String, reason: String },

    #[error("LLM returned invalid JSON: {reason}")]
    InvalidJson { reason: String },

    #[error("LLM returned an incomplete response: {reason}")]
    IncompleteResponse { reason: String },

    #[error("out of memory during translation")]
    OutOfMemory,

    #[error("llama.cpp inference failed: {reason}")]
    LlmFailure { reason: String },

    #[error("worker crashed during translation")]
    WorkerCrash,

    #[error("translation cancelled")]
    Cancelled,

    #[error("unknown prompt version: {name:?}")]
    UnknownPromptVersion { name: String },

    #[error("unknown recommended translation preset: {preset:?}")]
    UnknownPreset { preset: String },

    #[error("failed to download translation model: {reason}")]
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

impl TranslationError {
    /// Best-effort mapping from a Python-side RPC error code to our
    /// Rust variants. Everything unknown becomes ``Worker(...)`` so
    /// the UI still sees a structured error.
    pub fn from_rpc(err: RpcError) -> Self {
        match err.code.as_str() {
            "E_CANCELLED" => Self::Cancelled,
            "TRANSLATE_LLAMA_NOT_INSTALLED" => Self::LlamaNotInstalled,
            "TRANSLATE_MODEL_NOT_INSTALLED" => Self::ModelNotInstalled {
                name: extract_str(&err, "model").unwrap_or_default(),
            },
            "TRANSLATE_MODEL_LOAD_FAILED" => Self::ModelLoadFailed {
                name: extract_str(&err, "model").unwrap_or_default(),
                reason: err.message.clone(),
            },
            "TRANSLATE_UNKNOWN_PROMPT" => Self::UnknownPromptVersion {
                name: extract_str(&err, "name").unwrap_or_default(),
            },
            "TRANSLATE_INVALID_JSON" => Self::InvalidJson {
                reason: err.message.clone(),
            },
            "TRANSLATE_INCOMPLETE_RESPONSE" => Self::IncompleteResponse {
                reason: err.message.clone(),
            },
            "TRANSLATE_OUT_OF_MEMORY" => Self::OutOfMemory,
            "TRANSLATE_LLM_FAILURE" => Self::LlmFailure {
                reason: err.message.clone(),
            },
            "TRANSLATE_WORKER_CRASH" => Self::WorkerCrash,
            "TRANSLATE_UNKNOWN_PRESET" => Self::UnknownPreset {
                preset: extract_str(&err, "preset").unwrap_or_default(),
            },
            "TRANSLATE_DOWNLOAD_FAILED" => Self::DownloadFailed {
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
        let err = TranslationError::from_rpc(rpc("E_CANCELLED", "cancelled", None));
        assert!(matches!(err, TranslationError::Cancelled));
    }

    #[test]
    fn maps_llama_not_installed() {
        let err =
            TranslationError::from_rpc(rpc("TRANSLATE_LLAMA_NOT_INSTALLED", "no llama", None));
        assert!(matches!(err, TranslationError::LlamaNotInstalled));
    }

    #[test]
    fn maps_model_not_installed_with_data() {
        let err = TranslationError::from_rpc(rpc(
            "TRANSLATE_MODEL_NOT_INSTALLED",
            "install it",
            Some(json!({"model": "qwen2.gguf"})),
        ));
        match err {
            TranslationError::ModelNotInstalled { name } => assert_eq!(name, "qwen2.gguf"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn unknown_code_falls_back_to_worker() {
        let err = TranslationError::from_rpc(rpc("SOMETHING_ELSE", "?", None));
        assert!(matches!(err, TranslationError::Worker(_)));
    }

    #[test]
    fn maps_unknown_preset_with_data() {
        let err = TranslationError::from_rpc(rpc(
            "TRANSLATE_UNKNOWN_PRESET",
            "no such preset",
            Some(json!({"preset": "qwen2.5-999b"})),
        ));
        match err {
            TranslationError::UnknownPreset { preset } => {
                assert_eq!(preset, "qwen2.5-999b")
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn maps_download_failed_preserves_message() {
        let err = TranslationError::from_rpc(rpc(
            "TRANSLATE_DOWNLOAD_FAILED",
            "connection reset by peer",
            None,
        ));
        match err {
            TranslationError::DownloadFailed { reason } => {
                assert_eq!(reason, "connection reset by peer")
            }
            _ => panic!("wrong variant"),
        }
    }
}
