//! Phase 10 — errors for the local model manager.
//!
//! Every failure is either recoverable (the user can fix it in the
//! Model Manager UI) or a real IO/permission issue. Errors carry the
//! stable code that the frontend maps to messages + hints in
//! `errors.rs`.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelManagerError {
    #[error("unsupported model kind `{kind}`")]
    UnsupportedKind { kind: String },

    #[error("model source path is not absolute or is empty")]
    InvalidSourcePath,

    #[error("model source does not exist: `{path}`")]
    SourceNotFound { path: PathBuf },

    #[error("model source is not a supported {kind} artefact ({reason})")]
    UnsupportedSource { kind: &'static str, reason: String },

    #[error("model source is missing required file `{file}` under `{path}`")]
    MissingRequiredFile { file: String, path: PathBuf },

    #[error("model source is not readable ({reason})")]
    Unreadable { reason: String },

    #[error("model name `{name}` is not valid (must not contain path separators or `..`)")]
    InvalidName { name: String },

    #[error("a model named `{name}` already exists at `{path}`; remove it first")]
    AlreadyExists { name: String, path: PathBuf },

    #[error("permission denied at `{path}`")]
    PermissionDenied { path: PathBuf },

    #[error("model directory `{path}` is not writable ({reason})")]
    ModelsDirNotWritable { path: PathBuf, reason: String },

    #[error("network access is disabled by Offline Mode; install the model manually")]
    NetworkDisabled,

    #[error("io error at `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl ModelManagerError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedKind { .. } => "MODEL_UNSUPPORTED_KIND",
            Self::InvalidSourcePath => "MODEL_INVALID_SOURCE_PATH",
            Self::SourceNotFound { .. } => "MODEL_SOURCE_NOT_FOUND",
            Self::UnsupportedSource { .. } => "MODEL_UNSUPPORTED_SOURCE",
            Self::MissingRequiredFile { .. } => "MODEL_MISSING_REQUIRED_FILE",
            Self::Unreadable { .. } => "MODEL_UNREADABLE",
            Self::InvalidName { .. } => "MODEL_INVALID_NAME",
            Self::AlreadyExists { .. } => "MODEL_ALREADY_EXISTS",
            Self::PermissionDenied { .. } => "MODEL_PERMISSION_DENIED",
            Self::ModelsDirNotWritable { .. } => "MODEL_DIR_NOT_WRITABLE",
            Self::NetworkDisabled => "MODEL_NETWORK_DISABLED",
            Self::Io { .. } => "MODEL_IO",
        }
    }

    pub fn is_recoverable(&self) -> bool {
        !matches!(self, Self::Io { .. })
    }
}
