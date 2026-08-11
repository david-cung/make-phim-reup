//! Job lifecycle: enums + registry + DB persistence.
//!
//! Phase 2 uses this only for `extract_audio`; Phase 3+ will queue
//! transcription, translation, TTS, mix and render alongside it. The
//! shape is deliberately reusable.

pub mod persistence;
pub mod registry;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use persistence::{JobRow, JobsRepo};
pub use registry::{JobHandle, JobRegistry, JobRegistryError};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "queued" => Self::Queued,
            "running" => Self::Running,
            "paused" => Self::Paused,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => return None,
        })
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStage {
    ExtractAudio,
    Transcribe,
    Translate,
    Tts,
    /// Phase 7 — timing-align each per-segment TTS WAV to its
    /// subtitle window (silence pad or `atempo` stretch).
    Sync,
    Mix,
    Render,
}

impl JobStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ExtractAudio => "extract_audio",
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
            Self::Tts => "tts",
            Self::Sync => "sync",
            Self::Mix => "mix",
            Self::Render => "render",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "extract_audio" => Self::ExtractAudio,
            "transcribe" => Self::Transcribe,
            "translate" => Self::Translate,
            "tts" => Self::Tts,
            "sync" => Self::Sync,
            "mix" => Self::Mix,
            "render" => Self::Render,
            _ => return None,
        })
    }
}

/// Snapshot emitted on `job://update` events and returned by
/// `list_active_jobs`. Progress lives on the separate `job://progress`
/// channel to avoid flooding the UI with status updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSnapshot {
    pub id: String,
    pub project_id: String,
    pub stage: JobStage,
    pub status: JobStatus,
    pub progress: f32,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Payload of `job://progress` events. Cheap struct — emitted many
/// times per second, so keep it tiny.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProgressEvent {
    pub id: String,
    pub project_id: String,
    pub stage: JobStage,
    pub progress: f32,
}
