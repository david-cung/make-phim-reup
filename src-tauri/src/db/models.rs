//! Serde-friendly row types. `camelCase` on the wire, `snake_case` in
//! Rust and SQL. Only the tables Phase 1 reads are represented here as
//! typed structs; other tables are created by migrations and will get
//! their structs when the phases that use them land.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProjectStatus {
    Created,
    Ready,
    Processing,
    Error,
    Archived,
}

impl ProjectStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Ready => "ready",
            Self::Processing => "processing",
            Self::Error => "error",
            Self::Archived => "archived",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "created" => Self::Created,
            "ready" => Self::Ready,
            "processing" => Self::Processing,
            "error" => Self::Error,
            "archived" => Self::Archived,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub source_language: String,
    pub target_language: String,
    pub status: ProjectStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_opened_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SourceImportMode {
    Reference,
    Copy,
}

impl SourceImportMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Copy => "copy",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "reference" => Self::Reference,
            "copy" => Self::Copy,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub source_language: String,
    pub target_language: String,
    pub root_path: String,
    pub source_media_path: Option<String>,
    pub status: ProjectStatus,
    pub progress: BTreeMap<String, f32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_opened_at: Option<DateTime<Utc>>,
    // Phase 2 additions
    #[serde(default)]
    pub source_hash: Option<String>,
    #[serde(default)]
    pub source_size: Option<i64>,
    #[serde(default)]
    pub source_modified_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub source_import_mode: Option<SourceImportMode>,
    // Phase 10 — per-project model selection. Every field is
    // optional so pre-Phase-10 projects, and projects the user
    // hasn't configured yet, fall back to the global settings
    // default. Missing referenced models never silently substitute
    // another model — the per-stage service raises a structured
    // error instead.
    #[serde(default)]
    pub whisper_model: Option<String>,
    #[serde(default)]
    pub translation_model: Option<String>,
    #[serde(default)]
    pub tts_engine: Option<String>,
    #[serde(default)]
    pub tts_voice_id: Option<String>,
}

/// Partial patch used by `ProjectService::update_models`. Every field
/// uses `Option<Option<T>>` so `undefined` = "don't change" and
/// `null` = "clear".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectModelPatch {
    #[serde(default)]
    pub whisper_model: Option<Option<String>>,
    #[serde(default)]
    pub translation_model: Option<Option<String>>,
    #[serde(default)]
    pub tts_engine: Option<Option<String>>,
    #[serde(default)]
    pub tts_voice_id: Option<Option<String>>,
}

impl ProjectRecord {
    pub fn to_summary(&self) -> ProjectSummary {
        ProjectSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            source_language: self.source_language.clone(),
            target_language: self.target_language.clone(),
            status: self.status,
            created_at: self.created_at,
            updated_at: self.updated_at,
            last_opened_at: self.last_opened_at,
        }
    }
}
