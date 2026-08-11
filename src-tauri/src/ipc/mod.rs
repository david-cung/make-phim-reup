//! Wire types exposed via Tauri commands. Kept in one place so the
//! TypeScript mirror (`src/ipc/types.ts`) is easy to keep in sync.

use serde::Serialize;

use crate::audio::AudioCacheEntry;
use crate::ffmpeg::VideoMetadata;
use crate::jobs::JobSnapshot;
use crate::mix::MixSummary;
use crate::render::RenderSummary;
use crate::stt::TranscriptSummary;
use crate::subtitles::SubtitleSummary;
use crate::sync::SyncSummary;
use crate::translation::TranslationSummary;
use crate::tts::TtsSummary;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppInfo {
    pub app_name: String,
    pub app_version: String,
    pub data_dir: String,
    pub config_dir: String,
    pub log_dir: String,
    pub projects_dir: String,
    pub models_dir: String,
    pub os: String,
    pub arch: String,
}

/// Everything the Project screen needs to render its media panel:
/// the probed metadata (if any), the cached audio entry (if any),
/// and any live jobs (extraction currently running).
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ProjectMediaState {
    pub metadata: Option<VideoMetadata>,
    pub audio: Option<AudioCacheEntry>,
    pub audio_absolute_path: Option<String>,
    pub transcript: Option<TranscriptSummary>,
    pub translation: Option<TranslationSummary>,
    pub subtitles: Option<SubtitleSummary>,
    pub tts: Option<TtsSummary>,
    pub sync: Option<SyncSummary>,
    pub mix: Option<MixSummary>,
    pub render: Option<RenderSummary>,
    pub active_jobs: Vec<JobSnapshot>,
}
