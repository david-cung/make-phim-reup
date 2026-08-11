//! Phase 7 — Voice / subtitle synchronisation.
//!
//! Host-side orchestration only. The actual audio work (silence
//! padding, FFmpeg `atempo` stretch) happens in the Python worker
//! (`movie_translator_worker.sync`), so this crate stays free of
//! audio-processing dependencies.
//!
//! Public shape mirrors the earlier phases (`stt`, `translation`,
//! `subtitles`, `tts`): a [`SyncService`] owns the per-project
//! `voices/synced/sync.json` manifest, dispatches sync jobs into the
//! worker, and exposes a narrow [`SyncSummary`]/[`SyncManifest`]
//! surface to the frontend.

pub mod cache;
pub mod errors;
pub mod models;
pub mod service;
#[cfg(test)]
mod tests;

pub use cache::{SyncCacheFile, SYNCED_MANIFEST_FILENAME, SYNCED_MANIFEST_RELATIVE, SYNCED_SUBDIR};
pub use errors::SyncError;
pub use models::{
    build_sync_cache_key, PreviewSyncResult, SyncEnv, SyncGenerateStart, SyncManifest, SyncMode,
    SyncPlan, SyncRequest, SyncSegmentEntry, SyncSettings, SyncStatus, SyncSummary,
    SYNC_CACHE_SCHEMA_VERSION,
};
pub use service::SyncService;
