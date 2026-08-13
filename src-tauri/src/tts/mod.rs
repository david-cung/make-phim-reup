//! Phase 6 — Local Text-to-Speech.
//!
//! Host-side orchestration only. All synthesis happens in the Python
//! worker (`movie_translator_worker.tts`) via Piper (or any future
//! `TTSProvider`), so this crate stays free of ONNX / neural runtime
//! dependencies.
//!
//! Public shape mirrors the other phases (`stt`, `translation`,
//! `subtitles`): a `TtsService` owns the per-project ``voices/`` cache
//! manifest, dispatches synthesis jobs into the worker, and exposes a
//! narrow ``VoiceInfo``/``TtsSummary`` surface to the frontend.

pub mod cache;
pub mod errors;
pub mod models;
pub mod service;
#[cfg(test)]
mod tests;

pub use cache::{TtsCacheFile, VOICES_FILENAME, VOICES_RELATIVE, VOICES_SUBDIR};
pub use errors::TtsError;
pub use models::{
    build_segment_cache_key, text_hash, CreateVoiceProfileRequest, GenerateMode, GenerateRequest,
    PreviewResult, RecommendedVoicePreset, TtsDevice, TtsEngineInfo, TtsEnv, TtsGenerateStart,
    TtsManifest, TtsSegmentEntry, TtsSettings, TtsSummary, VoiceInfo, TTS_CACHE_SCHEMA_VERSION,
};
pub use service::TtsService;
