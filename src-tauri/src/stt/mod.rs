//! Phase 3 — local speech-to-text.
//!
//! Everything here is *host-side* orchestration only. The actual
//! inference lives in the Python worker (`movie_translator_worker.stt`)
//! so the Rust side stays free of any AI dependency.

pub mod cache;
pub mod errors;
pub mod models;
pub mod service;
#[cfg(test)]
mod service_tests;

pub use cache::{TranscriptCacheFile, TRANSCRIPT_FILENAME, TRANSCRIPT_RELATIVE};
pub use errors::SttError;
pub use models::{
    ModelInfo, SttDeviceInfo, SttEnv, SttOptions, TranscribeSegment, TranscribeStart, Transcript,
    TranscriptAudio, TranscriptSummary, WhisperWord,
};
pub use service::SttService;
