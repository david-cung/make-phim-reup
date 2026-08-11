//! Audio extraction orchestration (Phase 2).
//!
//! Composes `ffmpeg`, `jobs`, `media::fingerprint` and the per-project
//! `cache/audio_cache.json` into a single service consumed by the
//! commands layer.

pub mod cache;
pub mod extractor;

pub use cache::{AudioCacheEntry, AudioCacheFile, AUDIO_CACHE_FILENAME};
pub use extractor::{
    AudioExtractor, ExtractError, ExtractionRequest, ExtractionResult, ExtractionStart,
};
