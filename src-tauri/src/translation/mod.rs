//! Phase 4 — local LLM translation.
//!
//! Host-side orchestration only. All inference happens in the Python
//! worker (`movie_translator_worker.translation`) via
//! `llama-cpp-python`, so this crate stays free of any LLM
//! dependency.

pub mod cache;
pub mod errors;
pub mod models;
pub mod service;
#[cfg(test)]
mod service_tests;

pub use cache::{TranslationCacheFile, TRANSLATION_FILENAME, TRANSLATION_RELATIVE};
pub use errors::TranslationError;
pub use models::{
    RecommendedPreset, TranslateOptions, TranslateStart, TranslatedSegment, TranslationDoc,
    TranslationEnv, TranslationModelInfo, TranslationSummary,
};
pub use service::TranslationService;
