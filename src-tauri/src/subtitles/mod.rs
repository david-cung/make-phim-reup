//! Phase 5 — canonical subtitle model + editor.
//!
//! This module owns the *canonical* subtitle representation used from
//! Phase 5 onwards: `id, start, end, sourceText, translatedText,
//! speaker?, voiceId?`. Everything downstream (Phase 6 TTS, Phase 8
//! mix, Phase 9 render) reads from here — the transcript and
//! translation files remain untouched.
//!
//! Nothing in here talks to the Python worker: subtitle math is pure
//! Rust, no AI, no external processes.

pub mod ass;
pub mod cache;
pub mod derive;
pub mod errors;
pub mod models;
pub mod service;
pub mod srt;
#[cfg(test)]
mod tests;

pub use cache::{SubtitleCacheFile, SUBTITLES_FILENAME, SUBTITLES_RELATIVE};
pub use errors::SubtitleError;
pub use models::{
    DirtyFlags, ExportKind, ExportSubtitlesResult, ImportSubtitlesResult, SubtitleDoc,
    SubtitleFormat, SubtitleSegment, SubtitleSegmentPatch, SubtitleSummary, SubtitleWord,
};
pub use service::SubtitleService;
