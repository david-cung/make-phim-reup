//! Thin, typed wrappers around the FFmpeg CLI.
//!
//! Everything in this module composes the `ffmpeg` and `ffprobe`
//! executables via `tokio::process::Command`. There is intentionally no
//! codec logic, no muxing math, no timing math — that lives in the
//! callers that string these primitives together (Phase 2's
//! `audio::extractor`, Phase 8's mixer, Phase 9's renderer).

pub mod detection;
pub mod errors;
pub mod extract;
pub mod probe;
pub mod progress;

pub use detection::{FfmpegAvailability, FfmpegService};
pub use errors::FfmpegError;
pub use extract::{build_extract_command, AudioExtractParams, ExtractionOutcome};
pub use probe::{probe_video, StreamSummary, VideoMetadata};
pub use progress::{parse_progress_block, ProgressBlock};
