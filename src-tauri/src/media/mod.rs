//! Cross-cutting media utilities used by Phase 2+.
//!
//! * [`formats`] — accepted container extensions and content-type helpers.
//! * [`fingerprint`] — fast, deterministic content fingerprint used as a
//!   cache key. Never hashes the whole file — that's cheap enough for
//!   metadata but ruinous for a 20 GB movie.
//! * [`disk`] — cross-platform available-space check.

pub mod disk;
pub mod fingerprint;
pub mod formats;

pub use fingerprint::{fingerprint_file, SourceFingerprint};
pub use formats::{is_supported_extension, SupportedContainer, SUPPORTED_EXTENSIONS};
