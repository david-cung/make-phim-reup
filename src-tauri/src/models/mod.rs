//! Phase 10 — Local Model Manager.
//!
//! The three per-stage services (Phase 3 STT, Phase 4 translation,
//! Phase 6 TTS) each own their own on-disk registry inside the
//! Python worker. Phase 10 does *not* rewrite those — the worker
//! remains authoritative for what "installed" means for a given
//! stage. Instead, this module:
//!
//! * Aggregates every registry into one flat [`LocalModel`] list so
//!   the Settings › AI Models pane can render every stage in one
//!   table (`ModelRegistry`).
//! * Supports importing a locally-downloaded model into the models
//!   directory (`import_local_model`) with a copy-or-symlink strategy
//!   so multi-gigabyte snapshots are not needlessly duplicated.
//! * Encodes the offline-first contract: `NetworkDisabled` is a
//!   first-class error variant so the download path used by
//!   `stt.download_model` (the only network entrypoint in the app)
//!   can bail out early with a stable error code the UI can render.
//!
//! Everything here is metadata-only — model binaries never load into
//! Rust; that's the worker's job on the actual inference paths.

pub mod errors;
pub mod import;
pub mod registry;

pub use errors::ModelManagerError;
pub use import::{import_local_model, ImportSpec, ImportStrategy};
pub use registry::{
    probe_writable, LocalModel, ModelDirectoryInfo, ModelKind, ModelRegistry, ModelStatus,
};
