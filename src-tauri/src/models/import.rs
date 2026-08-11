//! Phase 10 — importing a locally-downloaded model into the models
//! directory.
//!
//! The spec asks us to "avoid unnecessary copying of very large
//! models". We support two strategies:
//!
//! * `Link` (default) — creates a symlink at the destination. Fast,
//!   zero-copy, works on macOS/Linux out of the box. On Windows we
//!   attempt `symlink_file` / `symlink_dir` and gracefully fall back
//!   to `Copy` if the platform refuses (e.g. no developer mode).
//! * `Copy` — walks the source and copies bytes. Slow for very large
//!   models but guarantees the models directory is self-contained
//!   (useful for moving projects between machines).
//!
//! We never overwrite an existing destination — the caller must
//! delete it first. This is deliberate: the user gets a clear
//! "already exists" error, not a silent replace.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::errors::ModelManagerError;
use super::registry::{LocalModel, ModelKind, ModelStatus};
use crate::paths;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImportStrategy {
    /// Prefer a symlink (falls back to copy on Windows without dev mode).
    #[default]
    Link,
    /// Always copy bytes.
    Copy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportSpec {
    pub kind: ModelKind,
    /// Absolute path to the source file or directory the user picked.
    pub source_path: String,
    /// Optional override for the on-disk name (e.g. rename a GGUF at
    /// import time). Empty ⇒ derive from the source's basename.
    #[serde(default)]
    pub name: Option<String>,
    /// Voice models live under `<models>/tts/<engine>/`. Ignored for
    /// other kinds. Defaults to `piper`.
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub strategy: ImportStrategy,
}

/// Perform the import and return the freshly-registered model entry.
/// The caller is expected to invalidate the registry cache after this
/// returns.
pub fn import_local_model(
    models_root: &Path,
    spec: ImportSpec,
) -> Result<LocalModel, ModelManagerError> {
    let src = PathBuf::from(spec.source_path.trim());
    if src.as_os_str().is_empty() || !src.is_absolute() {
        return Err(ModelManagerError::InvalidSourcePath);
    }
    if !src.exists() {
        return Err(ModelManagerError::SourceNotFound { path: src });
    }

    match spec.kind {
        ModelKind::Whisper => import_whisper(models_root, &src, spec.name, spec.strategy),
        ModelKind::Translation => import_translation(models_root, &src, spec.name, spec.strategy),
        ModelKind::Tts | ModelKind::Voice => import_voice(
            models_root,
            &src,
            spec.name,
            spec.engine.as_deref().unwrap_or("piper"),
            spec.strategy,
        ),
    }
}

// ------------------------------------------------------------------ whisper

fn import_whisper(
    models_root: &Path,
    src: &Path,
    name_override: Option<String>,
    strategy: ImportStrategy,
) -> Result<LocalModel, ModelManagerError> {
    if !src.is_dir() {
        return Err(ModelManagerError::UnsupportedSource {
            kind: "whisper",
            reason: "expected a Whisper snapshot directory (CTranslate2 layout)".into(),
        });
    }
    for required in ["model.bin", "config.json"] {
        let f = src.join(required);
        if !f.is_file() {
            return Err(ModelManagerError::MissingRequiredFile {
                file: required.into(),
                path: src.to_path_buf(),
            });
        }
    }

    let name = name_override.unwrap_or_else(|| dirname_or_default(src, "whisper-model"));
    validate_flat_name(&name)?;

    let root = models_root.join("whisper");
    ensure_dir_writable(&root)?;
    let dest = root.join(&name);
    if dest.exists() {
        return Err(ModelManagerError::AlreadyExists { name, path: dest });
    }

    place_directory(src, &dest, strategy)?;

    let size = dir_size(&dest).ok();
    Ok(LocalModel {
        id: format!("whisper:{name}"),
        name,
        kind: ModelKind::Whisper,
        engine: Some("faster-whisper".into()),
        language: None,
        path: Some(dest.display().to_string()),
        size_bytes: size,
        version: None,
        status: ModelStatus::Available,
        hint: None,
    })
}

// -------------------------------------------------------------- translation

fn import_translation(
    models_root: &Path,
    src: &Path,
    name_override: Option<String>,
    strategy: ImportStrategy,
) -> Result<LocalModel, ModelManagerError> {
    if !src.is_file() {
        return Err(ModelManagerError::UnsupportedSource {
            kind: "translation",
            reason: "expected a .gguf file".into(),
        });
    }
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if ext != "gguf" {
        return Err(ModelManagerError::UnsupportedSource {
            kind: "translation",
            reason: format!("expected a .gguf file (got .{ext})"),
        });
    }
    match std::fs::metadata(src) {
        Ok(md) if md.len() == 0 => {
            return Err(ModelManagerError::UnsupportedSource {
                kind: "translation",
                reason: "file is empty".into(),
            })
        }
        Err(e) => {
            return Err(ModelManagerError::Unreadable {
                reason: e.to_string(),
            })
        }
        _ => {}
    }

    let name = name_override.unwrap_or_else(|| filename_or_default(src, "model.gguf"));
    validate_flat_name(&name)?;
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(ModelManagerError::InvalidName { name });
    }

    let root = models_root.join("translation");
    ensure_dir_writable(&root)?;
    let dest = root.join(&name);
    if dest.exists() {
        return Err(ModelManagerError::AlreadyExists { name, path: dest });
    }

    place_file(src, &dest, strategy)?;

    let size = std::fs::metadata(&dest).map(|md| md.len()).unwrap_or(0);
    Ok(LocalModel {
        id: format!("translation:{name}"),
        name,
        kind: ModelKind::Translation,
        engine: Some("llama.cpp".into()),
        language: None,
        path: Some(dest.display().to_string()),
        size_bytes: Some(size),
        version: None,
        status: ModelStatus::Available,
        hint: None,
    })
}

// -------------------------------------------------------------------- voice

fn import_voice(
    models_root: &Path,
    src: &Path,
    name_override: Option<String>,
    engine: &str,
    strategy: ImportStrategy,
) -> Result<LocalModel, ModelManagerError> {
    validate_flat_name(engine)?;

    // A voice may be given as either a directory (preferred, contains
    // model.onnx + *.onnx.json) or as the .onnx file itself (we then
    // copy its sibling .onnx.json next to it).
    let (src_dir, primary_onnx) = if src.is_dir() {
        let onnx =
            first_matching(src, "onnx").ok_or_else(|| ModelManagerError::UnsupportedSource {
                kind: "tts voice",
                reason: "directory does not contain a .onnx file".into(),
            })?;
        (src.to_path_buf(), onnx)
    } else if src.is_file()
        && src
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.eq_ignore_ascii_case("onnx"))
            .unwrap_or(false)
    {
        (
            src.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
            src.to_path_buf(),
        )
    } else {
        return Err(ModelManagerError::UnsupportedSource {
            kind: "tts voice",
            reason: "expected a voice directory or a .onnx file".into(),
        });
    };

    match std::fs::metadata(&primary_onnx) {
        Ok(md) if md.len() == 0 => {
            return Err(ModelManagerError::UnsupportedSource {
                kind: "tts voice",
                reason: "ONNX file is empty".into(),
            })
        }
        Err(e) => {
            return Err(ModelManagerError::Unreadable {
                reason: e.to_string(),
            })
        }
        _ => {}
    }

    let name = name_override.unwrap_or_else(|| dirname_or_default(&src_dir, "voice"));
    validate_flat_name(&name)?;

    let root = models_root.join("tts").join(engine);
    ensure_dir_writable(&root)?;
    let dest = root.join(&name);
    if dest.exists() {
        return Err(ModelManagerError::AlreadyExists { name, path: dest });
    }

    place_directory(&src_dir, &dest, strategy)?;

    let size = std::fs::metadata(dest.join(primary_onnx.file_name().unwrap_or_default()))
        .map(|md| md.len())
        .ok();

    Ok(LocalModel {
        id: format!("tts:{engine}:{name}"),
        name,
        kind: ModelKind::Voice,
        engine: Some(engine.to_string()),
        language: None,
        path: Some(dest.display().to_string()),
        size_bytes: size,
        version: None,
        status: ModelStatus::Available,
        hint: None,
    })
}

// -------------------------------------------------------------- placement

fn place_file(src: &Path, dest: &Path, strategy: ImportStrategy) -> Result<(), ModelManagerError> {
    if matches!(strategy, ImportStrategy::Link) {
        if let Some(parent) = dest.parent() {
            ensure_dir_writable(parent)?;
        }
        if symlink_file(src, dest).is_ok() {
            return Ok(());
        }
        tracing::warn!(
            "symlink failed, falling back to copy for {}",
            dest.display()
        );
    }
    std::fs::copy(src, dest).map_err(|source| io_err(dest, source))?;
    Ok(())
}

fn place_directory(
    src: &Path,
    dest: &Path,
    strategy: ImportStrategy,
) -> Result<(), ModelManagerError> {
    if matches!(strategy, ImportStrategy::Link) {
        if let Some(parent) = dest.parent() {
            ensure_dir_writable(parent)?;
        }
        if symlink_dir(src, dest).is_ok() {
            return Ok(());
        }
        tracing::warn!(
            "directory symlink failed, falling back to copy for {}",
            dest.display()
        );
    }
    copy_dir_recursive(src, dest)?;
    Ok(())
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), ModelManagerError> {
    std::fs::create_dir_all(dest).map_err(|source| io_err(dest, source))?;
    let entries = std::fs::read_dir(src).map_err(|source| io_err(src, source))?;
    for entry in entries {
        let entry = entry.map_err(|source| io_err(src, source))?;
        let file_type = entry.file_type().map_err(|source| io_err(src, source))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if file_type.is_file() {
            std::fs::copy(&from, &to).map_err(|source| io_err(&to, source))?;
        }
        // Ignore other types (fifos, sockets) — they'll never be legitimate
        // pieces of a model snapshot.
    }
    Ok(())
}

fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            total = total.saturating_add(dir_size(&entry.path())?);
        } else if ft.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

#[cfg(unix)]
fn symlink_file(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dest)
}

#[cfg(unix)]
fn symlink_dir(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dest)
}

#[cfg(windows)]
fn symlink_file(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(src, dest)
}

#[cfg(windows)]
fn symlink_dir(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(src, dest)
}

// ------------------------------------------------------------------ helpers

fn dirname_or_default(p: &Path, fallback: &str) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn filename_or_default(p: &Path, fallback: &str) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn first_matching(dir: &Path, ext: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut hits: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case(ext))
                    .unwrap_or(false)
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

fn validate_flat_name(name: &str) -> Result<(), ModelManagerError> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(ModelManagerError::InvalidName {
            name: name.to_string(),
        });
    }
    Ok(())
}

fn ensure_dir_writable(dir: &Path) -> Result<(), ModelManagerError> {
    paths::ensure_dir(dir).map_err(|e| match e {
        paths::PathError::Io { path, source } => ModelManagerError::Io {
            path: PathBuf::from(path),
            source,
        },
        other => ModelManagerError::ModelsDirNotWritable {
            path: dir.to_path_buf(),
            reason: other.to_string(),
        },
    })
}

fn io_err(path: &Path, source: std::io::Error) -> ModelManagerError {
    if source.kind() == std::io::ErrorKind::PermissionDenied {
        return ModelManagerError::PermissionDenied {
            path: path.to_path_buf(),
        };
    }
    ModelManagerError::Io {
        path: path.to_path_buf(),
        source,
    }
}
