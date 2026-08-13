//! Persistence for `<project>/voices/voices.json`.
//!
//! Atomic write (tmp + rename) so a crash mid-write can never corrupt
//! the manifest. Individual `.wav` files live next to it and are
//! written directly by the Python worker.

use std::io;
use std::path::{Path, PathBuf};

use super::models::{TtsManifest, TTS_CACHE_SCHEMA_VERSION};

pub const VOICES_SUBDIR: &str = "voices";
pub const VOICES_FILENAME: &str = "voices.json";
pub const VOICES_RELATIVE: &str = "voices/voices.json";

#[derive(Debug, Clone, Copy)]
pub struct TtsCacheFile;

impl TtsCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<TtsManifest>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: TtsManifest = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid voices.json: {e}")))?;
                if doc.version == TTS_CACHE_SCHEMA_VERSION {
                    Ok(Some(doc))
                } else {
                    Ok(None)
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &TtsManifest) -> io::Result<PathBuf> {
        let path = manifest_path(project_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let text =
            serde_json::to_string_pretty(doc).map_err(|e| io::Error::other(e.to_string()))?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(path)
    }
}

pub fn manifest_path(project_root: &Path) -> PathBuf {
    project_root.join(VOICES_SUBDIR).join(VOICES_FILENAME)
}

pub fn voices_dir(project_root: &Path) -> PathBuf {
    project_root.join(VOICES_SUBDIR)
}
