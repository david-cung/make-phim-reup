//! Persistence for `<project>/voices/synced/sync.json`.
//!
//! Atomic write (tmp + rename) so a crash mid-write can never corrupt
//! the manifest. Individual `.wav` files live next to it and are
//! written directly by the Python worker.

use std::io;
use std::path::{Path, PathBuf};

use super::models::SyncManifest;

pub const SYNCED_SUBDIR: &str = "voices/synced";
pub const SYNCED_MANIFEST_FILENAME: &str = "sync.json";
pub const SYNCED_MANIFEST_RELATIVE: &str = "voices/synced/sync.json";

#[derive(Debug, Clone, Copy)]
pub struct SyncCacheFile;

impl SyncCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<SyncManifest>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: SyncManifest = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid sync.json: {e}")))?;
                Ok(Some(doc))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &SyncManifest) -> io::Result<PathBuf> {
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
    project_root
        .join(SYNCED_SUBDIR)
        .join(SYNCED_MANIFEST_FILENAME)
}

pub fn synced_dir(project_root: &Path) -> PathBuf {
    project_root.join(SYNCED_SUBDIR)
}
