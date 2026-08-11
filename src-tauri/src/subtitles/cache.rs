//! Persistence for `<project>/subtitles/subtitles.json`.
//!
//! Atomic write (tmp + rename) so a crash mid-write can never corrupt
//! the deliverable. Subtitles ARE the deliverable from Phase 5
//! onwards — they must survive anything.

use std::io;
use std::path::{Path, PathBuf};

use super::models::SubtitleDoc;

pub const SUBTITLES_FILENAME: &str = "subtitles.json";
pub const SUBTITLES_RELATIVE: &str = "subtitles/subtitles.json";

#[derive(Debug, Clone, Copy)]
pub struct SubtitleCacheFile;

impl SubtitleCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<SubtitleDoc>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: SubtitleDoc = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid subtitles.json: {e}")))?;
                Ok(Some(doc))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &SubtitleDoc) -> io::Result<PathBuf> {
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
    project_root.join("subtitles").join(SUBTITLES_FILENAME)
}
