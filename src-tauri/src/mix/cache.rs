//! Persistence for `<project>/audio/mix.json`.
//!
//! Atomic write (tmp + rename) so a crash mid-write can never corrupt
//! the manifest. The mixed WAV lives at
//! `<project>/audio/mixed_vi.wav` right next to it.

use std::io;
use std::path::{Path, PathBuf};

use super::models::MixManifest;

/// Directory under the project root that holds the mix + its manifest.
pub const MIX_SUBDIR: &str = "audio";
pub const MIX_MANIFEST_FILENAME: &str = "mix.json";
pub const MIX_MANIFEST_RELATIVE: &str = "audio/mix.json";
/// Relative path of the produced mix WAV. Kept as a constant so both
/// the FFmpeg command and the frontend agree on the location.
pub const MIX_OUTPUT_RELATIVE: &str = "audio/mixed_vi.wav";

#[derive(Debug, Clone, Copy)]
pub struct MixCacheFile;

impl MixCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<MixManifest>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: MixManifest = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid mix.json: {e}")))?;
                Ok(Some(doc))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &MixManifest) -> io::Result<PathBuf> {
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
    project_root.join(MIX_SUBDIR).join(MIX_MANIFEST_FILENAME)
}

pub fn mix_output_path(project_root: &Path) -> PathBuf {
    project_root.join(MIX_OUTPUT_RELATIVE)
}
