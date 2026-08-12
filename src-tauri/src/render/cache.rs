//! Persistence for `<project>/output/render.json`.
//!
//! Atomic write (tmp + rename) so a crash mid-write can never corrupt
//! the manifest. The rendered movie lives next to it, either at
//! `<project>/output/movie_vi.mp4` (default) or at whatever custom
//! absolute path the user picked in the settings.

use std::io;
use std::path::{Path, PathBuf};

use super::models::{OutputFormat, RenderManifest, SubtitleMode};

/// Directory under the project root that holds the render + its
/// manifest.
pub const RENDER_SUBDIR: &str = "output";
pub const RENDER_MANIFEST_FILENAME: &str = "render.json";
pub const RENDER_MANIFEST_RELATIVE: &str = "output/render.json";
/// Base filename (without extension) for the default render output.
pub const RENDER_OUTPUT_BASENAME: &str = "movie_vi";
/// Base filename for the external SRT sidecar (when subtitle mode =
/// external).
pub const RENDER_SUBTITLE_BASENAME: &str = "movie_vi";

#[derive(Debug, Clone, Copy)]
pub struct RenderCacheFile;

impl RenderCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<RenderManifest>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: RenderManifest = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid render.json: {e}")))?;
                Ok(Some(doc.migrated()))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &RenderManifest) -> io::Result<PathBuf> {
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
        .join(RENDER_SUBDIR)
        .join(RENDER_MANIFEST_FILENAME)
}

/// Default output path for the rendered movie — always inside the
/// project's `output/` subdirectory.
pub fn default_render_output_path(project_root: &Path, format: OutputFormat) -> PathBuf {
    project_root
        .join(RENDER_SUBDIR)
        .join(format!("{RENDER_OUTPUT_BASENAME}.{}", format.extension()))
}

/// Default sidecar SRT path — always inside the project's `output/`
/// subdirectory.
pub fn default_subtitle_output_path(project_root: &Path) -> PathBuf {
    project_root
        .join(RENDER_SUBDIR)
        .join(format!("{RENDER_SUBTITLE_BASENAME}.srt"))
}

/// Where to write the SRT for a given render.
///
/// In `External` mode the SRT is part of what the user gets, so it has
/// to sit beside the movie and share its stem — players only auto-load a
/// sidecar they find next to the video file, and a custom output path
/// (e.g. `~/Downloads`) is nowhere near the project folder. `Burned`
/// only feeds it to FFmpeg's `subtitles=` filter, so it can stay inside
/// the project.
pub fn subtitle_sidecar_path(
    project_root: &Path,
    output_path: &Path,
    mode: SubtitleMode,
) -> PathBuf {
    match mode {
        SubtitleMode::External => output_path.with_extension("srt"),
        _ => default_subtitle_output_path(project_root),
    }
}
