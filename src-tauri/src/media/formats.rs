//! Supported container formats for the MVP.
//!
//! Whitelist is intentionally narrow. Adding formats later is a one-line
//! change; opening the door too wide risks surfacing containers where our
//! extraction pipeline (Phase 2 / 3) has never been exercised.

use std::path::Path;

use serde::Serialize;

/// The extensions the app is willing to import.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["mp4", "m4v", "mkv", "mov", "avi", "webm"];

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SupportedContainer {
    Mp4,
    Mkv,
    Mov,
    Avi,
    Webm,
}

impl SupportedContainer {
    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "mp4" | "m4v" => Self::Mp4,
            "mkv" => Self::Mkv,
            "mov" => Self::Mov,
            "avi" => Self::Avi,
            "webm" => Self::Webm,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Mov => "mov",
            Self::Avi => "avi",
            Self::Webm => "webm",
        }
    }
}

pub fn is_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .map(|s| SUPPORTED_EXTENSIONS.contains(&s.as_str()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn accepts_common_extensions() {
        for name in ["a.mp4", "b.MKV", "c.mov", "d.Avi", "e.webm", "f.m4v"] {
            assert!(
                is_supported_extension(Path::new(name)),
                "expected accept: {name}"
            );
        }
    }

    #[test]
    fn rejects_unsupported() {
        for name in ["a.mp3", "b.txt", "c", "d.flv", "e.wmv", "f.ogv"] {
            assert!(
                !is_supported_extension(Path::new(name)),
                "expected reject: {name}"
            );
        }
    }

    #[test]
    fn container_from_extension() {
        assert_eq!(
            SupportedContainer::from_extension("mp4"),
            Some(SupportedContainer::Mp4)
        );
        assert_eq!(
            SupportedContainer::from_extension("M4V"),
            Some(SupportedContainer::Mp4)
        );
        assert_eq!(
            SupportedContainer::from_extension("mov"),
            Some(SupportedContainer::Mov)
        );
        assert_eq!(SupportedContainer::from_extension("nope"), None);
    }
}
