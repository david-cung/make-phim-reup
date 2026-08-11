//! Per-project audio cache manifest.
//!
//! Path: `<project_root>/cache/audio_cache.json`.
//!
//! Structure is versioned so a Phase-3+ change (e.g. adding cache
//! entries for other Whisper models) can extend without breaking old
//! files. Missing fields deserialise as `None`.

use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ffmpeg::extract::AudioExtractParams;
use crate::media::SourceFingerprint;

pub const AUDIO_CACHE_FILENAME: &str = "audio_cache.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioCacheEntry {
    pub source: SourceFingerprint,
    pub source_path: String,
    pub params: AudioExtractParams,
    /// Path *relative* to the project root (portable across renames).
    pub output_relative: String,
    pub output_size_bytes: u64,
    pub duration_secs: f64,
    pub created_at: DateTime<Utc>,
}

impl AudioCacheEntry {
    /// True iff the same source + same params would produce this file.
    pub fn matches(&self, source: &SourceFingerprint, params: &AudioExtractParams) -> bool {
        self.source.matches_content(source) && &self.params == params
    }

    pub fn absolute_output(&self, project_root: &Path) -> PathBuf {
        project_root.join(&self.output_relative)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioCacheFile {
    /// Version of the manifest schema (bump on breaking changes).
    #[serde(default = "default_version")]
    pub version: u32,
    /// The extraction destined for STT (Phase 3). Named so future
    /// extractions for other stages fit alongside it.
    #[serde(default)]
    pub original_wav: Option<AudioCacheEntry>,
}

fn default_version() -> u32 {
    1
}

impl AudioCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Self> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<AudioCacheFile>(&text)
                .map_err(|e| io::Error::other(format!("invalid audio_cache.json: {e}"))),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err),
        }
    }

    pub fn save(&self, project_root: &Path) -> io::Result<()> {
        let path = manifest_path(project_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let text =
            serde_json::to_string_pretty(self).map_err(|e| io::Error::other(e.to_string()))?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    /// Returns the cached original WAV entry iff:
    ///   * fingerprint + params match
    ///   * the file it points to still exists and its size matches
    pub fn hit(
        &self,
        project_root: &Path,
        source: &SourceFingerprint,
        params: &AudioExtractParams,
    ) -> Option<AudioCacheEntry> {
        let entry = self.original_wav.as_ref()?;
        if !entry.matches(source, params) {
            return None;
        }
        let abs = entry.absolute_output(project_root);
        let meta = std::fs::metadata(&abs).ok()?;
        if meta.len() != entry.output_size_bytes {
            return None;
        }
        Some(entry.clone())
    }
}

fn manifest_path(project_root: &Path) -> PathBuf {
    project_root.join("cache").join(AUDIO_CACHE_FILENAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_project() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "lmt-audio-cache-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(root.join("audio")).unwrap();
        std::fs::create_dir_all(root.join("cache")).unwrap();
        root
    }

    fn sample_entry(hash: &str, wav_size: u64) -> AudioCacheEntry {
        AudioCacheEntry {
            source: SourceFingerprint {
                hash: hash.into(),
                size_bytes: 12345,
                modified_at: Utc::now(),
            },
            source_path: "/tmp/movie.mkv".into(),
            params: AudioExtractParams::whisper_default(),
            output_relative: "audio/original.wav".into(),
            output_size_bytes: wav_size,
            duration_secs: 3.0,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn round_trip() {
        let root = tmp_project();
        let entry = sample_entry("sha256:aa", 42);
        let file = AudioCacheFile {
            version: 1,
            original_wav: Some(entry.clone()),
        };
        file.save(&root).unwrap();
        let loaded = AudioCacheFile::load(&root).unwrap();
        assert_eq!(
            loaded.original_wav.as_ref().unwrap().source.hash,
            "sha256:aa"
        );
    }

    #[test]
    fn missing_manifest_returns_default() {
        let root = tmp_project();
        let file = AudioCacheFile::load(&root).unwrap();
        assert_eq!(file, AudioCacheFile::default());
    }

    #[test]
    fn hit_requires_matching_source_and_file() {
        let root = tmp_project();
        // Create the actual WAV file so `hit` can stat it.
        let wav = root.join("audio/original.wav");
        let mut f = std::fs::File::create(&wav).unwrap();
        f.write_all(&[0u8; 42]).unwrap();

        let source_fp = SourceFingerprint {
            hash: "sha256:aa".into(),
            size_bytes: 12345,
            modified_at: Utc::now(),
        };
        let file = AudioCacheFile {
            version: 1,
            original_wav: Some(sample_entry("sha256:aa", 42)),
        };
        assert!(file
            .hit(&root, &source_fp, &AudioExtractParams::whisper_default())
            .is_some());
    }

    #[test]
    fn hit_misses_on_deleted_file() {
        let root = tmp_project();
        // No file on disk.
        let source_fp = SourceFingerprint {
            hash: "sha256:aa".into(),
            size_bytes: 12345,
            modified_at: Utc::now(),
        };
        let file = AudioCacheFile {
            version: 1,
            original_wav: Some(sample_entry("sha256:aa", 42)),
        };
        assert!(file
            .hit(&root, &source_fp, &AudioExtractParams::whisper_default())
            .is_none());
    }

    #[test]
    fn hit_misses_on_different_hash() {
        let root = tmp_project();
        std::fs::File::create(root.join("audio/original.wav")).unwrap();
        let other = SourceFingerprint {
            hash: "sha256:bb".into(),
            size_bytes: 12345,
            modified_at: Utc::now(),
        };
        let file = AudioCacheFile {
            version: 1,
            original_wav: Some(sample_entry("sha256:aa", 0)),
        };
        assert!(file
            .hit(&root, &other, &AudioExtractParams::whisper_default())
            .is_none());
    }
}
