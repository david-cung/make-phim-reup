//! Persistence for ``<project>/transcription/transcription.json``.
//!
//! The file is BOTH the cache manifest AND the deliverable — we store
//! the transcript segments alongside the metadata (model, device,
//! options, source hash) that let us skip re-transcription when
//! nothing relevant has changed.

use std::io;
use std::path::{Path, PathBuf};

use super::models::Transcript;

pub const TRANSCRIPT_FILENAME: &str = "transcription.json";
pub const TRANSCRIPT_RELATIVE: &str = "transcription/transcription.json";

/// Wrapper around the on-disk transcript file. All IO is atomic
/// (write-to-tmp + rename) so a crash mid-write leaves the previous
/// good copy intact.
#[derive(Debug, Clone)]
pub struct TranscriptCacheFile {
    pub transcript: Transcript,
}

impl TranscriptCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<Transcript>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let t: Transcript = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid transcription.json: {e}")))?;
                Ok(Some(t))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, transcript: &Transcript) -> io::Result<PathBuf> {
        let path = manifest_path(project_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(transcript)
            .map_err(|e| io::Error::other(e.to_string()))?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// Cache hit iff both the cache key AND the audio hash match. The
    /// cache key alone would suffice mathematically, but we double-check
    /// so a stray manual edit of the file can't produce a false hit.
    pub fn hit(
        transcript: &Transcript,
        expected_cache_key: &str,
        expected_audio_hash: &str,
    ) -> bool {
        transcript.cache_key == expected_cache_key && transcript.audio.hash == expected_audio_hash
    }
}

pub fn manifest_path(project_root: &Path) -> PathBuf {
    project_root.join("transcription").join(TRANSCRIPT_FILENAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::models::{TranscribeSegment, Transcript, TranscriptAudio, TranscriptSummary};
    use chrono::Utc;
    use serde_json::json;

    fn tmp_project() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "lmt-stt-cache-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(root.join("transcription")).unwrap();
        root
    }

    fn sample(cache_key: &str, audio_hash: &str) -> Transcript {
        Transcript {
            version: 1,
            language: "en".into(),
            segments: vec![TranscribeSegment {
                id: 0,
                start: 0.0,
                end: 1.0,
                text: "hi".into(),
                speaker_id: None,
                speaker_confidence: None,
                raw_text: None,
                normalized_text: None,
                source_segment_id: None,
                source_sub_segment_id: None,
                source_quality: None,
                semantic_facts: None,
                avg_logprob: None,
                no_speech_prob: None,
                words: None,
            }],
            model: "small".into(),
            device: "cpu".into(),
            compute_type: "int8".into(),
            word_timestamps: false,
            audio: TranscriptAudio {
                path: "audio/original.wav".into(),
                hash: audio_hash.into(),
            },
            duration_secs: 1.0,
            cache_key: cache_key.into(),
            created_at: Utc::now(),
            provider: "faster-whisper".into(),
            options: json!({}),
            speaker_memory: json!({}),
        }
    }

    #[test]
    fn round_trip_preserves_fields() {
        let root = tmp_project();
        let t = sample("sha256:aa", "sha256:bb");
        TranscriptCacheFile::save(&root, &t).unwrap();
        let back = TranscriptCacheFile::load(&root).unwrap().unwrap();
        assert_eq!(back.language, "en");
        assert_eq!(back.segments.len(), 1);
        assert_eq!(back.cache_key, "sha256:aa");
        assert_eq!(back.audio.hash, "sha256:bb");
    }

    #[test]
    fn load_missing_returns_none() {
        let root = tmp_project();
        assert!(TranscriptCacheFile::load(&root).unwrap().is_none());
    }

    #[test]
    fn hit_matches_only_when_both_match() {
        let t = sample("sha256:aa", "sha256:bb");
        assert!(TranscriptCacheFile::hit(&t, "sha256:aa", "sha256:bb"));
        assert!(!TranscriptCacheFile::hit(&t, "sha256:aa", "sha256:cc"));
        assert!(!TranscriptCacheFile::hit(&t, "sha256:xx", "sha256:bb"));
    }

    #[test]
    fn summary_derives_segment_count() {
        let t = sample("k", "h");
        let s = TranscriptSummary::from_transcript(&t, "transcription/transcription.json");
        assert_eq!(s.segment_count, 1);
        assert_eq!(s.language, "en");
        assert_eq!(s.relative_path, "transcription/transcription.json");
    }
}
