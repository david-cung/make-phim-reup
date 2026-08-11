//! Persistence for ``<project>/translation/translation.json``.
//!
//! The file is BOTH the cache manifest AND the deliverable — we store
//! every translated segment alongside the metadata that lets us decide
//! whether to reuse or regenerate.
//!
//! IO is atomic (write-to-tmp + rename) so a crash mid-write leaves
//! the previous good copy intact — important because the service
//! persists after every completed chunk.

use std::io;
use std::path::{Path, PathBuf};

use chrono::Utc;

use super::models::{TranslatedSegment, TranslationDoc};

pub const TRANSLATION_FILENAME: &str = "translation.json";
pub const TRANSLATION_RELATIVE: &str = "translation/translation.json";

#[derive(Debug, Clone)]
pub struct TranslationCacheFile;

impl TranslationCacheFile {
    pub fn load(project_root: &Path) -> io::Result<Option<TranslationDoc>> {
        let path = manifest_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let doc: TranslationDoc = serde_json::from_str(&text)
                    .map_err(|e| io::Error::other(format!("invalid translation.json: {e}")))?;
                Ok(Some(doc))
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn save(project_root: &Path, doc: &TranslationDoc) -> io::Result<PathBuf> {
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

    /// A cache hit means the file is completely translated for this
    /// exact input: both cache_key AND every segment has a
    /// translation. Anything short of that is a partial that the
    /// service can resume — not a hit.
    pub fn is_complete_hit(
        doc: &TranslationDoc,
        expected_cache_key: &str,
        expected_transcript_key: &str,
    ) -> bool {
        doc.cache_key == expected_cache_key
            && doc.transcript_cache_key == expected_transcript_key
            && doc
                .segments
                .iter()
                .all(|s| !s.translation.trim().is_empty())
    }

    /// A cache is *resumable* when the identity matches but some
    /// segments are still empty. The service will only ask the LLM to
    /// fill in the missing ones.
    pub fn is_resumable(
        doc: &TranslationDoc,
        expected_cache_key: &str,
        expected_transcript_key: &str,
    ) -> bool {
        doc.cache_key == expected_cache_key && doc.transcript_cache_key == expected_transcript_key
    }
}

pub fn manifest_path(project_root: &Path) -> PathBuf {
    project_root.join("translation").join(TRANSLATION_FILENAME)
}

/// Every ingredient needed to seed a fresh :struct:`TranslationDoc`.
/// Grouped so callers don't have to remember a huge positional list.
pub struct EmptyDocParams {
    pub source_language: String,
    pub target_language: String,
    pub model: String,
    pub prompt_version: String,
    pub cache_key: String,
    pub transcript_cache_key: String,
    pub audio_hash: String,
    pub options_value: serde_json::Value,
}

/// Build a fresh doc with empty translations for every transcript
/// segment. Used the first time a project is translated (or after
/// options changed and the cache_key doesn't match anymore).
pub fn empty_doc_from(
    segments: &[(u32, String, f64, f64)],
    params: EmptyDocParams,
) -> TranslationDoc {
    let now = Utc::now();
    let segs = segments
        .iter()
        .map(|(id, text, start, end)| TranslatedSegment {
            id: *id,
            source_text: text.clone(),
            translation: String::new(),
            start: *start,
            end: *end,
            edited: false,
        })
        .collect();
    TranslationDoc {
        version: 1,
        source_language: params.source_language,
        target_language: params.target_language,
        segments: segs,
        model: params.model,
        prompt_version: params.prompt_version,
        cache_key: params.cache_key,
        transcript_cache_key: params.transcript_cache_key,
        audio_hash: params.audio_hash,
        created_at: now,
        updated_at: now,
        provider: "llama.cpp".into(),
        options: params.options_value,
    }
}

/// Apply a chunk-completed batch of translations to an in-memory doc.
/// Returns the number of segments actually updated (ids not in the
/// doc are silently ignored).
pub fn apply_chunk(doc: &mut TranslationDoc, updates: &[(u32, String)]) -> u32 {
    let mut hits = 0u32;
    for (id, text) in updates {
        if let Some(seg) = doc.segments.iter_mut().find(|s| s.id == *id) {
            seg.translation = text.clone();
            hits += 1;
        }
    }
    if hits > 0 {
        doc.updated_at = Utc::now();
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn tmp_project() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "lmt-tr-cache-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(root.join("translation")).unwrap();
        root
    }

    fn sample(cache_key: &str, transcript_key: &str, filled: bool) -> TranslationDoc {
        let now = Utc::now();
        TranslationDoc {
            version: 1,
            source_language: "en".into(),
            target_language: "vi".into(),
            segments: vec![
                TranslatedSegment {
                    id: 0,
                    source_text: "Hello".into(),
                    translation: if filled {
                        "Xin chào".into()
                    } else {
                        "".into()
                    },
                    start: 0.0,
                    end: 1.0,
                    edited: false,
                },
                TranslatedSegment {
                    id: 1,
                    source_text: "World".into(),
                    translation: if filled {
                        "thế giới".into()
                    } else {
                        "".into()
                    },
                    start: 1.0,
                    end: 2.0,
                    edited: false,
                },
            ],
            model: "qwen2.gguf".into(),
            prompt_version: "translation_prompt_v1".into(),
            cache_key: cache_key.into(),
            transcript_cache_key: transcript_key.into(),
            audio_hash: "sha256:aa".into(),
            created_at: now,
            updated_at: now,
            provider: "llama.cpp".into(),
            options: serde_json::json!({}),
        }
    }

    #[test]
    fn round_trip_preserves_fields() {
        let root = tmp_project();
        let doc = sample("k", "t", true);
        TranslationCacheFile::save(&root, &doc).unwrap();
        let back = TranslationCacheFile::load(&root).unwrap().unwrap();
        assert_eq!(back.segments.len(), 2);
        assert_eq!(back.cache_key, "k");
        assert_eq!(back.transcript_cache_key, "t");
        assert_eq!(back.segments[0].translation, "Xin chào");
    }

    #[test]
    fn load_missing_returns_none() {
        let root = tmp_project();
        assert!(TranslationCacheFile::load(&root).unwrap().is_none());
    }

    #[test]
    fn full_hit_requires_all_segments_translated() {
        let full = sample("k", "t", true);
        assert!(TranslationCacheFile::is_complete_hit(&full, "k", "t"));
        let partial = sample("k", "t", false);
        assert!(!TranslationCacheFile::is_complete_hit(&partial, "k", "t"));
        assert!(TranslationCacheFile::is_resumable(&partial, "k", "t"));
    }

    #[test]
    fn hit_rejects_key_mismatch() {
        let full = sample("k", "t", true);
        assert!(!TranslationCacheFile::is_complete_hit(&full, "k2", "t"));
        assert!(!TranslationCacheFile::is_resumable(&full, "k2", "t"));
    }

    #[test]
    fn apply_chunk_updates_only_matching_ids() {
        let mut doc = sample("k", "t", false);
        let hits = apply_chunk(&mut doc, &[(0, "Xin chào".into()), (5, "ignored".into())]);
        assert_eq!(hits, 1);
        assert_eq!(doc.segments[0].translation, "Xin chào");
        assert_eq!(doc.segments[1].translation, "");
    }
}
