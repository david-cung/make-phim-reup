//! Small tests for the pure functions (derivation, doc math). The
//! service itself is covered indirectly through IPC in dev/integration
//! runs; unit-testing it would require a full project fixture and
//! that isn't warranted for Phase 5.

use chrono::Utc;

use crate::stt::{TranscribeSegment, Transcript, TranscriptAudio};
use crate::translation::{TranslatedSegment, TranslationDoc};

use super::derive::{derive_from_sources, merge_preserving_edits};
use super::models::{DirtyFlags, SubtitleDoc};

fn transcript(segments: Vec<(u32, f64, f64, &str)>) -> Transcript {
    Transcript {
        version: 1,
        language: "en".into(),
        segments: segments
            .into_iter()
            .map(|(id, start, end, text)| TranscribeSegment {
                id,
                start,
                end,
                text: text.into(),
                avg_logprob: None,
                no_speech_prob: None,
                words: None,
            })
            .collect(),
        model: "small".into(),
        device: "cpu".into(),
        compute_type: "int8".into(),
        word_timestamps: false,
        audio: TranscriptAudio {
            path: "audio/original.wav".into(),
            hash: "sha256:aa".into(),
        },
        duration_secs: 10.0,
        cache_key: "sha256:tk".into(),
        created_at: Utc::now(),
        provider: "faster-whisper".into(),
        options: serde_json::json!({}),
    }
}

fn translation(pairs: Vec<(u32, &str)>) -> TranslationDoc {
    let now = Utc::now();
    TranslationDoc {
        version: 1,
        source_language: "en".into(),
        target_language: "vi".into(),
        segments: pairs
            .into_iter()
            .map(|(id, text)| TranslatedSegment {
                id,
                source_text: "".into(),
                translation: text.into(),
                dubbing: text.into(),
                start: 0.0,
                end: 0.0,
                edited: false,
            })
            .collect(),
        model: "qwen2.gguf".into(),
        prompt_version: "translation_prompt_v1".into(),
        cache_key: "sha256:tr".into(),
        transcript_cache_key: "sha256:tk".into(),
        audio_hash: "sha256:aa".into(),
        created_at: now,
        updated_at: now,
        provider: "llama.cpp".into(),
        options: serde_json::json!({}),
    }
}

#[test]
fn derive_combines_transcript_and_translation() {
    let t = transcript(vec![(0, 0.0, 1.0, "Hello"), (1, 1.0, 2.0, "World")]);
    let tr = translation(vec![(0, "Xin chào")]);
    let doc = derive_from_sources(&t, Some(&tr), "en", "vi");
    assert_eq!(doc.segments.len(), 2);
    assert_eq!(doc.segments[0].source_text, "Hello");
    assert_eq!(doc.segments[0].translated_text, "Xin chào");
    assert_eq!(doc.segments[1].translated_text, ""); // no translation for id 1
    assert_eq!(doc.next_id, 2);
    assert_eq!(
        doc.derived_from.transcript_cache_key.as_deref(),
        Some("sha256:tk")
    );
}

#[test]
fn merge_preserves_user_edits() {
    let t = transcript(vec![(0, 0.0, 1.0, "Hello"), (1, 1.0, 2.0, "World")]);
    let tr = translation(vec![(0, "Xin chào"), (1, "Thế giới")]);
    let mut prev = derive_from_sources(&t, Some(&tr), "en", "vi");
    // User edits — should survive a rebuild.
    prev.segments[0].translated_text = "Chào bạn".into();
    prev.segments[0].speaker = Some("Alice".into());
    prev.segments[1].voice_id = Some("piper-vi-1".into());

    // Fresh derivation from the same sources.
    let fresh = derive_from_sources(&t, Some(&tr), "en", "vi");
    let merged = merge_preserving_edits(fresh, Some(&prev));

    assert_eq!(merged.segments[0].translated_text, "Chào bạn");
    assert_eq!(merged.segments[0].speaker.as_deref(), Some("Alice"));
    assert_eq!(merged.segments[1].voice_id.as_deref(), Some("piper-vi-1"));
}

#[test]
fn allocate_id_never_collides() {
    let mut doc = SubtitleDoc::empty("en".into(), "vi".into());
    // Simulate a doc that has already handed out ids up to 9.
    doc.next_id = 10;
    let a = doc.allocate_id();
    let b = doc.allocate_id();
    assert_eq!(a, 10);
    assert_eq!(b, 11);
    assert_eq!(doc.next_id, 12);
}

#[test]
fn dirty_flags_mark_and_check() {
    let mut d = DirtyFlags::default();
    assert!(!d.any());
    d.mark_downstream();
    assert!(d.tts && d.mix && d.render);
    assert!(d.any());
}

#[test]
fn overlap_detection_reports_adjacent_pairs() {
    let t = transcript(vec![(0, 0.0, 2.0, "a"), (1, 1.0, 3.0, "b")]);
    let doc = derive_from_sources(&t, None, "en", "vi");
    let overlaps = doc.overlap_pairs();
    assert_eq!(overlaps, vec![(0, 1)]);
}
