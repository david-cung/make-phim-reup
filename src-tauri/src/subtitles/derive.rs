//! Build the canonical subtitle model from the Phase 3 transcript and
//! (optionally) the Phase 4 translation.
//!
//! Timing is copied from the transcript — Whisper segments are the
//! authoritative timing source (see ARCHITECTURE §4.4). Translated
//! text is copied from the translation doc when a matching id
//! exists; missing translations produce a segment with an empty
//! `translatedText`.

use chrono::Utc;

use crate::stt::Transcript;
use crate::translation::TranslationDoc;

use super::models::{
    DerivedFrom, DirtyFlags, SubtitleDoc, SubtitleSegment, SubtitleWord, SUBTITLE_SCHEMA_VERSION,
};

/// Build a fresh `SubtitleDoc` from the transcript (required) and
/// the optional translation doc.
pub fn derive_from_sources(
    transcript: &Transcript,
    translation: Option<&TranslationDoc>,
    source_language: &str,
    target_language: &str,
) -> SubtitleDoc {
    let now = Utc::now();
    let mut segments: Vec<SubtitleSegment> = transcript
        .segments
        .iter()
        .map(|t| SubtitleSegment {
            id: t.id,
            start: t.start,
            end: t.end,
            source_text: t.text.clone(),
            translated_text: translation
                .and_then(|td| td.segments.iter().find(|s| s.id == t.id))
                .map(|s| s.translation.clone())
                .unwrap_or_default(),
            dubbing_text: translation
                .and_then(|td| td.segments.iter().find(|s| s.id == t.id))
                .map(|s| {
                    let d = s.dubbing.trim();
                    if d.is_empty() {
                        s.translation.clone()
                    } else {
                        s.dubbing.clone()
                    }
                })
                .unwrap_or_default(),
            words: t.words.as_ref().map(|words| {
                words
                    .iter()
                    .map(|w| SubtitleWord {
                        text: w.word.clone(),
                        start: w.start,
                        end: w.end,
                    })
                    .collect()
            }),
            speaker: t
                .speaker_id
                .as_ref()
                .filter(|speaker| !speaker.trim().is_empty())
                .cloned(),
            speaker_confidence: t.speaker_confidence,
            voice_id: None,
        })
        .collect();
    segments.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let next_id = segments
        .iter()
        .map(|s| s.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    SubtitleDoc {
        version: SUBTITLE_SCHEMA_VERSION,
        source_language: source_language.to_string(),
        target_language: target_language.to_string(),
        segments,
        derived_from: DerivedFrom {
            transcript_cache_key: Some(transcript.cache_key.clone()),
            translation_cache_key: translation.map(|t| t.cache_key.clone()),
            origin: "transcript+translation".into(),
        },
        dirty: DirtyFlags::default(),
        next_id,
        created_at: now,
        updated_at: now,
    }
}

/// Preserve manual edits from a prior doc onto a fresh derivation.
///
/// Rules:
/// * Segments whose `id` matches AND whose `sourceText` (from the
///   transcript) is unchanged keep the user's manual `translated_text`
///   / `speaker` / `voice_id` fields — never overwrite human work.
/// * Timing always comes from the new transcript.
/// * New ids in the transcript get fresh empty rows.
/// * Ids that vanished from the transcript are dropped.
///
/// The returned doc inherits the dirty flags from `previous` (a
/// rebuild neither creates nor clears downstream invalidation on its
/// own — the fields that changed did that).
pub fn merge_preserving_edits(fresh: SubtitleDoc, previous: Option<&SubtitleDoc>) -> SubtitleDoc {
    let Some(prev) = previous else {
        return fresh;
    };
    let mut merged = fresh;
    for seg in merged.segments.iter_mut() {
        let Some(prev_seg) = prev.segments.iter().find(|s| s.id == seg.id) else {
            continue;
        };
        // Preserve user's speaker / voice / manual translation edits
        // whenever they carry information.
        if prev_seg.speaker.is_some() {
            seg.speaker = prev_seg.speaker.clone();
        }
        if prev_seg.speaker_confidence.is_some() {
            seg.speaker_confidence = prev_seg.speaker_confidence;
        }
        if prev_seg.voice_id.is_some() {
            seg.voice_id = prev_seg.voice_id.clone();
        }
        // If the previous translated text is non-empty AND differs
        // from what the fresh derivation produced, treat it as a
        // manual edit worth preserving.
        if !prev_seg.translated_text.trim().is_empty()
            && prev_seg.translated_text != seg.translated_text
        {
            seg.translated_text = prev_seg.translated_text.clone();
        }
        if !prev_seg.dubbing_text.trim().is_empty()
            && prev_seg.dubbing_text != seg.dubbing_text
        {
            seg.dubbing_text = prev_seg.dubbing_text.clone();
        }
        if prev_seg.words.is_some() && seg.words.is_none() {
            seg.words = prev_seg.words.clone();
        }
    }
    // A rebuild is a downstream-invalidating action if anything moved.
    if merged.segments != prev.segments {
        merged.dirty.mark_downstream();
    } else {
        merged.dirty = prev.dirty;
    }
    merged.created_at = prev.created_at;
    // `next_id` must not regress below anything the previous doc
    // already handed out, or newly-allocated ids could collide with
    // ids the user has already edited in prior sessions.
    merged.next_id = merged.next_id.max(prev.next_id).max(
        prev.segments
            .iter()
            .map(|s| s.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
    );
    merged
}
