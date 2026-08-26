//! Wire types for the subtitle subsystem.
//!
//! The `SubtitleSegment` here is *the* canonical subtitle model —
//! phases beyond 5 (TTS, mix, render) consume this exact shape. It
//! deliberately does NOT re-invent whisper word timestamps or
//! translation prompt versions; those live in their own manifests and
//! are referenced through `derived_from`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const SUBTITLE_SCHEMA_VERSION: u32 = 1;

/// The single canonical subtitle segment.
///
/// `id` is stable across edits: split/merge/add/delete never renumber
/// existing rows so downstream stages can key TTS/mix results by id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleSegment {
    pub id: u32,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub source_text: String,
    #[serde(default)]
    pub translated_text: String,
    /// Spoken line used for TTS. May be slightly shorter than the
    /// on-screen subtitle. Empty means "use translated_text".
    #[serde(default)]
    pub dubbing_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<SubtitleWord>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl SubtitleSegment {
    /// Guardrail used by every mutation path. Returns an owned error
    /// message on failure so callers can wrap it in the appropriate
    /// `SubtitleError` variant.
    pub fn validate_timing(&self) -> Result<(), String> {
        if !self.start.is_finite() || !self.end.is_finite() {
            return Err("start/end must be finite".into());
        }
        if self.start < 0.0 {
            return Err("start must be >= 0".into());
        }
        if self.end <= self.start {
            return Err(format!(
                "end ({}) must be strictly greater than start ({})",
                self.end, self.start
            ));
        }
        Ok(())
    }
}

/// Where this subtitle document was originally derived from. Used to
/// decide "your transcript changed — rebuild subtitles?" prompts on
/// the UI side.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DerivedFrom {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation_cache_key: Option<String>,
    /// `"transcript+translation"`, `"srt-import"`, `"ass-import"`,
    /// or `"manual"` — a coarse tag for the UI.
    #[serde(default = "default_origin")]
    pub origin: String,
}

fn default_origin() -> String {
    "transcript+translation".into()
}

/// Downstream stages that need to re-run because a subtitle changed.
///
/// Granularity, per the phase spec:
///
/// * `translatedText` / `speaker` / `voiceId` change → **TTS is stale**
///   → therefore sync / mix / render are stale too (they consume the
///   TTS WAVs).
/// * `start` / `end` change → TTS output itself is still valid (the
///   audio for the line hasn't changed), but **sync / mix / render**
///   are stale because the target window moved.
/// * `sourceText` alone changes nothing downstream (nothing reads it).
///
/// The struct doubles as a "clear these bits" mask consumed by
/// [`SubtitleService::clear_dirty_flags`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DirtyFlags {
    #[serde(default)]
    pub tts: bool,
    /// Introduced in Phase 7 — the per-segment timing-adjusted WAV
    /// in `voices/synced/` needs to be regenerated.
    #[serde(default)]
    pub sync: bool,
    #[serde(default)]
    pub mix: bool,
    #[serde(default)]
    pub render: bool,
}

impl DirtyFlags {
    /// Content change (text / speaker / voice) → invalidate TTS and
    /// everything downstream of it.
    pub fn mark_content_dirty(&mut self) {
        self.tts = true;
        self.sync = true;
        self.mix = true;
        self.render = true;
    }

    /// Timing-only change → TTS audio is still valid, but the sync
    /// window moved so sync/mix/render must re-run.
    pub fn mark_timing_dirty(&mut self) {
        self.sync = true;
        self.mix = true;
        self.render = true;
    }

    /// Backwards-compatible shorthand used by structural edits (add /
    /// delete / split / merge / import) where both content and timing
    /// may have changed at once.
    pub fn mark_downstream(&mut self) {
        self.mark_content_dirty();
    }

    pub fn any(&self) -> bool {
        self.tts || self.sync || self.mix || self.render
    }

    /// Bitwise-AND-NOT semantics used by `clear_dirty_flags`: for
    /// every field where `mask` is true, clear our matching field.
    pub fn clear_where(&mut self, mask: DirtyFlags) {
        if mask.tts {
            self.tts = false;
        }
        if mask.sync {
            self.sync = false;
        }
        if mask.mix {
            self.mix = false;
        }
        if mask.render {
            self.render = false;
        }
    }

    /// Convenience: a mask with a single bit set.
    pub const fn only_tts() -> Self {
        Self {
            tts: true,
            sync: false,
            mix: false,
            render: false,
        }
    }
    pub const fn only_sync() -> Self {
        Self {
            tts: false,
            sync: true,
            mix: false,
            render: false,
        }
    }
    /// Introduced in Phase 8 — mask used by `MixService::maybe_clear_dirty`
    /// after a successful mix pass so timing / TTS / sync bits are
    /// preserved untouched.
    pub const fn only_mix() -> Self {
        Self {
            tts: false,
            sync: false,
            mix: true,
            render: false,
        }
    }
    /// Introduced in Phase 9 — mask used by `RenderService::maybe_clear_dirty`
    /// after a successful render pass so timing / TTS / sync / mix bits
    /// are preserved untouched.
    pub const fn only_render() -> Self {
        Self {
            tts: false,
            sync: false,
            mix: false,
            render: true,
        }
    }
    pub const fn all() -> Self {
        Self {
            tts: true,
            sync: true,
            mix: true,
            render: true,
        }
    }
}

/// Full on-disk subtitle document persisted at
/// `<project>/subtitles/subtitles.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleDoc {
    pub version: u32,
    pub source_language: String,
    pub target_language: String,
    pub segments: Vec<SubtitleSegment>,
    #[serde(default)]
    pub derived_from: DerivedFrom,
    #[serde(default)]
    pub dirty: DirtyFlags,
    #[serde(default)]
    pub next_id: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SubtitleDoc {
    pub fn empty(source_language: String, target_language: String) -> Self {
        let now = Utc::now();
        Self {
            version: SUBTITLE_SCHEMA_VERSION,
            source_language,
            target_language,
            segments: Vec::new(),
            derived_from: DerivedFrom {
                transcript_cache_key: None,
                translation_cache_key: None,
                origin: "manual".into(),
            },
            dirty: DirtyFlags::default(),
            next_id: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Reserve and return the next stable id.
    pub fn allocate_id(&mut self) -> u32 {
        // `next_id` should always be >= max(existing ids) + 1, but we
        // recompute defensively in case the file was hand-edited.
        let existing_max = self.segments.iter().map(|s| s.id).max().unwrap_or(0);
        let candidate = self.next_id.max(existing_max.saturating_add(1));
        self.next_id = candidate.saturating_add(1);
        candidate
    }

    /// Sort segments by start time. Used after operations that may
    /// disturb ordering (add, timing edit).
    pub fn sort_by_time(&mut self) {
        self.segments.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return `(index_of_first_overlap, index_of_second_overlap)` for
    /// each adjacent pair whose spans overlap after sorting. Purely a
    /// diagnostic; the model never rejects overlaps — the spec asks
    /// us to *warn* about them, not forbid them.
    pub fn overlap_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 1..self.segments.len() {
            let prev = &self.segments[i - 1];
            let cur = &self.segments[i];
            if cur.start < prev.end {
                out.push((i - 1, i));
            }
        }
        out
    }
}

/// Client-side patch used by `update_subtitle_segment`. `None` fields
/// leave the current value alone; `Some(_)` overwrites.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleSegmentPatch {
    #[serde(default)]
    pub start: Option<f64>,
    #[serde(default)]
    pub end: Option<f64>,
    #[serde(default)]
    pub source_text: Option<String>,
    #[serde(default)]
    pub translated_text: Option<String>,
    #[serde(default)]
    pub dubbing_text: Option<String>,
    /// `Some(Some(_))` = set, `Some(None)` = clear, `None` = leave.
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub speaker: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub voice_id: Option<Option<String>>,
}

fn deserialize_optional_option<'de, D>(deserializer: D) -> Result<Option<Option<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Wire distinguishes {absent, null, string}: absent -> None (no
    // change), null -> Some(None) (clear), string -> Some(Some(s)).
    // Since serde's default treats null the same as absent, we accept
    // an explicit sentinel string `""` to mean "clear" in addition to
    // JSON null.
    let opt: Option<String> = serde::Deserialize::deserialize(deserializer)?;
    Ok(Some(opt.filter(|s| !s.is_empty())))
}

impl SubtitleSegmentPatch {
    /// Returns `true` iff this patch touches a field whose change
    /// invalidates the TTS output itself (text / speaker / voice).
    pub fn touches_content(&self) -> bool {
        self.translated_text.is_some()
            || self.dubbing_text.is_some()
            || self.speaker.is_some()
            || self.voice_id.is_some()
    }

    /// Returns `true` iff this patch changes timing without touching
    /// the TTS input; only sync / mix / render need to re-run.
    pub fn touches_timing(&self) -> bool {
        self.start.is_some() || self.end.is_some()
    }

    /// Kept for backward compatibility: true whenever *anything*
    /// downstream may need invalidating.
    pub fn touches_downstream(&self) -> bool {
        self.touches_content() || self.touches_timing()
    }
}

// ------------------------------------------------------------------ summary

/// Compact summary the frontend uses in the media panel and the
/// dashboard — never ship the whole segment array unless the user
/// opens the editor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleSummary {
    pub source_language: String,
    pub target_language: String,
    pub segment_count: u32,
    pub translated_count: u32,
    pub speaker_count: u32,
    pub overlap_count: u32,
    pub dirty: DirtyFlags,
    pub derived_from: DerivedFrom,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub relative_path: String,
}

impl SubtitleSummary {
    pub fn from_doc(doc: &SubtitleDoc, relative_path: &str) -> Self {
        use std::collections::BTreeSet;
        let translated = doc
            .segments
            .iter()
            .filter(|s| !s.translated_text.trim().is_empty())
            .count() as u32;
        let speakers: BTreeSet<&str> = doc
            .segments
            .iter()
            .filter_map(|s| s.speaker.as_deref())
            .filter(|s| !s.trim().is_empty())
            .collect();
        Self {
            source_language: doc.source_language.clone(),
            target_language: doc.target_language.clone(),
            segment_count: doc.segments.len() as u32,
            translated_count: translated,
            speaker_count: speakers.len() as u32,
            overlap_count: doc.overlap_pairs().len() as u32,
            dirty: doc.dirty,
            derived_from: doc.derived_from.clone(),
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            relative_path: relative_path.into(),
        }
    }
}

// ------------------------------------------------------------------- I/O DTOs

/// Which subtitle file format the parser produced or the exporter is
/// asked to produce. Wire-friendly, lowercased on the JSON side.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SubtitleFormat {
    Srt,
    Ass,
}

impl SubtitleFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "srt" => Some(Self::Srt),
            "ass" | "ssa" => Some(Self::Ass),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Ass => "ass",
        }
    }
}

/// What field to export as the subtitle text. Almost always
/// `Translated` (the whole point of the app), but exposing `Source`
/// and `Bilingual` costs nothing and helps debugging.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExportKind {
    #[default]
    Translated,
    Source,
    Bilingual,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportSubtitlesResult {
    pub path: String,
    pub format: SubtitleFormat,
    pub segment_count: u32,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportSubtitlesResult {
    pub doc: SubtitleDoc,
    pub format: SubtitleFormat,
    pub source_path: String,
    pub segment_count: u32,
}
