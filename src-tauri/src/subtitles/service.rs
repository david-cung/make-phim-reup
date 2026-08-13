//! Host-side orchestration for the subtitle editor.
//!
//! All mutations use atomic writes for the JSON file and grab a
//! per-service mutex around the on-disk `subtitles.json` before
//! load-modify-save so overlapping requests cannot lose an edit.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::projects::{ProjectError, ProjectService};
use crate::stt::TranscriptCacheFile;
use crate::translation::TranslationCacheFile;

use super::ass;
use super::cache::{SubtitleCacheFile, SUBTITLES_RELATIVE};
use super::derive::{derive_from_sources, merge_preserving_edits};
use super::errors::SubtitleError;
use super::models::{
    DirtyFlags, ExportKind, ExportSubtitlesResult, ImportSubtitlesResult, SubtitleDoc,
    SubtitleFormat, SubtitleSegment, SubtitleSegmentPatch, SubtitleSummary,
};
use super::srt;

/// A tiny wrapper struct so callers get one Arc that owns the whole
/// subtitle capability.
pub struct SubtitleService {
    projects: Arc<ProjectService>,
    /// Serialise load-modify-save cycles per project. A single global
    /// mutex is fine — subtitle mutations are fast, and the UI never
    /// issues more than one at a time from a single project screen.
    doc_lock: Arc<Mutex<()>>,
}

impl SubtitleService {
    pub fn new(projects: Arc<ProjectService>) -> Arc<Self> {
        Arc::new(Self {
            projects,
            doc_lock: Arc::new(Mutex::new(())),
        })
    }

    // -------------------------------------------------------- read-only

    pub async fn get_summary(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<SubtitleSummary>, SubtitleError> {
        let root = self.project_root(&project_id).await?;
        match SubtitleCacheFile::load(&root).map_err(io_err(&root))? {
            Some(doc) => Ok(Some(SubtitleSummary::from_doc(&doc, SUBTITLES_RELATIVE))),
            None => Ok(None),
        }
    }

    pub async fn get_doc(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<Option<SubtitleDoc>, SubtitleError> {
        let root = self.project_root(&project_id).await?;
        SubtitleCacheFile::load(&root).map_err(io_err(&root))
    }

    // --------------------------------------------------------- rebuild

    /// Rebuild the subtitle document from the current transcript
    /// (required) and translation (optional). If a document already
    /// exists, user edits (speaker / voice / manual translations) are
    /// preserved wherever the transcript id still matches.
    pub async fn rebuild_from_sources(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<SubtitleDoc, SubtitleError> {
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);
        let source_language = rec.source_language.clone();
        let target_language = rec.target_language.clone();

        let _guard = self.doc_lock.lock();
        let transcript = TranscriptCacheFile::load(&root)
            .map_err(io_err(&root))?
            .ok_or(SubtitleError::NoTranscript)?;
        let translation = TranslationCacheFile::load(&root).map_err(io_err(&root))?;
        let previous = SubtitleCacheFile::load(&root).map_err(io_err(&root))?;

        let fresh = derive_from_sources(
            &transcript,
            translation.as_ref(),
            &source_language,
            &target_language,
        );
        let merged = merge_preserving_edits(fresh, previous.as_ref());
        SubtitleCacheFile::save(&root, &merged).map_err(io_err(&root))?;
        Ok(merged)
    }

    // ---------------------------------------------------------- edits

    pub async fn update_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
        patch: SubtitleSegmentPatch,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            // Decide dirty semantics *before* we mutate: content
            // changes invalidate TTS (and thus everything downstream);
            // timing-only changes only invalidate sync/mix/render.
            let content = patch.touches_content();
            let timing = patch.touches_timing();
            let idx = doc
                .segments
                .iter()
                .position(|s| s.id == segment_id)
                .ok_or(SubtitleError::SegmentNotFound { id: segment_id })?;
            let seg = &mut doc.segments[idx];
            if let Some(v) = patch.start {
                seg.start = v;
            }
            if let Some(v) = patch.end {
                seg.end = v;
            }
            if let Some(v) = patch.source_text {
                seg.source_text = v;
            }
            if let Some(v) = patch.translated_text {
                seg.translated_text = v;
            }
            if let Some(v) = patch.dubbing_text {
                seg.dubbing_text = v;
            }
            if let Some(v) = patch.speaker {
                seg.speaker = v;
            }
            if let Some(v) = patch.voice_id {
                seg.voice_id = v;
            }
            seg.validate_timing()
                .map_err(|reason| SubtitleError::InvalidTiming { reason })?;
            if content {
                doc.dirty.mark_content_dirty();
            } else if timing {
                doc.dirty.mark_timing_dirty();
            }
            Ok(())
        })
        .await
    }

    pub async fn add_segment(
        self: &Arc<Self>,
        project_id: String,
        after_id: Option<u32>,
        start: f64,
        end: f64,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            // Default to inserting right after the `after_id` segment
            // (or at the end if unspecified).
            let insert_pos = match after_id {
                Some(id) => doc
                    .segments
                    .iter()
                    .position(|s| s.id == id)
                    .map(|i| i + 1)
                    .ok_or(SubtitleError::SegmentNotFound { id })?,
                None => doc.segments.len(),
            };
            let new_id = doc.allocate_id();
            let seg = SubtitleSegment {
                id: new_id,
                start,
                end,
                source_text: String::new(),
                translated_text: String::new(),
                dubbing_text: String::new(),
                words: None,
                speaker: None,
                voice_id: None,
            };
            seg.validate_timing()
                .map_err(|reason| SubtitleError::InvalidTiming { reason })?;
            doc.segments.insert(insert_pos, seg);
            doc.sort_by_time();
            doc.dirty.mark_downstream();
            Ok(())
        })
        .await
    }

    pub async fn delete_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            let before = doc.segments.len();
            doc.segments.retain(|s| s.id != segment_id);
            if doc.segments.len() == before {
                return Err(SubtitleError::SegmentNotFound { id: segment_id });
            }
            doc.dirty.mark_downstream();
            Ok(())
        })
        .await
    }

    /// Split a segment at `split_time`. The original segment's `id`
    /// is preserved on the first half; the second half gets a fresh
    /// id from the allocator so downstream keys don't collide.
    pub async fn split_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
        split_time: f64,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            let idx = doc
                .segments
                .iter()
                .position(|s| s.id == segment_id)
                .ok_or(SubtitleError::SegmentNotFound { id: segment_id })?;
            let original = doc.segments[idx].clone();
            if split_time <= original.start + 1e-6 || split_time >= original.end - 1e-6 {
                return Err(SubtitleError::InvalidSplit {
                    id: segment_id,
                    time: split_time,
                });
            }
            let new_id = doc.allocate_id();
            let mut left = original.clone();
            let mut right = original;
            left.end = split_time;
            right.id = new_id;
            right.start = split_time;
            // Split the translated text at the closest whitespace to
            // the midpoint of the substring. Cheap heuristic, gives
            // the user a reasonable starting point.
            let (l_text, r_text) = split_text_in_half(&left.translated_text);
            left.translated_text = l_text;
            right.translated_text = r_text;
            let (l_src, r_src) = split_text_in_half(&left.source_text);
            left.source_text = l_src;
            right.source_text = r_src;
            let (l_dub, r_dub) = split_text_in_half(&left.dubbing_text);
            left.dubbing_text = l_dub;
            right.dubbing_text = r_dub;
            left.words = None;
            right.words = None;
            doc.segments[idx] = left;
            doc.segments.insert(idx + 1, right);
            doc.dirty.mark_downstream();
            Ok(())
        })
        .await
    }

    /// Merge `segment_id` with the following segment. The merged
    /// row keeps `segment_id`.
    pub async fn merge_segment(
        self: &Arc<Self>,
        project_id: String,
        segment_id: u32,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            let idx = doc
                .segments
                .iter()
                .position(|s| s.id == segment_id)
                .ok_or(SubtitleError::SegmentNotFound { id: segment_id })?;
            if idx + 1 >= doc.segments.len() {
                return Err(SubtitleError::NoMergeTarget { id: segment_id });
            }
            let next = doc.segments.remove(idx + 1);
            let cur = &mut doc.segments[idx];
            cur.end = next.end.max(cur.end);
            cur.source_text = join_nonempty(&cur.source_text, &next.source_text, " ");
            cur.translated_text = join_nonempty(&cur.translated_text, &next.translated_text, " ");
            cur.dubbing_text = join_nonempty(&cur.dubbing_text, &next.dubbing_text, " ");
            if cur.speaker.is_none() {
                cur.speaker = next.speaker;
            }
            if cur.voice_id.is_none() {
                cur.voice_id = next.voice_id;
            }
            cur.validate_timing()
                .map_err(|reason| SubtitleError::InvalidTiming { reason })?;
            doc.dirty.mark_downstream();
            Ok(())
        })
        .await
    }

    /// Clear every downstream dirty flag at once — used by the UI's
    /// explicit "mark clean" action.
    pub async fn clear_dirty(
        self: &Arc<Self>,
        project_id: String,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.clear_dirty_flags(project_id, DirtyFlags::all()).await
    }

    /// Clear a specific subset of dirty flags after a successful
    /// downstream pass. `mask.tts = true` clears the tts flag, etc.
    /// This is the entry point Phase 6 (TTS) and Phase 7 (sync) call.
    pub async fn clear_dirty_flags(
        self: &Arc<Self>,
        project_id: String,
        mask: DirtyFlags,
    ) -> Result<SubtitleDoc, SubtitleError> {
        self.mutate(project_id, |doc| {
            doc.dirty.clear_where(mask);
            Ok(())
        })
        .await
    }

    // -------------------------------------------------- import / export

    pub async fn import_from_file(
        self: &Arc<Self>,
        project_id: String,
        source_path: String,
    ) -> Result<ImportSubtitlesResult, SubtitleError> {
        let path = PathBuf::from(source_path.trim());
        if path.as_os_str().is_empty() || !path.is_absolute() {
            return Err(SubtitleError::InvalidExportPath);
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let format = SubtitleFormat::from_extension(&ext).ok_or_else(|| {
            SubtitleError::UnsupportedFormat {
                path: path.display().to_string(),
            }
        })?;
        let text = std::fs::read_to_string(&path).map_err(|source| SubtitleError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let rec = self
            .projects
            .open(project_id.clone())
            .await
            .map_err(map_project_err)?;
        let root = PathBuf::from(&rec.root_path);

        let _guard = self.doc_lock.lock();
        let segments = match format {
            SubtitleFormat::Srt => {
                srt::parse(&text, 1).map_err(|reason| SubtitleError::InvalidSubtitleFile {
                    path: path.display().to_string(),
                    reason,
                })?
            }
            SubtitleFormat::Ass => {
                ass::parse(&text, 1).map_err(|reason| SubtitleError::InvalidSubtitleFile {
                    path: path.display().to_string(),
                    reason,
                })?
            }
        };

        // Merge imported translations onto the current source text
        // when we have a transcript to align against. If not, treat
        // the file as fully authoritative — the imported text goes
        // into both `source` and `translated` so users can still
        // edit meaningfully.
        let transcript = TranscriptCacheFile::load(&root).map_err(io_err(&root))?;
        let mut doc = if let Some(t) = transcript {
            let mut base =
                derive_from_sources(&t, None, &rec.source_language, &rec.target_language);
            base.derived_from.origin = format_origin(format);
            for imported in &segments {
                if let Some(target) = base
                    .segments
                    .iter_mut()
                    .find(|s| overlaps(s.start, s.end, imported.start, imported.end))
                {
                    target.translated_text = imported.translated_text.clone();
                    if imported.speaker.is_some() {
                        target.speaker = imported.speaker.clone();
                    }
                }
            }
            base
        } else {
            let now = chrono::Utc::now();
            let mut base = SubtitleDoc {
                version: super::models::SUBTITLE_SCHEMA_VERSION,
                source_language: rec.source_language.clone(),
                target_language: rec.target_language.clone(),
                segments: segments
                    .iter()
                    .map(|s| SubtitleSegment {
                        source_text: s.translated_text.clone(),
                        ..s.clone()
                    })
                    .collect(),
                derived_from: super::models::DerivedFrom {
                    transcript_cache_key: None,
                    translation_cache_key: None,
                    origin: format_origin(format),
                },
                dirty: super::models::DirtyFlags::default(),
                next_id: segments
                    .iter()
                    .map(|s| s.id)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1),
                created_at: now,
                updated_at: now,
            };
            base.sort_by_time();
            base
        };
        doc.dirty.mark_downstream();
        doc.touch();
        SubtitleCacheFile::save(&root, &doc).map_err(io_err(&root))?;
        let count = doc.segments.len() as u32;
        Ok(ImportSubtitlesResult {
            doc,
            format,
            source_path: path.display().to_string(),
            segment_count: count,
        })
    }

    pub async fn export_to_file(
        self: &Arc<Self>,
        project_id: String,
        output_path: String,
        format: SubtitleFormat,
        kind: ExportKind,
    ) -> Result<ExportSubtitlesResult, SubtitleError> {
        let dest = PathBuf::from(output_path.trim());
        if dest.as_os_str().is_empty() || !dest.is_absolute() {
            return Err(SubtitleError::InvalidExportPath);
        }
        let root = self.project_root(&project_id).await?;
        let doc = SubtitleCacheFile::load(&root)
            .map_err(io_err(&root))?
            .ok_or(SubtitleError::NoSubtitles)?;
        let text_for = |s: &SubtitleSegment| match kind {
            ExportKind::Translated => s.translated_text.clone(),
            ExportKind::Source => s.source_text.clone(),
            ExportKind::Bilingual => {
                if s.source_text.is_empty() {
                    s.translated_text.clone()
                } else if s.translated_text.is_empty() {
                    s.source_text.clone()
                } else {
                    format!("{}\n{}", s.translated_text, s.source_text)
                }
            }
        };
        let body = match format {
            SubtitleFormat::Srt => srt::write(&doc.segments, text_for),
            SubtitleFormat::Ass => ass::write(
                &doc.segments,
                &doc.source_language,
                &doc.target_language,
                text_for,
            ),
        };
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| SubtitleError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }
        // Atomic write: tmp + rename so a crash mid-export leaves the
        // previous file intact.
        let tmp = dest.with_extension(format!("{}.tmp", format.extension()));
        std::fs::write(&tmp, body.as_bytes()).map_err(|source| SubtitleError::Io {
            path: tmp.display().to_string(),
            source,
        })?;
        std::fs::rename(&tmp, &dest).map_err(|source| SubtitleError::Io {
            path: dest.display().to_string(),
            source,
        })?;
        Ok(ExportSubtitlesResult {
            path: dest.display().to_string(),
            format,
            segment_count: doc.segments.len() as u32,
            bytes_written: body.len() as u64,
        })
    }

    // -------------------------------------------------------- internal

    async fn project_root(&self, project_id: &str) -> Result<PathBuf, SubtitleError> {
        let rec = self
            .projects
            .open(project_id.to_string())
            .await
            .map_err(map_project_err)?;
        Ok(PathBuf::from(&rec.root_path))
    }

    /// Load `subtitles.json`, apply `mutator`, write back atomically.
    /// If `mutator` returns `Err`, nothing is written.
    async fn mutate<F>(
        self: &Arc<Self>,
        project_id: String,
        mutator: F,
    ) -> Result<SubtitleDoc, SubtitleError>
    where
        F: FnOnce(&mut SubtitleDoc) -> Result<(), SubtitleError>,
    {
        let root = self.project_root(&project_id).await?;
        let doc_lock = self.doc_lock.clone();
        let _guard = doc_lock.lock();
        let mut doc = SubtitleCacheFile::load(&root)
            .map_err(io_err(&root))?
            .ok_or(SubtitleError::NoSubtitles)?;
        mutator(&mut doc)?;
        doc.touch();
        SubtitleCacheFile::save(&root, &doc).map_err(io_err(&root))?;
        Ok(doc)
    }
}

fn map_project_err(err: ProjectError) -> SubtitleError {
    match err {
        ProjectError::Db(d) => SubtitleError::Db(d),
        other => SubtitleError::Io {
            path: String::new(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

fn io_err(root: &Path) -> impl Fn(std::io::Error) -> SubtitleError + '_ {
    move |source| SubtitleError::Io {
        path: root.display().to_string(),
        source,
    }
}

fn overlaps(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> bool {
    a_start < b_end && b_start < a_end
}

fn split_text_in_half(text: &str) -> (String, String) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (String::new(), String::new());
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mid = chars.len() / 2;
    // Prefer splitting at whitespace within a small window around
    // the midpoint so we don't chop words in half.
    let window = (chars.len() / 6).max(1);
    let lo = mid.saturating_sub(window);
    let hi = (mid + window).min(chars.len());
    let split_at = (lo..hi)
        .rev()
        .find(|&i| chars[i].is_whitespace())
        .unwrap_or(mid);
    let left: String = chars[..split_at].iter().collect();
    let right: String = chars[split_at..].iter().collect();
    (left.trim().to_string(), right.trim().to_string())
}

fn join_nonempty(a: &str, b: &str, sep: &str) -> String {
    match (a.trim().is_empty(), b.trim().is_empty()) {
        (true, true) => String::new(),
        (true, false) => b.to_string(),
        (false, true) => a.to_string(),
        (false, false) => format!("{a}{sep}{b}"),
    }
}

fn format_origin(fmt: SubtitleFormat) -> String {
    match fmt {
        SubtitleFormat::Srt => "srt-import".into(),
        SubtitleFormat::Ass => "ass-import".into(),
    }
}
