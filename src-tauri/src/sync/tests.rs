//! Pure-Rust unit tests for the sync module.
//!
//! These stay entirely in-memory — no worker, no FFmpeg. They cover
//! the classifier + cache-key contract that the Python side has to
//! agree with.

use chrono::Utc;

use crate::subtitles::models::DerivedFrom;
use crate::subtitles::{DirtyFlags, SubtitleDoc, SubtitleSegment};
use crate::tts::{
    build_segment_cache_key, text_hash, TtsManifest, TtsSegmentEntry, TtsSettings,
    TTS_CACHE_SCHEMA_VERSION,
};

use super::models::{
    build_sync_cache_key, SyncManifest, SyncMode, SyncRequest, SyncSegmentEntry, SyncSettings,
    SyncStatus,
};
use super::service::{build_summary, plan_for, plan_generation};

fn subtitle(id: u32, start: f64, end: f64, text: &str) -> SubtitleSegment {
    SubtitleSegment {
        id,
        start,
        end,
        source_text: String::new(),
        translated_text: text.into(),
        dubbing_text: text.into(),
        words: None,
        speaker: None,
        speaker_confidence: None,
        voice_id: Some("vi-narrator".into()),
    }
}

fn subtitle_doc(segs: Vec<SubtitleSegment>) -> SubtitleDoc {
    let now = Utc::now();
    let next = segs.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    SubtitleDoc {
        version: 1,
        source_language: "en".into(),
        target_language: "vi".into(),
        segments: segs,
        derived_from: DerivedFrom::default(),
        dirty: DirtyFlags::default(),
        next_id: next,
        created_at: now,
        updated_at: now,
    }
}

fn tts_entry(id: u32, text: &str, duration: f64, sample_rate: u32) -> TtsSegmentEntry {
    let settings = TtsSettings::default();
    let cache_key = build_segment_cache_key("piper", "vi-narrator", "model.onnx", text, &settings);
    TtsSegmentEntry {
        segment_id: id,
        engine: "piper".into(),
        voice_id: "vi-narrator".into(),
        model_name: "model.onnx".into(),
        cache_key,
        text_hash: text_hash(text),
        text: text.into(),
        speed: 1.0,
        pitch: 0.0,
        volume: 1.0,
        device: settings.device,
        file: format!("voices/{id:06}.wav"),
        duration_secs: duration,
        sample_rate,
        channels: 1,
        size_bytes: 1024,
        generated_at: Utc::now(),
    }
}

fn tts_manifest(entries: Vec<TtsSegmentEntry>) -> TtsManifest {
    let now = Utc::now();
    TtsManifest {
        version: TTS_CACHE_SCHEMA_VERSION,
        engine: "piper".into(),
        default_voice_id: "vi-narrator".into(),
        voice_profiles: Vec::new(),
        segments: entries,
        created_at: now,
        updated_at: now,
    }
}

fn sync_entry_for(
    seg: &SubtitleSegment,
    tts: &TtsSegmentEntry,
    settings: &SyncSettings,
) -> SyncSegmentEntry {
    let target = (seg.end - seg.start).max(0.0);
    let plan = plan_for(target, tts.duration_secs, settings);
    let cache_key = build_sync_cache_key(&tts.cache_key, target, settings);
    SyncSegmentEntry {
        segment_id: seg.id,
        status: plan.status,
        target_start: seg.start,
        target_end: seg.end,
        target_duration_secs: target,
        original_duration_secs: tts.duration_secs,
        final_duration_secs: plan.final_duration_secs,
        speed_factor: plan.speed_factor,
        cache_key,
        tts_cache_key: tts.cache_key.clone(),
        file: format!("voices/synced/{:06}.wav", seg.id),
        sample_rate: tts.sample_rate,
        channels: 1,
        size_bytes: 4096,
        generated_at: Utc::now(),
    }
}

fn sync_manifest(entries: Vec<SyncSegmentEntry>, settings: SyncSettings) -> SyncManifest {
    let now = Utc::now();
    SyncManifest {
        version: 1,
        settings,
        segments: entries,
        created_at: now,
        updated_at: now,
    }
}

// -----------------------------------------------------------------
// Planner classification
// -----------------------------------------------------------------

#[test]
fn planner_marks_short_voice_as_fits() {
    let plan = plan_for(4.0, 2.0, &SyncSettings::default());
    assert_eq!(plan.status, SyncStatus::Fits);
    assert!((plan.final_duration_secs - 4.0).abs() < 1e-9);
    assert!((plan.speed_factor - 1.0).abs() < 1e-9);
}

#[test]
fn planner_marks_voice_within_range_as_adjusted() {
    let plan = plan_for(4.0, 4.4, &SyncSettings::default());
    assert_eq!(plan.status, SyncStatus::Adjusted);
    assert!(plan.speed_factor > 1.0 && plan.speed_factor <= 1.12 + 1e-6);
    assert!((plan.final_duration_secs - 4.0).abs() < 1e-3);
}

#[test]
fn planner_marks_extreme_voice_as_too_long() {
    let plan = plan_for(2.0, 4.0, &SyncSettings::default());
    assert_eq!(plan.status, SyncStatus::TooLong);
    assert!((plan.speed_factor - 1.12).abs() < 1e-6);
    // Still emit a stretched clip, longer than target.
    assert!(plan.final_duration_secs > 2.0);
}

#[test]
fn planner_marks_missing_source_as_empty() {
    let plan = plan_for(3.0, 0.0, &SyncSettings::default());
    assert_eq!(plan.status, SyncStatus::Empty);
    assert!((plan.final_duration_secs - 3.0).abs() < 1e-9);
}

// -----------------------------------------------------------------
// Cache key stability
// -----------------------------------------------------------------

#[test]
fn cache_key_changes_when_target_duration_changes() {
    let s = SyncSettings::default();
    let a = build_sync_cache_key("sha256:tts", 4.20, &s);
    let b = build_sync_cache_key("sha256:tts", 4.30, &s);
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_when_tts_cache_key_changes() {
    let s = SyncSettings::default();
    let a = build_sync_cache_key("sha256:aaa", 4.20, &s);
    let b = build_sync_cache_key("sha256:bbb", 4.20, &s);
    assert_ne!(a, b);
}

#[test]
fn cache_key_stable_across_ms_wobble() {
    let s = SyncSettings::default();
    let a = build_sync_cache_key("sha256:tts", 4.2000001, &s);
    let b = build_sync_cache_key("sha256:tts", 4.20000001, &s);
    assert_eq!(a, b);
}

// -----------------------------------------------------------------
// Planning + summary
// -----------------------------------------------------------------

#[test]
fn plan_generation_skips_matching_cache_hits() {
    let s = SyncSettings::default();
    let seg = subtitle(1, 0.0, 4.0, "xin chào");
    let tts = tts_entry(1, "xin chào", 3.5, 22050);
    let synced = sync_entry_for(&seg, &tts, &s);
    let doc = subtitle_doc(vec![seg]);
    let tts_manifest = tts_manifest(vec![tts]);
    let manifest = sync_manifest(vec![synced], s);
    let request = SyncRequest {
        settings: s,
        mode: SyncMode::Missing,
    };
    let todo = plan_generation(&doc, &tts_manifest, Some(&manifest), &request, &s);
    assert!(
        todo.is_empty(),
        "no work expected when cache identity matches"
    );
}

#[test]
fn plan_generation_forces_all_when_requested() {
    let s = SyncSettings::default();
    let seg = subtitle(1, 0.0, 4.0, "hello");
    let tts = tts_entry(1, "hello", 3.0, 22050);
    let synced = sync_entry_for(&seg, &tts, &s);
    let doc = subtitle_doc(vec![seg]);
    let tts_manifest = tts_manifest(vec![tts]);
    let manifest = sync_manifest(vec![synced], s);
    let request = SyncRequest {
        settings: s,
        mode: SyncMode::All,
    };
    let todo = plan_generation(&doc, &tts_manifest, Some(&manifest), &request, &s);
    assert_eq!(todo.len(), 1);
}

#[test]
fn plan_generation_skips_missing_tts() {
    let s = SyncSettings::default();
    let seg = subtitle(1, 0.0, 4.0, "hi");
    let doc = subtitle_doc(vec![seg]);
    let tts_manifest = tts_manifest(vec![]);
    let request = SyncRequest::default();
    let todo = plan_generation(&doc, &tts_manifest, None, &request, &s);
    assert!(todo.is_empty(), "no TTS → nothing to sync");
}

#[test]
fn summary_counts_fits_and_stale() {
    let s = SyncSettings::default();
    let a = subtitle(1, 0.0, 4.0, "one");
    let b = subtitle(2, 5.0, 6.0, "two");
    let tts_a = tts_entry(1, "one", 3.5, 22050);
    let tts_b = tts_entry(2, "two", 0.8, 22050);
    let mut synced_a = sync_entry_for(&a, &tts_a, &s);
    // Force a stale entry for `b`.
    synced_a.cache_key = "sha256:zzz".into();
    let doc = subtitle_doc(vec![a.clone(), b.clone()]);
    let tts_m = tts_manifest(vec![tts_a.clone(), tts_b.clone()]);
    let sync_m = sync_manifest(vec![synced_a], s);
    let summary = build_summary(&doc, Some(&tts_m), Some(&sync_m), &s);
    assert_eq!(summary.subtitle_count, 2);
    assert_eq!(summary.synced_count, 0);
    assert_eq!(summary.stale_count, 1);
    assert_eq!(summary.missing_count, 1);
}

#[test]
fn timing_change_marks_only_sync_mix_render() {
    let mut d = DirtyFlags::default();
    d.mark_timing_dirty();
    assert!(!d.tts, "timing change must NOT invalidate TTS");
    assert!(d.sync);
    assert!(d.mix);
    assert!(d.render);
}

#[test]
fn content_change_invalidates_tts_and_downstream() {
    let mut d = DirtyFlags::default();
    d.mark_content_dirty();
    assert!(d.tts);
    assert!(d.sync);
    assert!(d.mix);
    assert!(d.render);
}

#[test]
fn clear_where_only_clears_masked_flags() {
    let mut d = DirtyFlags {
        tts: true,
        sync: true,
        mix: true,
        render: true,
    };
    d.clear_where(DirtyFlags::only_sync());
    assert!(d.tts);
    assert!(!d.sync);
    assert!(d.mix);
    assert!(d.render);
}
