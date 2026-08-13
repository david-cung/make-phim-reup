//! Unit tests for the pure functions in the TTS module — planner,
//! summary, cache-key stability, manifest upsert semantics.

use chrono::Utc;

use crate::subtitles::models::DerivedFrom;
use crate::subtitles::{DirtyFlags, SubtitleDoc, SubtitleSegment};

use super::models::{
    build_segment_cache_key, text_hash, GenerateMode, GenerateRequest, TtsManifest,
    TtsSegmentEntry, TtsSettings,
};
use super::service::{build_summary, SummaryRequest};

fn seg(id: u32, src: &str, translated: &str, voice: Option<&str>) -> SubtitleSegment {
    SubtitleSegment {
        id,
        start: (id as f64) * 2.0,
        end: (id as f64) * 2.0 + 1.5,
        source_text: src.into(),
        translated_text: translated.into(),
        dubbing_text: translated.into(),
        words: None,
        speaker: None,
        voice_id: voice.map(str::to_string),
    }
}

fn doc(segments: Vec<SubtitleSegment>) -> SubtitleDoc {
    let now = Utc::now();
    let next_id = segments.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    SubtitleDoc {
        version: 1,
        source_language: "en".into(),
        target_language: "vi".into(),
        segments,
        derived_from: DerivedFrom::default(),
        dirty: DirtyFlags::default(),
        next_id,
        created_at: now,
        updated_at: now,
    }
}

fn entry(id: u32, text: &str, voice: &str, engine: &str, settings: TtsSettings) -> TtsSegmentEntry {
    let key = build_segment_cache_key(engine, voice, "model.onnx", text, &settings);
    TtsSegmentEntry {
        segment_id: id,
        engine: engine.into(),
        voice_id: voice.into(),
        model_name: "model.onnx".into(),
        cache_key: key,
        text_hash: text_hash(text),
        text: text.into(),
        speed: settings.speed,
        pitch: settings.pitch,
        volume: settings.volume,
        device: settings.device,
        file: format!("voices/{:06}.wav", id),
        duration_secs: 1.0,
        sample_rate: 22050,
        channels: 1,
        size_bytes: 44100,
        generated_at: Utc::now(),
    }
}

#[test]
fn cache_key_is_deterministic() {
    let s = TtsSettings::default();
    let a = build_segment_cache_key("piper", "vi_male_01", "model.onnx", "Xin chào", &s);
    let b = build_segment_cache_key("piper", "vi_male_01", "model.onnx", "Xin chào", &s);
    assert_eq!(a, b);
}

#[test]
fn cache_key_changes_with_text() {
    let s = TtsSettings::default();
    let a = build_segment_cache_key("piper", "vi_male_01", "model.onnx", "one", &s);
    let b = build_segment_cache_key("piper", "vi_male_01", "model.onnx", "two", &s);
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_settings() {
    let a = build_segment_cache_key(
        "piper",
        "vi_male_01",
        "model.onnx",
        "Xin chào",
        &TtsSettings {
            speed: 1.0,
            ..Default::default()
        },
    );
    let b = build_segment_cache_key(
        "piper",
        "vi_male_01",
        "model.onnx",
        "Xin chào",
        &TtsSettings {
            speed: 1.5,
            ..Default::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_voice() {
    let s = TtsSettings::default();
    let a = build_segment_cache_key("piper", "vi_male_01", "model.onnx", "hi", &s);
    let b = build_segment_cache_key("piper", "vi_female_01", "model.onnx", "hi", &s);
    assert_ne!(a, b);
}

#[test]
fn summary_counts_generated_missing_and_stale() {
    let doc = doc(vec![
        seg(1, "Hello", "Xin chào", None),
        seg(2, "How are you", "Bạn khỏe không", None),
        seg(3, "", "", None),
    ]);
    let mut manifest = TtsManifest::empty("piper".into(), "vi_male_01".into());
    manifest.upsert(entry(
        1,
        "Xin chào",
        "vi_male_01",
        "piper",
        TtsSettings::default(),
    ));
    // Entry 2 is stale — text has changed since generation.
    manifest.upsert(entry(
        2,
        "Bạn khỏe không?",
        "vi_male_01",
        "piper",
        TtsSettings::default(),
    ));
    let req = SummaryRequest {
        engine: "piper".into(),
        default_voice_id: "vi_male_01".into(),
        settings: TtsSettings::default(),
    };
    let sum = build_summary(&doc, Some(&manifest), &req);
    assert_eq!(sum.subtitle_count, 3);
    assert_eq!(sum.generated_count, 1);
    assert_eq!(sum.missing_count, 1);
    assert_eq!(sum.stale_count, 1);
}

#[test]
fn manifest_upsert_replaces_existing() {
    let mut m = TtsManifest::empty("piper".into(), "vi_male_01".into());
    m.upsert(entry(1, "a", "vi_male_01", "piper", TtsSettings::default()));
    m.upsert(entry(1, "b", "vi_male_01", "piper", TtsSettings::default()));
    assert_eq!(m.segments.len(), 1);
    assert_eq!(m.segments[0].text_hash, text_hash("b"));
}

#[test]
fn manifest_upsert_keeps_segments_sorted() {
    let mut m = TtsManifest::empty("piper".into(), "vi_male_01".into());
    m.upsert(entry(3, "c", "v", "piper", TtsSettings::default()));
    m.upsert(entry(1, "a", "v", "piper", TtsSettings::default()));
    m.upsert(entry(2, "b", "v", "piper", TtsSettings::default()));
    let ids: Vec<u32> = m.segments.iter().map(|s| s.segment_id).collect();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn manifest_remove_is_a_noop_when_absent() {
    let mut m = TtsManifest::empty("piper".into(), "vi_male_01".into());
    assert!(!m.remove(42));
    m.upsert(entry(1, "a", "v", "piper", TtsSettings::default()));
    assert!(m.remove(1));
    assert!(m.segments.is_empty());
}

#[test]
fn settings_normalisation_clamps_ranges() {
    let s = TtsSettings {
        speed: 100.0,
        pitch: -100.0,
        volume: -1.0,
        ..Default::default()
    }
    .normalised();
    assert!((s.speed - 4.0).abs() < 1e-6);
    assert!((s.pitch - -12.0).abs() < 1e-6);
    assert!((s.volume - 0.0).abs() < 1e-6);
}

#[test]
fn generate_mode_default_is_missing() {
    // Round-trip through JSON matches the Python default too.
    let v = serde_json::to_value(GenerateRequest {
        engine: "piper".into(),
        default_voice_id: "v".into(),
        settings: TtsSettings::default(),
        mode: GenerateMode::default(),
    })
    .unwrap();
    assert_eq!(v["mode"]["kind"], "missing");
}
