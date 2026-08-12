//! Pure-Rust unit tests for the mix module.
//!
//! No FFmpeg, no worker. Cover the invariants the frontend depends on:
//! cache-key stability, filter-graph shape, and the planner ↔ summary
//! contract.

use std::path::PathBuf;

use chrono::Utc;

use crate::media::SourceFingerprint;
use crate::subtitles::models::DerivedFrom;
use crate::subtitles::{DirtyFlags, SubtitleDoc, SubtitleSegment};
use crate::sync::{
    build_sync_cache_key, SyncManifest, SyncSegmentEntry, SyncSettings, SyncStatus,
    SYNC_CACHE_SCHEMA_VERSION,
};

use super::ffmpeg_cmd::{build_filter_graph, build_mix_command, duck_ratio_from_depth_db};
use super::models::{
    build_mix_cache_key, MixManifest, MixSettings, MixStatus, MixVoiceInput,
    MIX_CACHE_SCHEMA_VERSION,
};
use super::service::{build_summary, collect_voice_inputs};

// -----------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------

fn fingerprint() -> SourceFingerprint {
    SourceFingerprint {
        hash: "sha256:deadbeef".into(),
        size_bytes: 1_000_000,
        modified_at: Utc::now(),
    }
}

fn subtitle(id: u32, start: f64, end: f64) -> SubtitleSegment {
    SubtitleSegment {
        id,
        start,
        end,
        source_text: String::new(),
        translated_text: "xin chào".into(),
        speaker: None,
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

fn sync_entry(id: u32, start: f64, end: f64, status: SyncStatus) -> SyncSegmentEntry {
    let target = (end - start).max(0.0);
    let tts_key = format!("sha256:tts_{id}");
    let s = SyncSettings::default();
    let cache_key = build_sync_cache_key(&tts_key, target, &s);
    SyncSegmentEntry {
        segment_id: id,
        status,
        target_start: start,
        target_end: end,
        target_duration_secs: target,
        original_duration_secs: target * 0.9,
        final_duration_secs: target,
        speed_factor: 1.0,
        cache_key,
        tts_cache_key: tts_key,
        file: format!("voices/synced/{id:06}.wav"),
        sample_rate: 22050,
        channels: 1,
        size_bytes: 4096,
        generated_at: Utc::now(),
    }
}

fn sync_manifest(entries: Vec<SyncSegmentEntry>) -> SyncManifest {
    let now = Utc::now();
    SyncManifest {
        version: SYNC_CACHE_SCHEMA_VERSION,
        settings: SyncSettings::default(),
        segments: entries,
        created_at: now,
        updated_at: now,
    }
}

fn voice(id: u32, start: f64, key: &str, empty: bool) -> MixVoiceInput {
    MixVoiceInput {
        segment_id: id,
        target_start_secs: start,
        relative_file: format!("voices/synced/{id:06}.wav"),
        sync_cache_key: key.into(),
        is_empty: empty,
    }
}

// -----------------------------------------------------------------
// Cache key stability
// -----------------------------------------------------------------

#[test]
fn cache_key_stable_across_reorder() {
    let s = MixSettings::default();
    let fp = fingerprint();
    let a = build_mix_cache_key(
        &fp,
        &[voice(2, 5.0, "k2", false), voice(1, 0.0, "k1", false)],
        &s,
    );
    let b = build_mix_cache_key(
        &fp,
        &[voice(1, 0.0, "k1", false), voice(2, 5.0, "k2", false)],
        &s,
    );
    assert_eq!(a, b, "voice order must not affect cache key");
}

#[test]
fn cache_key_changes_with_settings() {
    let fp = fingerprint();
    let voices = vec![voice(1, 0.0, "k1", false)];
    let a = build_mix_cache_key(&fp, &voices, &MixSettings::default());
    let b = build_mix_cache_key(
        &fp,
        &voices,
        &MixSettings {
            voice_volume: 0.5,
            ..MixSettings::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_ducking_toggle() {
    let fp = fingerprint();
    let voices = vec![voice(1, 0.0, "k1", false)];
    let a = build_mix_cache_key(
        &fp,
        &voices,
        &MixSettings {
            ducking_enabled: true,
            ..MixSettings::default()
        },
    );
    let b = build_mix_cache_key(
        &fp,
        &voices,
        &MixSettings {
            ducking_enabled: false,
            ..MixSettings::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_source_fingerprint() {
    let voices = vec![voice(1, 0.0, "k1", false)];
    let a = build_mix_cache_key(&fingerprint(), &voices, &MixSettings::default());
    let mut fp2 = fingerprint();
    fp2.hash = "sha256:beadfeed".into();
    let b = build_mix_cache_key(&fp2, &voices, &MixSettings::default());
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_voice_cache_key() {
    let fp = fingerprint();
    let a = build_mix_cache_key(&fp, &[voice(1, 0.0, "k1", false)], &MixSettings::default());
    let b = build_mix_cache_key(&fp, &[voice(1, 0.0, "k2", false)], &MixSettings::default());
    assert_ne!(a, b);
}

#[test]
fn cache_key_versioned() {
    // Guard against a silent schema bump — anyone changing the version
    // constant must accept the fallout of invalidating everyone's mix.
    // v2 rebalanced the defaults and added the output limiter, both of
    // which change the WAV without changing any user-visible setting, so
    // the bump is what forces existing mixes to regenerate.
    assert_eq!(MIX_CACHE_SCHEMA_VERSION, 2);
    let fp = fingerprint();
    let key = build_mix_cache_key(&fp, &[], &MixSettings::default());
    assert!(key.contains("sha256:"));
    // Just make sure the function actually hashed something.
    assert!(key.len() > 20);
}

// -----------------------------------------------------------------
// Filter graph shape
// -----------------------------------------------------------------

#[test]
fn filter_graph_includes_adelay_for_each_voice() {
    let voices = [voice(1, 0.5, "k1", false), voice(2, 12.75, "k2", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();
    let g = build_filter_graph(&refs, &MixSettings::default());
    // 500ms and 12750ms delays land on integer millisecond boundaries.
    assert!(g.contains("adelay=delays=500:all=1"));
    assert!(g.contains("adelay=delays=12750:all=1"));
}

#[test]
fn filter_graph_uses_sidechaincompress_when_ducking_on() {
    let voices = [voice(1, 0.0, "k1", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();
    let g = build_filter_graph(
        &refs,
        &MixSettings {
            ducking_enabled: true,
            ..MixSettings::default()
        },
    );
    assert!(g.contains("sidechaincompress"));
    assert!(g.contains("[orig_ducked][voice_g]amix=inputs=2"));
}

#[test]
fn filter_graph_skips_sidechaincompress_when_ducking_off() {
    let voices = [voice(1, 0.0, "k1", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();
    let g = build_filter_graph(
        &refs,
        &MixSettings {
            ducking_enabled: false,
            ..MixSettings::default()
        },
    );
    assert!(!g.contains("sidechaincompress"));
    // With ducking off we still finish with the two-input amix.
    assert!(g.contains("[orig_g][voice_g]amix=inputs=2"));
}

#[test]
fn filter_graph_with_no_voices_produces_original_only() {
    let g = build_filter_graph(&[], &MixSettings::default());
    // Must terminate on the [mix] label so `-map [mix]` still resolves.
    assert!(g.ends_with("[mix]"));
    assert!(!g.contains("adelay"));
    assert!(!g.contains("sidechaincompress"));
}

// -----------------------------------------------------------------
// Output ceiling
// -----------------------------------------------------------------

/// Summing the ducked original with the voice-over overshot full scale on
/// loud passages and the WAV clipped — measurably, thousands of samples
/// pinned at peak. Every path has to end in the limiter.
#[test]
fn filter_graph_limits_the_summed_output() {
    let voices = [voice(1, 0.0, "k1", false), voice(2, 4.0, "k2", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();
    for graph in [
        build_filter_graph(&refs, &MixSettings::default()),
        build_filter_graph(&[], &MixSettings::default()),
    ] {
        assert!(graph.contains("alimiter"), "graph was: {graph}");
        assert!(
            graph.ends_with("[mix]"),
            "limiter must be the last node feeding -map [mix]",
        );
        // Auto-level would drag quiet stretches up to the ceiling and
        // cancel out the ducking.
        assert!(graph.contains("level=disabled"), "graph was: {graph}");
    }
}

#[test]
fn limiter_leaves_headroom_below_full_scale() {
    let g = build_filter_graph(&[], &MixSettings::default());
    let limit: f32 = g
        .split("limit=")
        .nth(1)
        .and_then(|rest| rest.split(':').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no limit= in graph: {g}"));
    assert!(
        limit > 0.0 && limit < 1.0,
        "limit={limit} must sit under full scale",
    );
}

// -----------------------------------------------------------------
// Defaults and migration
// -----------------------------------------------------------------

/// The voice-over is the reason the file exists; the source track is
/// background. Defaults that put them near parity make the foreign
/// dialogue win, since it's the one matching the picture.
#[test]
fn defaults_keep_the_original_well_under_the_voice() {
    let s = MixSettings::default();
    assert!(
        s.original_volume <= 0.3,
        "original at {} is loud enough to fight the voice",
        s.original_volume,
    );
    assert!(s.voice_volume >= 1.0);
    assert!(s.ducking_enabled);
    assert!(
        s.ducking_depth_db >= 18.0,
        "ducking of {} dB barely moves the original",
        s.ducking_depth_db,
    );
}

#[test]
fn migration_adopts_the_new_balance_for_untouched_v1_projects() {
    let mut old = MixManifest::empty(MixSettings::default());
    old.version = 1;
    old.settings.original_volume = 0.70;
    old.settings.ducking_depth_db = 12.0;

    let m = old.migrated();
    assert_eq!(m.version, MIX_CACHE_SCHEMA_VERSION);
    assert_eq!(
        m.settings.original_volume,
        MixSettings::default().original_volume
    );
    assert_eq!(
        m.settings.ducking_depth_db,
        MixSettings::default().ducking_depth_db,
    );
}

/// A user who deliberately moved a slider shouldn't have it yanked back by
/// an upgrade.
#[test]
fn migration_preserves_deliberately_chosen_values() {
    let mut old = MixManifest::empty(MixSettings::default());
    old.version = 1;
    old.settings.original_volume = 1.4;
    old.settings.ducking_depth_db = 3.0;

    let m = old.migrated();
    assert_eq!(m.settings.original_volume, 1.4);
    assert_eq!(m.settings.ducking_depth_db, 3.0);
}

/// Migration runs on every load, so it has to be a no-op once applied —
/// otherwise re-picking a v1 value would silently flip it again.
#[test]
fn migration_leaves_current_manifests_alone() {
    let mut m = MixManifest::empty(MixSettings::default());
    m.settings.original_volume = 0.70;
    m.settings.ducking_depth_db = 12.0;
    let out = m.clone().migrated();
    assert_eq!(out, m);
}

#[test]
fn filter_graph_single_voice_skips_intermediate_amix() {
    let voices = [voice(1, 0.0, "k1", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();
    let g = build_filter_graph(&refs, &MixSettings::default());
    // With N=1 we shouldn't emit an amix that only mixes one voice.
    assert!(!g.contains("[v1]amix=inputs=1"));
    // The single voice feeds volume then compressor directly.
    assert!(g.contains("[v1]volume="));
}

#[test]
fn build_mix_command_carries_progress_flag() {
    let src = PathBuf::from("/movies/movie.mkv");
    let out = PathBuf::from("/proj/audio/mixed_vi.wav");
    let voices = [voice(1, 0.0, "k1", false)];
    let cmd = build_mix_command(&src, &voices, &MixSettings::default(), &out);
    // -progress pipe:1 must be present so the service can parse frames.
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-progress pipe:1"));
    assert!(joined.contains("-shortest"));
    assert!(cmd.output.contains("mixed_vi.wav"));
}

#[test]
fn build_mix_command_filters_empty_voice_inputs() {
    let src = PathBuf::from("/movies/movie.mkv");
    let out = PathBuf::from("/proj/audio/mixed_vi.wav");
    let voices = [
        voice(1, 0.0, "k1", true), // empty — must be skipped
        voice(2, 5.0, "k2", false),
    ];
    let cmd = build_mix_command(&src, &voices, &MixSettings::default(), &out);
    // Only one -i for the source and one for the non-empty voice.
    let dash_i = cmd.args.iter().filter(|a| a.as_str() == "-i").count();
    assert_eq!(dash_i, 2, "empty voice input must be filtered out");
}

#[test]
fn duck_ratio_is_monotonic() {
    let r0 = duck_ratio_from_depth_db(0.0);
    let r10 = duck_ratio_from_depth_db(10.0);
    let r30 = duck_ratio_from_depth_db(30.0);
    assert!(r0 < r10);
    assert!(r10 < r30);
    // Both endpoints must sit in FFmpeg's ratio range.
    assert!(r0 >= 1.0 && r30 <= 20.0);
}

/// FFmpeg rejects the whole graph if any `sidechaincompress` parameter
/// falls outside its documented range — we shipped `makeup=0` and mixing
/// died with "Value 0.000000 for parameter 'makeup' out of range [1-64]".
/// Assert every knob we emit stays inside the ranges FFmpeg accepts, at
/// the extremes of what the UI can produce.
#[test]
fn sidechaincompress_params_stay_in_ffmpeg_ranges() {
    let voices = [voice(1, 0.0, "k1", false)];
    let refs: Vec<&MixVoiceInput> = voices.iter().collect();

    for (depth, thresh, attack, release) in [
        (0.0, 0.0, 1.0, 10.0),
        (30.0, -60.0, 500.0, 5000.0),
        (12.0, -24.0, 20.0, 300.0),
    ] {
        let settings = MixSettings {
            ducking_enabled: true,
            ducking_depth_db: depth,
            ducking_threshold_db: thresh,
            ducking_attack_ms: attack,
            ducking_release_ms: release,
            ..MixSettings::default()
        }
        .normalised();
        let g = build_filter_graph(&refs, &settings);

        let node = g
            .split(';')
            .find(|part| part.contains("sidechaincompress"))
            .expect("ducking graph must contain the compressor");

        // The first segment is `[orig_g][voice_g]sidechaincompress=threshold=…`,
        // so match `name=` anywhere and read the number that follows.
        let param = |name: &str| -> f32 {
            let key = format!("{name}=");
            let at = node
                .find(&key)
                .unwrap_or_else(|| panic!("missing {name} in {node}"))
                + key.len();
            let rest = &node[at..];
            let end = rest
                .find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end]
                .parse()
                .unwrap_or_else(|e| panic!("unparseable {name} in {node}: {e}"))
        };

        // Ranges per FFmpeg's sidechaincompress documentation.
        let makeup = param("makeup");
        assert!(
            (1.0..=64.0).contains(&makeup),
            "makeup {makeup} out of [1, 64] for depth={depth}"
        );
        let ratio = param("ratio");
        assert!(
            (1.0..=20.0).contains(&ratio),
            "ratio {ratio} out of [1, 20] for depth={depth}"
        );
        let threshold = param("threshold");
        assert!(
            (0.000976563..=1.0).contains(&threshold),
            "threshold {threshold} out of range for thresh_db={thresh}"
        );
        let attack_v = param("attack");
        assert!(
            (0.01..=2000.0).contains(&attack_v),
            "attack {attack_v} out of [0.01, 2000]"
        );
        let release_v = param("release");
        assert!(
            (0.01..=9000.0).contains(&release_v),
            "release {release_v} out of [0.01, 9000]"
        );
    }
}

// -----------------------------------------------------------------
// Voice input collection + summary
// -----------------------------------------------------------------

#[test]
fn collect_voice_inputs_marks_empty_segments() {
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0), subtitle(2, 3.0, 5.0)]);
    let sync = sync_manifest(vec![
        sync_entry(1, 0.0, 2.0, SyncStatus::Fits),
        sync_entry(2, 3.0, 5.0, SyncStatus::Empty),
    ]);
    let inputs = collect_voice_inputs(&doc, &sync);
    assert_eq!(inputs.len(), 2);
    assert!(!inputs[0].is_empty);
    assert!(inputs[1].is_empty, "empty status → is_empty=true");
}

#[test]
fn collect_voice_inputs_skips_subtitles_without_sync_entry() {
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0), subtitle(2, 3.0, 5.0)]);
    let sync = sync_manifest(vec![sync_entry(1, 0.0, 2.0, SyncStatus::Fits)]);
    let inputs = collect_voice_inputs(&doc, &sync);
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].segment_id, 1);
}

#[test]
fn summary_reports_missing_when_manifest_absent() {
    let root = PathBuf::from("/nowhere/does/not/exist");
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0)]);
    let sync = sync_manifest(vec![sync_entry(1, 0.0, 2.0, SyncStatus::Fits)]);
    let summary = build_summary(
        &root,
        &doc,
        Some(&sync),
        None,
        &MixSettings::default(),
        Some(&fingerprint()),
    );
    assert!(matches!(summary.status, MixStatus::Missing));
    assert!(summary.needs_generate);
    assert_eq!(summary.voice_segment_count, 1);
    assert_eq!(summary.subtitle_count, 1);
}

#[test]
fn summary_flags_missing_when_no_sync_manifest() {
    let root = PathBuf::from("/nowhere");
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0)]);
    let summary = build_summary(
        &root,
        &doc,
        None,
        None,
        &MixSettings::default(),
        Some(&fingerprint()),
    );
    assert!(matches!(summary.status, MixStatus::Missing));
    assert!(summary.warning.is_some());
    assert!(summary.needs_generate);
}

// -----------------------------------------------------------------
// Dirty flag mask
// -----------------------------------------------------------------

#[test]
fn only_mix_mask_isolates_the_mix_bit() {
    let mask = DirtyFlags::only_mix();
    assert!(!mask.tts);
    assert!(!mask.sync);
    assert!(mask.mix);
    assert!(!mask.render);
}

#[test]
fn clearing_only_mix_leaves_other_flags() {
    let mut d = DirtyFlags {
        tts: true,
        sync: true,
        mix: true,
        render: true,
    };
    d.clear_where(DirtyFlags::only_mix());
    assert!(d.tts);
    assert!(d.sync);
    assert!(!d.mix);
    assert!(d.render);
}
