//! Pure-Rust unit tests for the render module.
//!
//! No FFmpeg, no worker. Cover the invariants the frontend depends on:
//! cache-key stability, argv shape (with & without burn-in, mp4 vs
//! mkv), summary contract, and dirty-flag masking.

use std::path::{Path, PathBuf};

use chrono::Utc;

use crate::media::SourceFingerprint;
use crate::mix::MixEntry;
use crate::mix::MixSettings;
use crate::subtitles::models::DerivedFrom;
use crate::subtitles::{DirtyFlags, SubtitleDoc, SubtitleSegment};

use super::cache::{default_subtitle_output_path, subtitle_sidecar_path};
use super::ffmpeg_cmd::{build_render_command, build_subtitles_filter};
use super::models::{
    build_render_cache_key, AudioCodec, OutputFormat, RenderManifest, RenderSettings, RenderStatus,
    SubtitleMode, VideoCodec, RENDER_CACHE_SCHEMA_VERSION,
};
use super::service::{build_summary, default_settings_for};

// -----------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------

fn fingerprint() -> SourceFingerprint {
    SourceFingerprint {
        hash: "sha256:cafef00d".into(),
        size_bytes: 5_000_000,
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
        voice_id: None,
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

fn mix_entry(cache_key: &str) -> MixEntry {
    MixEntry {
        cache_key: cache_key.into(),
        source_fingerprint: fingerprint(),
        file: "audio/mixed_vi.wav".into(),
        duration_secs: 3600.0,
        sample_rate: 44_100,
        channels: 2,
        size_bytes: 1_234_567,
        voice_segment_count: 42,
        subtitle_count: 45,
        settings: MixSettings::default(),
        generated_at: Utc::now(),
    }
}

// -----------------------------------------------------------------
// Cache key stability
// -----------------------------------------------------------------

#[test]
fn cache_key_versioned() {
    assert_eq!(RENDER_CACHE_SCHEMA_VERSION, 2);
    let key = build_render_cache_key(&fingerprint(), "sha256:m1", &RenderSettings::default());
    assert!(key.starts_with("sha256:"));
    assert!(key.len() > 20);
}

#[test]
fn cache_key_changes_with_source_fingerprint() {
    let s = RenderSettings::default();
    let a = build_render_cache_key(&fingerprint(), "sha256:m1", &s);
    let mut fp2 = fingerprint();
    fp2.hash = "sha256:beadfeed".into();
    let b = build_render_cache_key(&fp2, "sha256:m1", &s);
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_mix_key() {
    let s = RenderSettings::default();
    let a = build_render_cache_key(&fingerprint(), "sha256:m1", &s);
    let b = build_render_cache_key(&fingerprint(), "sha256:m2", &s);
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_subtitle_mode() {
    let a = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            subtitle_mode: SubtitleMode::External,
            ..RenderSettings::default()
        },
    );
    let b = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            subtitle_mode: SubtitleMode::Burned,
            ..RenderSettings::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_output_format() {
    let a = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            output_format: OutputFormat::Mp4,
            ..RenderSettings::default()
        },
    );
    let b = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            output_format: OutputFormat::Mkv,
            ..RenderSettings::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn cache_key_changes_with_audio_codec() {
    let a = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            audio_codec: AudioCodec {
                codec: "aac".into(),
                bitrate: Some("192k".into()),
            },
            ..RenderSettings::default()
        },
    );
    let b = build_render_cache_key(
        &fingerprint(),
        "sha256:m1",
        &RenderSettings {
            audio_codec: AudioCodec {
                codec: "libopus".into(),
                bitrate: Some("192k".into()),
            },
            ..RenderSettings::default()
        },
    );
    assert_ne!(a, b);
}

#[test]
fn burned_mode_promotes_video_copy_to_reencode() {
    let s = RenderSettings {
        subtitle_mode: SubtitleMode::Burned,
        video_codec: VideoCodec::Copy,
        ..RenderSettings::default()
    }
    .normalised();
    assert!(matches!(s.video_codec, VideoCodec::Reencode { .. }));
}

#[test]
fn cache_key_matches_after_burn_promotion() {
    // Two settings that differ only in the fact one explicitly asks
    // for libx264 while the other lets `normalised()` promote copy →
    // libx264 because burn is on — should collapse to the same key.
    let a = RenderSettings {
        subtitle_mode: SubtitleMode::Burned,
        video_codec: VideoCodec::Copy,
        ..RenderSettings::default()
    };
    let b = RenderSettings {
        subtitle_mode: SubtitleMode::Burned,
        video_codec: VideoCodec::Reencode {
            codec: "libx264".into(),
        },
        ..RenderSettings::default()
    };
    assert_eq!(
        build_render_cache_key(&fingerprint(), "sha256:m1", &a),
        build_render_cache_key(&fingerprint(), "sha256:m1", &b),
    );
}

// -----------------------------------------------------------------
// FFmpeg argv shape
// -----------------------------------------------------------------

#[test]
fn build_render_command_carries_progress_flag() {
    let src = PathBuf::from("/movies/movie.mkv");
    let mix = PathBuf::from("/proj/audio/mixed_vi.wav");
    let out = PathBuf::from("/proj/output/movie_vi.mp4");
    let cmd = build_render_command(&src, &mix, None, &RenderSettings::default(), &out);
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-progress pipe:1"));
    assert!(joined.contains("-shortest"));
    // Two inputs: source video + mixed audio.
    let dash_i = cmd.args.iter().filter(|a| a.as_str() == "-i").count();
    assert_eq!(dash_i, 2);
    assert!(cmd.output.ends_with("movie_vi.mp4"));
}

#[test]
fn default_external_mode_copies_video_and_reencodes_audio() {
    let src = PathBuf::from("/m/movie.mkv");
    let mix = PathBuf::from("/p/audio/mixed_vi.wav");
    let out = PathBuf::from("/p/output/movie_vi.mp4");
    let cmd = build_render_command(
        &src,
        &mix,
        None,
        &RenderSettings {
            subtitle_mode: SubtitleMode::External,
            ..RenderSettings::default()
        },
        &out,
    );
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-c:v copy"), "argv={joined}");
    assert!(joined.contains("-c:a aac"));
    assert!(joined.contains("-b:a 192k"));
    // Original audio is dropped: we map 0:v:0 and 1:a:0 only.
    assert!(joined.contains("-map 0:v:0"));
    assert!(joined.contains("-map 1:a:0"));
    assert!(!joined.contains("-vf"));
}

#[test]
fn burned_mode_emits_subtitles_filter_and_reencodes_video() {
    let src = PathBuf::from("/m/movie.mkv");
    let mix = PathBuf::from("/p/audio/mixed_vi.wav");
    let out = PathBuf::from("/p/output/movie_vi.mp4");
    let subs = PathBuf::from("/p/output/movie_vi.srt");
    let cmd = build_render_command(
        &src,
        &mix,
        Some(&subs),
        &RenderSettings {
            subtitle_mode: SubtitleMode::Burned,
            video_codec: VideoCodec::Copy, // will be promoted
            ..RenderSettings::default()
        },
        &out,
    );
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-vf"));
    assert!(joined.contains("subtitles=") && joined.contains("movie_vi.srt"));
    assert!(joined.contains("-c:v libx264"));
    assert!(!joined.contains("-c:v copy"));
    assert!(joined.contains("-preset medium"));
    assert!(joined.contains("-crf 20"));
}

#[test]
fn none_mode_copies_video_and_omits_subtitles() {
    let src = PathBuf::from("/m/movie.mkv");
    let mix = PathBuf::from("/p/audio/mixed_vi.wav");
    let out = PathBuf::from("/p/output/movie_vi.mp4");
    let cmd = build_render_command(
        &src,
        &mix,
        None,
        &RenderSettings {
            subtitle_mode: SubtitleMode::None,
            ..RenderSettings::default()
        },
        &out,
    );
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-c:v copy"));
    assert!(!joined.contains("-vf"));
    assert!(!joined.contains("subtitles="));
}

#[test]
fn mp4_output_includes_faststart_flag() {
    let src = PathBuf::from("/m/movie.mkv");
    let mix = PathBuf::from("/p/audio/mixed_vi.wav");
    let out = PathBuf::from("/p/output/movie_vi.mp4");
    let cmd = build_render_command(
        &src,
        &mix,
        None,
        &RenderSettings {
            output_format: OutputFormat::Mp4,
            ..RenderSettings::default()
        },
        &out,
    );
    let joined = cmd.args.join(" ");
    assert!(joined.contains("-movflags +faststart"));
}

#[test]
fn mkv_output_omits_faststart_flag() {
    let src = PathBuf::from("/m/movie.mkv");
    let mix = PathBuf::from("/p/audio/mixed_vi.wav");
    let out = PathBuf::from("/p/output/movie_vi.mkv");
    let cmd = build_render_command(
        &src,
        &mix,
        None,
        &RenderSettings {
            output_format: OutputFormat::Mkv,
            ..RenderSettings::default()
        },
        &out,
    );
    let joined = cmd.args.join(" ");
    assert!(!joined.contains("faststart"));
}

// -----------------------------------------------------------------
// Subtitles filter escaping
// -----------------------------------------------------------------

// -----------------------------------------------------------------
// Defaults and migration
// -----------------------------------------------------------------

/// A sidecar the viewer never loads reads as "no subtitles", which is what
/// the old default produced. Burn them in unless asked otherwise.
#[test]
fn default_subtitle_mode_is_visible_without_player_help() {
    assert_eq!(SubtitleMode::default(), SubtitleMode::Burned);
    assert_eq!(
        RenderSettings::default().subtitle_mode,
        SubtitleMode::Burned,
    );
}

/// Advertising a mode this FFmpeg can't perform just moves the failure to
/// render time, so the default has to bend to the build.
#[test]
fn advertised_default_falls_back_when_burning_is_unavailable() {
    assert_eq!(
        default_settings_for(true).subtitle_mode,
        SubtitleMode::Burned,
    );
    assert_eq!(
        default_settings_for(false).subtitle_mode,
        SubtitleMode::External,
    );
}

#[test]
fn migration_moves_v1_external_projects_to_burned() {
    let mut old = RenderManifest::empty(RenderSettings::default());
    old.version = 1;
    old.settings.subtitle_mode = SubtitleMode::External;

    let m = old.migrated();
    assert_eq!(m.version, RENDER_CACHE_SCHEMA_VERSION);
    assert_eq!(m.settings.subtitle_mode, SubtitleMode::Burned);
}

/// `None` is a choice the old default could never have produced, so it's
/// deliberate and stays put.
#[test]
fn migration_respects_an_explicit_no_subtitles_choice() {
    let mut old = RenderManifest::empty(RenderSettings::default());
    old.version = 1;
    old.settings.subtitle_mode = SubtitleMode::None;

    assert_eq!(old.migrated().settings.subtitle_mode, SubtitleMode::None);
}

/// Migration runs on every load, so picking `External` on a current
/// manifest must survive the next one.
#[test]
fn migration_leaves_current_manifests_alone() {
    let mut m = RenderManifest::empty(RenderSettings::default());
    m.settings.subtitle_mode = SubtitleMode::External;
    assert_eq!(m.clone().migrated(), m);
}

// -----------------------------------------------------------------
// Sidecar placement
// -----------------------------------------------------------------

/// We shipped the sidecar into the project's `output/` folder no matter
/// where the movie went, so rendering to `~/Downloads` left the SRT
/// stranded in Application Support and every player showed no subtitles.
#[test]
fn external_sidecar_lands_next_to_a_custom_output_path() {
    let project = Path::new("/data/projects/abc");
    let out = Path::new("/Users/dc/Downloads/movie_vi.mp4");
    let subs = subtitle_sidecar_path(project, out, SubtitleMode::External);
    assert_eq!(subs, PathBuf::from("/Users/dc/Downloads/movie_vi.srt"));
    assert_eq!(
        subs.parent(),
        out.parent(),
        "player needs them side by side"
    );
}

#[test]
fn external_sidecar_matches_the_movie_stem() {
    let project = Path::new("/data/projects/abc");
    let out = Path::new("/tmp/My Film (1080p).mkv");
    let subs = subtitle_sidecar_path(project, out, SubtitleMode::External);
    assert_eq!(subs, PathBuf::from("/tmp/My Film (1080p).srt"));
}

/// Burned mode only feeds FFmpeg's `subtitles=` filter, so the file is an
/// intermediate and belongs inside the project rather than beside the
/// user's movie.
#[test]
fn burned_sidecar_stays_inside_the_project() {
    let project = Path::new("/data/projects/abc");
    let out = Path::new("/Users/dc/Downloads/movie_vi.mp4");
    assert_eq!(
        subtitle_sidecar_path(project, out, SubtitleMode::Burned),
        default_subtitle_output_path(project),
    );
}

#[test]
fn subtitles_filter_wraps_path_in_single_quotes() {
    let f = build_subtitles_filter(Path::new("/p/output/movie_vi.srt"));
    assert!(f.starts_with("subtitles='"));
    assert!(f.ends_with("'"));
}

#[test]
fn subtitles_filter_escapes_colons_and_quotes_and_commas_and_backslashes() {
    let f = build_subtitles_filter(Path::new(r#"/weird/pa'th, with:chars\test.srt"#));
    // Every metacharacter must be backslash-escaped inside the
    // single-quoted value.
    assert!(f.contains(r"\'"));
    assert!(f.contains(r"\:"));
    assert!(f.contains(r"\,"));
    assert!(f.contains(r"\\"));
}

// -----------------------------------------------------------------
// Summary contract
// -----------------------------------------------------------------

#[test]
fn summary_missing_when_no_manifest() {
    let root = PathBuf::from("/nowhere");
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0)]);
    let mix = mix_entry("sha256:m1");
    let summary = build_summary(
        &root,
        Some(&doc),
        Some(&mix),
        None,
        &RenderSettings::default(),
        Some(&fingerprint()),
    );
    assert!(matches!(summary.status, RenderStatus::Missing));
    assert!(summary.needs_render);
    assert!(summary
        .default_output_absolute
        .ends_with("output/movie_vi.mp4"));
}

#[test]
fn summary_warning_when_no_mix() {
    let root = PathBuf::from("/nowhere");
    let doc = subtitle_doc(vec![subtitle(1, 0.0, 2.0)]);
    let summary = build_summary(
        &root,
        Some(&doc),
        None,
        None,
        &RenderSettings::default(),
        Some(&fingerprint()),
    );
    assert!(matches!(summary.status, RenderStatus::Missing));
    assert!(summary.warning.is_some());
    assert!(summary.needs_render);
}

// -----------------------------------------------------------------
// Dirty flag mask
// -----------------------------------------------------------------

#[test]
fn only_render_mask_isolates_the_render_bit() {
    let mask = DirtyFlags::only_render();
    assert!(!mask.tts);
    assert!(!mask.sync);
    assert!(!mask.mix);
    assert!(mask.render);
}

#[test]
fn clearing_only_render_leaves_other_flags() {
    let mut d = DirtyFlags {
        tts: true,
        sync: true,
        mix: true,
        render: true,
    };
    d.clear_where(DirtyFlags::only_render());
    assert!(d.tts);
    assert!(d.sync);
    assert!(d.mix);
    assert!(!d.render);
}
