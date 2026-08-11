//! Real FFmpeg integration tests.
//!
//! These synthesise a tiny video with `ffmpeg -f lavfi`, then run the
//! detection / probe / extraction pipeline end-to-end. They are skipped
//! automatically when `ffmpeg` is not available on `PATH` so CI
//! environments without FFmpeg still pass `cargo test`.

use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::sync::{Arc, Mutex};

use local_movie_translator_lib::ffmpeg::detection::{FfmpegPathOverride, FfmpegService};
use local_movie_translator_lib::ffmpeg::extract::{
    run_extraction, AudioExtractParams, CancelToken, ProgressFn,
};
use local_movie_translator_lib::ffmpeg::probe::probe_video;

fn ffmpeg_available() -> bool {
    which("ffmpeg").is_some() && which("ffprobe").is_some()
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lmt-e2e-{}-{}-{}",
        tag,
        std::process::id(),
        rand_hex()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{n:x}")
}

/// 2-second silent test movie: black video + silent stereo audio.
fn make_synthetic_mp4(dir: &Path) -> PathBuf {
    let out = dir.join("clip.mp4");
    let status = StdCommand::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-t",
            "2",
            "-c:a",
            "aac",
        ])
        .arg(&out)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "synthetic ffmpeg failed");
    out
}

#[tokio::test]
async fn detect_probe_extract_end_to_end() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let dir = tempdir("roundtrip");
    let clip = make_synthetic_mp4(&dir);

    let svc = FfmpegService::detect(FfmpegPathOverride::default())
        .await
        .expect("detection");
    assert!(!svc.version().is_empty(), "version should be non-empty");

    let meta = probe_video(svc.ffprobe(), &clip).await.expect("probe");
    assert_eq!(meta.video_stream_count, 1);
    assert_eq!(meta.audio_stream_count, 1);
    assert!(meta.duration_secs > 1.0);
    assert!(meta.duration_secs < 3.0);

    let out = dir.join("audio").join("original.wav");
    let params = AudioExtractParams::whisper_default();
    let progress_hits = Arc::new(Mutex::new(0u32));
    let hits_clone = progress_hits.clone();
    let on_progress: ProgressFn = Arc::new(move |_f: f32| {
        *hits_clone.lock().unwrap() += 1;
    });

    let outcome = run_extraction(
        svc.ffmpeg(),
        &clip,
        &out,
        &params,
        meta.duration_secs,
        on_progress,
        CancelToken::new(),
    )
    .await
    .expect("extraction");

    assert!(out.exists(), "output wav missing");
    assert!(outcome.output_size_bytes > 0);
    assert!(
        *progress_hits.lock().unwrap() > 0,
        "at least one progress tick expected"
    );
}

#[tokio::test]
async fn cancellation_terminates_ffmpeg_and_deletes_partial() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let dir = tempdir("cancel");
    // Pre-cancelled token: the runner should honour it and never
    // produce a valid output file. This tests the cancellation *path*
    // without racing a real timer against a fast-encoding synthetic
    // clip.
    let clip = dir.join("clip.mp4");
    let status = StdCommand::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5:sample_rate=48000",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-t",
            "5",
            "-c:a",
            "aac",
        ])
        .arg(&clip)
        .status()
        .unwrap();
    assert!(status.success());

    let svc = FfmpegService::detect(FfmpegPathOverride::default())
        .await
        .unwrap();
    let out = dir.join("audio").join("original.wav");
    let cancel = CancelToken::new();
    cancel.cancel(); // pre-cancelled

    let on_progress: ProgressFn = Arc::new(|_| {});
    let err = run_extraction(
        svc.ffmpeg(),
        &clip,
        &out,
        &AudioExtractParams::whisper_default(),
        5.0,
        on_progress,
        cancel,
    )
    .await
    .unwrap_err();

    assert!(matches!(
        err,
        local_movie_translator_lib::ffmpeg::FfmpegError::Cancelled
    ));
    assert!(!out.exists(), "partial output should be cleaned up");
}
