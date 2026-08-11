//! Audio extraction runner.
//!
//! Runs `ffmpeg` with `-progress pipe:1` and a cooperative cancellation
//! channel. All process arguments are typed — nothing from the frontend
//! ever gets interpolated into a shell string.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{watch, Notify};
use tokio::time::timeout;

use super::errors::FfmpegError;
use super::progress::{feed_line, ProgressBlock};

/// Extraction parameters. Fixed for the Whisper pipeline but represented
/// as a struct so the cache key can round-trip them and Phase 3 can vary
/// them per Whisper model if we ever need to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AudioExtractParams {
    pub sample_rate: u32,
    pub channels: u16,
    /// FFmpeg codec name (e.g. `pcm_s16le`).
    pub codec: String,
}

impl AudioExtractParams {
    pub fn whisper_default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            codec: "pcm_s16le".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtractionOutcome {
    pub output_path: PathBuf,
    pub duration_secs: f64,
    pub output_size_bytes: u64,
}

/// Build the exact argv the extraction runner will pass to ffmpeg.
/// Extracted for unit testing — no `Command` is spawned here.
pub fn build_extract_command(
    input: &Path,
    output: &Path,
    params: &AudioExtractParams,
) -> Vec<String> {
    vec![
        "-y".into(),       // overwrite existing (we manage the cache above this layer)
        "-nostdin".into(), // no interactive prompt
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-i".into(),
        input.to_string_lossy().into_owned(),
        "-vn".into(), // drop video
        "-sn".into(), // drop subtitles
        "-dn".into(), // drop data streams
        "-map".into(),
        "0:a:0".into(), // pick first audio stream
        "-ac".into(),
        params.channels.to_string(),
        "-ar".into(),
        params.sample_rate.to_string(),
        "-acodec".into(),
        params.codec.clone(),
        "-progress".into(),
        "pipe:1".into(), // machine-readable progress on stdout
        output.to_string_lossy().into_owned(),
    ]
}

/// Progress callback receives a fraction 0..=1 plus a hint like "extract".
pub type ProgressFn = Arc<dyn Fn(f32) + Send + Sync>;

/// Handle used to co-operatively cancel a running extraction. Cloneable
/// so both the runner and the registry can hold one.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    signal: Arc<Notify>,
    cancelled: Arc<parking_lot::Mutex<bool>>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        *self.cancelled.lock() = true;
        self.signal.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        *self.cancelled.lock()
    }

    pub async fn wait(&self) {
        if self.is_cancelled() {
            return;
        }
        self.signal.notified().await;
    }
}

/// Run an extraction to completion. On cancellation the child is asked
/// to terminate gracefully, then killed after a short grace period, and
/// any partial output file is removed.
pub async fn run_extraction(
    ffmpeg: &Path,
    input: &Path,
    output: &Path,
    params: &AudioExtractParams,
    duration_secs: f64,
    on_progress: ProgressFn,
    cancel: CancelToken,
) -> Result<ExtractionOutcome, FfmpegError> {
    if !input.exists() {
        return Err(FfmpegError::InputMissing {
            path: input.to_path_buf(),
        });
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|source| FfmpegError::io("create output dir", source))?;
    }

    let args = build_extract_command(input, output, params);
    tracing::debug!(?args, "spawning ffmpeg");

    let mut cmd = Command::new(ffmpeg);
    cmd.args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn().map_err(|source| match source.kind() {
        std::io::ErrorKind::NotFound => FfmpegError::NotFound {
            path: ffmpeg.to_path_buf(),
        },
        std::io::ErrorKind::PermissionDenied => FfmpegError::PermissionDenied {
            path: output.to_path_buf(),
        },
        _ => FfmpegError::io("spawning ffmpeg", source),
    })?;

    let pid = child.id();

    // stderr collector for post-mortem error messages.
    let stderr = child.stderr.take().expect("stderr was piped");
    let (stderr_tx, stderr_rx) = watch::channel::<String>(String::new());
    let stderr_task = tokio::spawn(collect_stderr(stderr, stderr_tx));

    // stdout progress reader.
    let stdout = child.stdout.take().expect("stdout was piped");
    let on_progress_clone = on_progress.clone();
    let progress_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut cur = ProgressBlock::default();
        while let Ok(Some(line)) = reader.next_line().await {
            if let Some(done) = feed_line(&mut cur, &line) {
                if let Some(frac) = done.fraction(duration_secs) {
                    (on_progress_clone)(frac);
                }
                if done.ended {
                    (on_progress_clone)(1.0);
                }
            }
        }
    });

    // Race the child against the cancel token.
    let wait_res = tokio::select! {
        biased;
        _ = cancel.wait() => {
            terminate_child(&mut child, pid).await;
            let _ = progress_task.await;
            let _ = stderr_task.await;
            cleanup_partial(output);
            return Err(FfmpegError::Cancelled);
        }
        r = child.wait() => r,
    };

    let status = wait_res.map_err(|source| FfmpegError::io("waiting on ffmpeg", source))?;
    let _ = progress_task.await;
    let _ = stderr_task.await;

    if !status.success() {
        cleanup_partial(output);
        let code = status.code().unwrap_or(-1);
        let stderr = stderr_rx.borrow().clone();
        return Err(classify_failure(code, &stderr));
    }

    let meta = std::fs::metadata(output)
        .map_err(|source| FfmpegError::io("stat output after ffmpeg", source))?;
    if meta.len() == 0 {
        cleanup_partial(output);
        return Err(FfmpegError::NoOutput {
            path: output.to_path_buf(),
        });
    }

    Ok(ExtractionOutcome {
        output_path: output.to_path_buf(),
        duration_secs,
        output_size_bytes: meta.len(),
    })
}

async fn collect_stderr(stderr: tokio::process::ChildStderr, tx: watch::Sender<String>) {
    let mut reader = BufReader::new(stderr).lines();
    let mut buf = String::new();
    while let Ok(Some(line)) = reader.next_line().await {
        if buf.len() < 8 * 1024 {
            buf.push_str(&line);
            buf.push('\n');
        }
        tracing::debug!(target: "ffmpeg", "{line}");
    }
    let _ = tx.send(buf);
}

async fn terminate_child(child: &mut Child, pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        // Ask ffmpeg politely; it finalises the current frame and exits.
        unsafe {
            let _ = libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;

    // Give it a short grace period; then kill for real.
    match timeout(Duration::from_millis(2_000), child.wait()).await {
        Ok(_) => {}
        Err(_) => {
            tracing::warn!("ffmpeg did not exit after SIGTERM; killing");
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

fn cleanup_partial(output: &Path) {
    if output.exists() {
        if let Err(err) = std::fs::remove_file(output) {
            tracing::warn!(%err, path = %output.display(), "failed to remove partial output");
        }
    }
}

fn classify_failure(code: i32, stderr: &str) -> FfmpegError {
    let s = stderr.to_ascii_lowercase();
    if s.contains("no such file") {
        FfmpegError::InputMissing {
            path: PathBuf::from("(from ffmpeg)"),
        }
    } else if s.contains("permission denied") {
        FfmpegError::PermissionDenied {
            path: PathBuf::from("(from ffmpeg)"),
        }
    } else if s.contains("invalid data found") || s.contains("could not find codec") {
        FfmpegError::UnsupportedCodec {
            details: shorten(stderr, 300),
        }
    } else if s.contains("does not contain any stream") {
        FfmpegError::NoAudioStream
    } else if s.contains("no space left on device") {
        FfmpegError::DiskSpaceLow {
            required_bytes: 0,
            available_bytes: 0,
        }
    } else {
        FfmpegError::RunFailed {
            code,
            stderr: shorten(stderr, 1_000),
        }
    }
}

fn shorten(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        trimmed.to_string()
    } else {
        let mut out: String = trimmed.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_are_deterministic_and_shell_free() {
        let args = build_extract_command(
            Path::new("/media/movie.mkv"),
            Path::new("/proj/audio/original.wav"),
            &AudioExtractParams::whisper_default(),
        );
        assert!(args.iter().any(|a| a == "-progress"));
        assert!(args.iter().any(|a| a == "pipe:1"));
        assert!(args.iter().any(|a| a == "16000"));
        assert!(args.iter().any(|a| a == "pcm_s16le"));
        assert!(args.iter().any(|a| a == "-nostdin"));
        assert!(args.iter().any(|a| a == "-y"));
        assert!(args.iter().any(|a| a == "-vn"));
        // Input path passed as a single argument — never re-interpreted by a shell.
        assert!(args.iter().any(|a| a == "/media/movie.mkv"));
    }

    #[test]
    fn cancel_token_is_observable() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn classify_permission_denied() {
        let err = classify_failure(1, "Permission denied: /tmp/nope.wav");
        assert!(matches!(err, FfmpegError::PermissionDenied { .. }));
    }

    #[test]
    fn classify_invalid_data() {
        let err = classify_failure(1, "Invalid data found when processing input");
        assert!(matches!(err, FfmpegError::UnsupportedCodec { .. }));
    }

    #[test]
    fn classify_generic() {
        let err = classify_failure(255, "some other message");
        assert!(matches!(err, FfmpegError::RunFailed { code: 255, .. }));
    }

    #[test]
    fn shorten_respects_max() {
        assert_eq!(shorten("hi", 10), "hi");
        let long: String = "a".repeat(20);
        let s = shorten(&long, 5);
        assert_eq!(s.chars().count(), 6); // 5 + ellipsis
    }
}
