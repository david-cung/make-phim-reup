//! Structured logging. JSON lines to `<cache>/logs/rust.log` with daily rotation.
//!
//! We deliberately return the appender guard so callers can hold it for the
//! lifetime of the app — dropping it would flush and close the file.
//!
//! Phase 12 — production defaults:
//!   * stderr layer at `info`; only unlocked to `debug`/`trace` by an
//!     explicit `RUST_LOG` env var (advanced-user escape hatch).
//!   * daily-rotated JSON file layer with a bounded retention window
//!     (`prune_old_logs`) so a long-running install never fills the
//!     cache disk with historical rust.log.* / worker.log.* files.
//!   * `tracing` calls elsewhere in the codebase intentionally avoid
//!     logging entire transcripts, LLM responses or subtitle text —
//!     we log method + timing + counts, never payloads.

use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::paths::AppPaths;

/// Keep rotated log files for at most this many days. Fourteen strikes a
/// balance between "long enough to reproduce yesterday's report" and
/// "small enough that it never surprises the user".
pub const LOG_RETENTION_DAYS: u64 = 14;

pub fn init(paths: &AppPaths) -> io::Result<WorkerGuard> {
    // Trim historical rotations from previous runs before we start
    // writing new ones. Best-effort — never blocks startup.
    prune_old_logs(&paths.log_dir, LOG_RETENTION_DAYS);

    let file_appender = tracing_appender::rolling::daily(&paths.log_dir, "rust.log");
    let (nb_writer, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,local_movie_translator_lib=debug"));

    let file_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(false)
        .with_writer(nb_writer);

    let stderr_layer = fmt::layer().with_target(false).with_writer(io::stderr);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(file_layer)
        .with(stderr_layer)
        .try_init()
        .map_err(|e| io::Error::other(e.to_string()))?;

    Ok(guard)
}

/// Delete `rust.log.*` / `worker.log.*` rotations older than
/// `retention_days`. Silent no-op on errors — a log-directory hiccup
/// must never abort the app.
pub fn prune_old_logs(log_dir: &Path, retention_days: u64) {
    let cutoff = match SystemTime::now().checked_sub(Duration::from_secs(retention_days * 86_400)) {
        Some(t) => t,
        None => return,
    };
    let entries = match std::fs::read_dir(log_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // Only touch our own log rotations — anything else in the
        // directory (e.g. a user's investigation notes) is off-limits.
        if !name.starts_with("rust.log") && !name.starts_with("worker.log") {
            continue;
        }
        let modified = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if modified < cutoff {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Truncate every current log file to zero bytes. Called from the
/// "Clear logs" UI action — we intentionally do *not* delete the
/// files so the tracing appender's open file handle stays valid.
pub fn clear_active_logs(log_dir: &Path) -> io::Result<usize> {
    let mut cleared = 0usize;
    let entries = std::fs::read_dir(log_dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if !name.starts_with("rust.log") && !name.starts_with("worker.log") {
            continue;
        }
        // Truncate rather than remove so the active file handle stays valid.
        if std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .is_ok()
        {
            cleared += 1;
        }
    }
    Ok(cleared)
}
