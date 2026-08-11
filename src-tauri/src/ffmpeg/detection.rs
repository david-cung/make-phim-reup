//! Locate `ffmpeg` / `ffprobe` on the host and verify they respond.
//!
//! Discovery order (first match wins):
//!   1. Explicit override paths.
//!   2. Bundled binaries next to the app executable (Phase 10 packaging).
//!   3. First `ffmpeg` / `ffprobe` found on `PATH`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;
use tokio::process::Command;

use super::errors::FfmpegError;

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FfmpegAvailability {
    pub available: bool,
    pub ffmpeg_path: Option<String>,
    pub ffprobe_path: Option<String>,
    pub version: Option<String>,
    /// Error message when `available` is false. Never a stack trace.
    pub error: Option<String>,
}

/// Ready-to-use handle for spawning FFmpeg / FFprobe subprocesses.
///
/// Constructed once at startup and swapped atomically whenever the user
/// changes the configured path — long-running jobs keep the `Arc` they
/// already have.
#[derive(Debug)]
pub struct FfmpegService {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    version: String,
}

impl FfmpegService {
    pub fn ffmpeg(&self) -> &Path {
        &self.ffmpeg
    }

    pub fn ffprobe(&self) -> &Path {
        &self.ffprobe
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn availability(&self) -> FfmpegAvailability {
        FfmpegAvailability {
            available: true,
            ffmpeg_path: Some(self.ffmpeg.display().to_string()),
            ffprobe_path: Some(self.ffprobe.display().to_string()),
            version: Some(self.version.clone()),
            error: None,
        }
    }

    /// Attempt to locate a working FFmpeg + FFprobe pair.
    pub async fn detect(override_paths: FfmpegPathOverride) -> Result<Self, FfmpegError> {
        let (ffmpeg, ffprobe) = resolve_paths(&override_paths)?;
        verify_executable(&ffmpeg, "ffmpeg").await?;
        verify_executable(&ffprobe, "ffprobe").await?;
        let version = probe_version(&ffmpeg).await?;
        tracing::info!(
            ffmpeg = %ffmpeg.display(),
            ffprobe = %ffprobe.display(),
            version,
            "ffmpeg service ready"
        );
        Ok(Self {
            ffmpeg,
            ffprobe,
            version,
        })
    }

    /// Non-fatal variant of [`detect`]. Always returns an
    /// [`FfmpegAvailability`] describing the outcome so the UI can render
    /// a helpful message; callers that need the actual service should use
    /// [`detect`].
    pub async fn probe(override_paths: FfmpegPathOverride) -> FfmpegAvailability {
        match Self::detect(override_paths).await {
            Ok(svc) => svc.availability(),
            Err(err) => {
                tracing::warn!(%err, "ffmpeg not available");
                FfmpegAvailability {
                    available: false,
                    ffmpeg_path: None,
                    ffprobe_path: None,
                    version: None,
                    error: Some(err.to_string()),
                }
            }
        }
    }
}

/// Configured overrides from user settings. Either may be `None`.
#[derive(Debug, Clone, Default)]
pub struct FfmpegPathOverride {
    pub ffmpeg: Option<PathBuf>,
    pub ffprobe: Option<PathBuf>,
    /// If set, we also probe this directory (used for a Phase 10 sidecar
    /// bundle: `<resource_dir>/bin/ffmpeg`).
    pub bundled_bin_dir: Option<PathBuf>,
}

/// Shared holder that supports hot-swapping the resolved service when
/// settings change. Callers borrow `Arc<FfmpegService>` snapshots.
#[derive(Debug, Default)]
pub struct FfmpegHandle {
    inner: RwLock<Option<Arc<FfmpegService>>>,
    last_availability: RwLock<FfmpegAvailability>,
}

impl FfmpegHandle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(None),
            last_availability: RwLock::new(FfmpegAvailability {
                available: false,
                ffmpeg_path: None,
                ffprobe_path: None,
                version: None,
                error: Some("FFmpeg has not been probed yet".into()),
            }),
        })
    }

    pub fn get(&self) -> Option<Arc<FfmpegService>> {
        self.inner.read().clone()
    }

    pub fn availability(&self) -> FfmpegAvailability {
        self.last_availability.read().clone()
    }

    pub async fn refresh(&self, override_paths: FfmpegPathOverride) -> FfmpegAvailability {
        match FfmpegService::detect(override_paths).await {
            Ok(svc) => {
                let av = svc.availability();
                *self.inner.write() = Some(Arc::new(svc));
                *self.last_availability.write() = av.clone();
                av
            }
            Err(err) => {
                let av = FfmpegAvailability {
                    available: false,
                    ffmpeg_path: None,
                    ffprobe_path: None,
                    version: None,
                    error: Some(err.to_string()),
                };
                *self.inner.write() = None;
                *self.last_availability.write() = av.clone();
                av
            }
        }
    }
}

// ------ discovery helpers ------

fn resolve_paths(ov: &FfmpegPathOverride) -> Result<(PathBuf, PathBuf), FfmpegError> {
    // Prefer explicit overrides — if only one is set, derive the sibling.
    if let (Some(ffm), Some(ffp)) = (&ov.ffmpeg, &ov.ffprobe) {
        return Ok((ffm.clone(), ffp.clone()));
    }
    if let Some(ffm) = &ov.ffmpeg {
        let ffp = sibling(ffm, "ffprobe");
        return Ok((ffm.clone(), ffp));
    }
    if let Some(ffp) = &ov.ffprobe {
        let ffm = sibling(ffp, "ffmpeg");
        return Ok((ffm, ffp.clone()));
    }

    // Bundled sidecar (`bin/ffmpeg[.exe]`).
    if let Some(dir) = &ov.bundled_bin_dir {
        let ffm = dir.join(exe_name("ffmpeg"));
        let ffp = dir.join(exe_name("ffprobe"));
        if ffm.is_file() && ffp.is_file() {
            return Ok((ffm, ffp));
        }
    }

    // PATH lookup.
    let ffmpeg = which_on_path("ffmpeg").ok_or(FfmpegError::NotFound {
        path: PathBuf::from("ffmpeg"),
    })?;
    let ffprobe = which_on_path("ffprobe").ok_or(FfmpegError::ProbeNotFound {
        path: PathBuf::from("ffprobe"),
    })?;
    Ok((ffmpeg, ffprobe))
}

fn sibling(path: &Path, name: &str) -> PathBuf {
    let name = exe_name(name);
    path.parent()
        .map(|p| p.join(&name))
        .unwrap_or_else(|| PathBuf::from(name))
}

fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn which_on_path(base: &str) -> Option<PathBuf> {
    let name = exe_name(base);
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(&name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

async fn verify_executable(path: &Path, kind: &'static str) -> Result<(), FfmpegError> {
    if !path.exists() {
        return Err(match kind {
            "ffprobe" => FfmpegError::ProbeNotFound {
                path: path.to_path_buf(),
            },
            _ => FfmpegError::NotFound {
                path: path.to_path_buf(),
            },
        });
    }
    let out = Command::new(path)
        .arg("-hide_banner")
        .arg("-version")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| FfmpegError::io("running -version", source))?;
    if !out.status.success() {
        return Err(FfmpegError::VersionUnknown {
            details: format!(
                "`{}` exited with {:?}; stderr: {}",
                path.display(),
                out.status.code(),
                String::from_utf8_lossy(&out.stderr).trim(),
            ),
        });
    }
    Ok(())
}

async fn probe_version(ffmpeg: &Path) -> Result<String, FfmpegError> {
    let out = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-version")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| FfmpegError::io("running -version", source))?;
    let first = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    // e.g. "ffmpeg version 8.1.1 Copyright …"
    let ver = first
        .split_whitespace()
        .nth(2)
        .map(str::to_string)
        .unwrap_or(first);
    if ver.is_empty() {
        return Err(FfmpegError::VersionUnknown {
            details: "empty stdout".into(),
        });
    }
    Ok(ver)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_swaps_basename() {
        assert_eq!(
            sibling(Path::new("/opt/bin/ffmpeg"), "ffprobe"),
            PathBuf::from(if cfg!(windows) {
                "/opt/bin/ffprobe.exe"
            } else {
                "/opt/bin/ffprobe"
            })
        );
    }

    #[test]
    fn override_pair_wins() {
        let ov = FfmpegPathOverride {
            ffmpeg: Some(PathBuf::from("/tmp/a")),
            ffprobe: Some(PathBuf::from("/tmp/b")),
            bundled_bin_dir: None,
        };
        let (a, b) = resolve_paths(&ov).unwrap();
        assert_eq!(a, PathBuf::from("/tmp/a"));
        assert_eq!(b, PathBuf::from("/tmp/b"));
    }

    #[test]
    fn single_override_derives_sibling() {
        let ov = FfmpegPathOverride {
            ffmpeg: Some(PathBuf::from("/tmp/ffm")),
            ffprobe: None,
            bundled_bin_dir: None,
        };
        let (_a, b) = resolve_paths(&ov).unwrap();
        assert!(b.ends_with(if cfg!(windows) {
            "ffprobe.exe"
        } else {
            "ffprobe"
        }));
    }
}
