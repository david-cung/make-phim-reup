//! Application settings persisted to `<config>/config.json`.
//!
//! Kept intentionally small in Phase 1. Only fields the UI actually surfaces
//! today exist. New fields must be added as `Option<T>` first so old configs
//! keep parsing, or with a Serde `default` attribute.

use std::io;
use std::path::Path;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub offline_mode: bool,
    pub source_language: String,
    pub target_language: String,
    pub max_concurrent_jobs: u32,
    pub log_level: LogLevel,
    #[serde(default)]
    pub whisper_model: Option<String>,
    #[serde(default)]
    pub translation_model: Option<String>,
    #[serde(default)]
    pub tts_voice: Option<String>,
    /// User-provided override for the ffmpeg binary. When `Some`,
    /// takes precedence over the bundled and `PATH` lookups.
    #[serde(default)]
    pub ffmpeg_path: Option<String>,
    /// Same for ffprobe. If only `ffmpeg_path` is set, we look for the
    /// sibling `ffprobe` binary next to it.
    #[serde(default)]
    pub ffprobe_path: Option<String>,
    /// Phase 10 — user override for the models directory. When
    /// `Some`, replaces the OS-default `<app_data>/models` path. The
    /// UI shows this in Settings › AI Models. Applied by the worker
    /// on next `initialize` (see `AppState::apply_models_dir`).
    #[serde(default)]
    pub models_dir_override: Option<String>,
    /// Phase 10 — the user has completed (or skipped) the optional
    /// first-run model setup at least once. When `false`, the
    /// dashboard shows a lightweight banner nudging them toward
    /// Settings › AI Models.
    #[serde(default)]
    pub first_run_completed: bool,
    /// Phase 11 — release Whisper / llama.cpp / Piper from RAM this
    /// many seconds after their last job settles. `None` or `0`
    /// disables auto-unload (models stay resident until the user
    /// closes the app or hits "Unload all"). Default is 90 s which
    /// balances reload cost against RAM pressure on 16 GB Macs.
    #[serde(default = "default_auto_unload_after_secs")]
    pub auto_unload_after_secs: Option<u32>,
    /// Phase 11 — how many CPU threads to hand to inference engines
    /// that expose a `n_threads`-style knob. `None` = let the engine
    /// pick (typically number of physical cores). Applies to
    /// llama.cpp and — where the backend supports it — Whisper's
    /// `cpu_threads` argument.
    #[serde(default)]
    pub cpu_threads: Option<u32>,
    /// Phase 11 — allow the worker to enable Metal / CUDA back-ends
    /// when the underlying engine supports them. Default `true`.
    /// The engine's own capability probe still gates real usage; we
    /// only ever *ask* — a machine without Metal will silently
    /// fall back to CPU regardless of this flag.
    #[serde(default = "default_true")]
    pub gpu_acceleration: bool,
}

fn default_auto_unload_after_secs() -> Option<u32> {
    Some(90)
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // Phase 12 — default to OFF for fresh installs. The only
            // network operation the app ever performs is an explicit,
            // user-initiated model download (see
            // `commands::download_whisper_model`); gating it by
            // default just leaves first-time users staring at a
            // disabled button with no obvious path forward. Users who
            // want a hard air-gap can flip this on after their models
            // are installed and the app will then refuse every HTTP
            // call for the rest of its life.
            offline_mode: false,
            source_language: "en".into(),
            target_language: "vi".into(),
            max_concurrent_jobs: 1,
            log_level: LogLevel::Info,
            whisper_model: None,
            translation_model: None,
            tts_voice: None,
            ffmpeg_path: None,
            ffprobe_path: None,
            models_dir_override: None,
            first_run_completed: false,
            auto_unload_after_secs: default_auto_unload_after_secs(),
            cpu_threads: None,
            gpu_acceleration: true,
        }
    }
}

/// Same shape as [`AppSettings`] but every field is optional — used for
/// partial updates coming from the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AppSettingsPatch {
    pub offline_mode: Option<bool>,
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub max_concurrent_jobs: Option<u32>,
    pub log_level: Option<LogLevel>,
    pub whisper_model: Option<Option<String>>,
    pub translation_model: Option<Option<String>>,
    pub tts_voice: Option<Option<String>>,
    pub ffmpeg_path: Option<Option<String>>,
    pub ffprobe_path: Option<Option<String>>,
    pub models_dir_override: Option<Option<String>>,
    pub first_run_completed: Option<bool>,
    pub auto_unload_after_secs: Option<Option<u32>>,
    pub cpu_threads: Option<Option<u32>>,
    pub gpu_acceleration: Option<bool>,
}

impl AppSettings {
    pub fn apply(&mut self, patch: AppSettingsPatch) {
        if let Some(v) = patch.offline_mode {
            self.offline_mode = v;
        }
        if let Some(v) = patch.source_language {
            self.source_language = v;
        }
        if let Some(v) = patch.target_language {
            self.target_language = v;
        }
        if let Some(v) = patch.max_concurrent_jobs {
            // Sanity clamp — protect the queue scheduler.
            self.max_concurrent_jobs = v.clamp(1, 8);
        }
        if let Some(v) = patch.log_level {
            self.log_level = v;
        }
        if let Some(v) = patch.whisper_model {
            self.whisper_model = v;
        }
        if let Some(v) = patch.translation_model {
            self.translation_model = v;
        }
        if let Some(v) = patch.tts_voice {
            self.tts_voice = v;
        }
        if let Some(v) = patch.ffmpeg_path {
            self.ffmpeg_path = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.ffprobe_path {
            self.ffprobe_path = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.models_dir_override {
            self.models_dir_override = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.first_run_completed {
            self.first_run_completed = v;
        }
        if let Some(v) = patch.auto_unload_after_secs {
            // Treat 0 as "disabled" so the UI can use a simple number
            // input without a separate on/off toggle.
            self.auto_unload_after_secs = v.filter(|s| *s > 0);
        }
        if let Some(v) = patch.cpu_threads {
            self.cpu_threads = v.filter(|n| *n > 0);
        }
        if let Some(v) = patch.gpu_acceleration {
            self.gpu_acceleration = v;
        }
    }

    pub fn load_or_default(path: &Path) -> io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str::<AppSettings>(&text)
                .map_err(|e| io::Error::other(format!("invalid config: {e}"))),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err),
        }
    }

    pub fn save_atomic(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("json.tmp");
        let text =
            serde_json::to_string_pretty(self).map_err(|e| io::Error::other(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }
}

/// Thread-safe in-memory settings cache. Rust owns the persistence side.
#[derive(Debug)]
pub struct SettingsStore {
    inner: RwLock<AppSettings>,
    path: std::path::PathBuf,
}

impl SettingsStore {
    pub fn open(path: std::path::PathBuf) -> io::Result<Self> {
        let settings = AppSettings::load_or_default(&path)?;
        Ok(Self {
            inner: RwLock::new(settings),
            path,
        })
    }

    pub fn snapshot(&self) -> AppSettings {
        self.inner.read().clone()
    }

    pub fn update(&self, patch: AppSettingsPatch) -> io::Result<AppSettings> {
        let mut guard = self.inner.write();
        guard.apply(patch);
        guard.save_atomic(&self.path)?;
        Ok(guard.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_online_and_vi() {
        // Phase 12 — fresh installs must ship with `offline_mode =
        // false` so `Download & transcribe` works out of the box.
        // The only entry that speaks HTTP is the explicit model
        // download command, so leaving this off by default is safe.
        let s = AppSettings::default();
        assert!(!s.offline_mode);
        assert_eq!(s.target_language, "vi");
        assert_eq!(s.max_concurrent_jobs, 1);
    }

    #[test]
    fn patch_updates_selectively() {
        let mut s = AppSettings::default();
        s.apply(AppSettingsPatch {
            source_language: Some("ja".into()),
            max_concurrent_jobs: Some(4),
            ..Default::default()
        });
        assert_eq!(s.source_language, "ja");
        assert_eq!(s.max_concurrent_jobs, 4);
        assert_eq!(s.target_language, "vi");
    }

    #[test]
    fn max_jobs_are_clamped() {
        let mut s = AppSettings::default();
        s.apply(AppSettingsPatch {
            max_concurrent_jobs: Some(999),
            ..Default::default()
        });
        assert_eq!(s.max_concurrent_jobs, 8);

        s.apply(AppSettingsPatch {
            max_concurrent_jobs: Some(0),
            ..Default::default()
        });
        assert_eq!(s.max_concurrent_jobs, 1);
    }

    #[test]
    fn round_trip_file() {
        let dir = tempdir();
        let path = dir.join("config.json");
        let s = AppSettings {
            source_language: "ko".into(),
            ..AppSettings::default()
        };
        s.save_atomic(&path).unwrap();

        let loaded = AppSettings::load_or_default(&path).unwrap();
        assert_eq!(loaded, s);
    }

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lmt-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
