//! Wiring: paths → config → db → services → worker supervisor.
//!
//! Everything that lives for the whole app lifetime is constructed here
//! and stored in Tauri's managed state.

use std::path::PathBuf;
use std::sync::Arc;

use tauri::AppHandle;

use crate::audio::AudioExtractor;
use crate::config::SettingsStore;
use crate::db::{Db, DbHandle};
use crate::ffmpeg::detection::{FfmpegAvailability, FfmpegHandle, FfmpegPathOverride};
use crate::jobs::{JobRegistry, JobsRepo};
use crate::integrations::youtube::YouTubeService;
use crate::mix::MixService;
use crate::models::ModelRegistry;
use crate::paths::AppPaths;
use crate::projects::ProjectService;
use crate::render::RenderService;
use crate::stt::SttService;
use crate::subtitles::SubtitleService;
use crate::sync::SyncService;
use crate::translation::TranslationService;
use crate::tts::TtsService;
use crate::worker::{self, WorkerConfig, WorkerSupervisor};

pub struct AppState {
    pub paths: AppPaths,
    pub settings: Arc<SettingsStore>,
    pub db: DbHandle,
    pub projects: Arc<ProjectService>,
    pub worker: Arc<WorkerSupervisor>,
    pub ffmpeg: Arc<FfmpegHandle>,
    pub jobs: Arc<JobRegistry>,
    pub audio: Arc<AudioExtractor>,
    pub stt: Arc<SttService>,
    pub translation: Arc<TranslationService>,
    pub subtitles: Arc<SubtitleService>,
    pub tts: Arc<TtsService>,
    pub sync: Arc<SyncService>,
    pub mix: Arc<MixService>,
    pub render: Arc<RenderService>,
    pub youtube: Arc<YouTubeService>,
    pub models: ModelRegistry,
}

impl AppState {
    pub fn bootstrap(app: AppHandle, paths: AppPaths) -> anyhow::Result<Self> {
        let settings = Arc::new(SettingsStore::open(paths.config_file.clone()).map_err(|e| {
            anyhow::anyhow!(
                "failed to open settings ({}): {e}",
                paths.config_file.display()
            )
        })?);

        let db = Db::open(&paths.db_path).map_err(|e| {
            anyhow::anyhow!("failed to open sqlite ({}): {e}", paths.db_path.display())
        })?;

        // Any jobs left in a running/queued state on the previous run are stale.
        match JobsRepo::reap_orphans(&db) {
            Ok(n) if n > 0 => tracing::warn!(n, "reaped orphaned jobs from previous run"),
            Ok(_) => {}
            Err(err) => tracing::warn!(%err, "reap_orphans failed; continuing"),
        }

        let projects = ProjectService::new(paths.clone(), db.clone());

        // Resolve the models directory: user override wins, otherwise
        // the OS-default `<app_data>/models` from `paths`. Making the
        // directory optional here (rather than lazily in the worker)
        // means the "Models dir" surface in Settings never lies.
        let effective_models_dir = effective_models_dir(&paths, &settings.snapshot());
        crate::paths::ensure_dir(&effective_models_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to prepare models dir ({}): {e}",
                effective_models_dir.display()
            )
        })?;

        let snap = settings.snapshot();
        let worker_cfg = WorkerConfig {
            python_bin: worker::supervisor_python_bin(),
            worker_root: worker::supervisor_worker_root(),
            data_dir: paths.data_dir.clone(),
            models_dir: effective_models_dir,
            log_level: format!("{:?}", snap.log_level).to_lowercase(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            // Phase 11 — user-tunable performance knobs. Providers
            // read these at model-load time and reload transparently
            // when `reinitialize_perf` is called after a settings
            // change.
            cpu_threads: snap.cpu_threads,
            gpu_acceleration: snap.gpu_acceleration,
        };
        tracing::info!(
            python_bin = %worker_cfg.python_bin.display(),
            worker_root = %worker_cfg.worker_root.display(),
            "worker config resolved"
        );

        let worker_sup = WorkerSupervisor::new(worker_cfg, app.clone());
        worker_sup.start();

        // FFmpeg + audio pipeline.
        let ffmpeg = FfmpegHandle::new();
        let jobs = JobRegistry::new();
        let audio = AudioExtractor::new(app.clone(), db.clone(), ffmpeg.clone(), jobs.clone());
        let stt = SttService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            jobs.clone(),
            worker_sup.clone(),
        );
        let translation = TranslationService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            jobs.clone(),
            worker_sup.clone(),
        );
        let subtitles = SubtitleService::new(projects.clone());
        let tts = TtsService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            subtitles.clone(),
            jobs.clone(),
            worker_sup.clone(),
        );
        let sync = SyncService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            subtitles.clone(),
            jobs.clone(),
            worker_sup.clone(),
            ffmpeg.clone(),
        );
        let mix = MixService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            subtitles.clone(),
            jobs.clone(),
            ffmpeg.clone(),
        );
        let render = RenderService::new(
            app.clone(),
            db.clone(),
            projects.clone(),
            subtitles.clone(),
            jobs.clone(),
            ffmpeg.clone(),
        );
        // Phase 13 stays isolated from the local media pipeline. Constructing
        // the service performs no network traffic; OAuth and API calls only
        // happen after an explicit YouTube action.
        let youtube = YouTubeService::new(app, &paths.config_dir, ffmpeg.clone());
        // Phase 10 — thin aggregation layer over the per-stage
        // Python-side registries. Zero-cost until the UI actually
        // asks for a scan.
        let models = ModelRegistry::new(stt.clone(), translation.clone(), tts.clone());
        // (worker_sup, projects, jobs, db still owned here — used below
        // for the returned AppState.)

        // Kick off ffmpeg detection in the background so startup isn't blocked.
        let ffmpeg_bg = ffmpeg.clone();
        let ov = current_ffmpeg_override(&settings.snapshot());
        tokio::spawn(async move {
            let av = ffmpeg_bg.refresh(ov).await;
            tracing::info!(?av.available, "ffmpeg detection complete");
        });

        // Phase 11 — best-effort sweep of `<projects>/**/*.tmp` files
        // left behind by a crashed atomic write on the previous run.
        // Cache writers always rename `foo.json.tmp` → `foo.json`
        // atomically; if the process died mid-write the sibling `.tmp`
        // is orphaned. Cleanup runs in the background so it never
        // delays startup even on hosts with many projects.
        let projects_dir = paths.projects_dir.clone();
        tokio::spawn(async move {
            match sweep_orphan_temp_files(&projects_dir) {
                Ok(n) if n > 0 => {
                    tracing::info!(n, "cleaned orphan .tmp files")
                }
                Ok(_) => {}
                Err(err) => tracing::warn!(%err, "orphan .tmp sweep failed"),
            }
        });

        Ok(Self {
            paths,
            settings,
            db,
            projects,
            worker: worker_sup,
            ffmpeg,
            jobs,
            audio,
            stt,
            translation,
            subtitles,
            tts,
            sync,
            mix,
            render,
            youtube,
            models,
        })
    }

    /// Phase 10 — the models directory the app is currently using,
    /// honouring `AppSettings::models_dir_override`.
    pub fn effective_models_dir(&self) -> PathBuf {
        effective_models_dir(&self.paths, &self.settings.snapshot())
    }

    /// Re-probe ffmpeg after a settings change. Returns the fresh
    /// availability snapshot.
    pub async fn refresh_ffmpeg(&self) -> FfmpegAvailability {
        let ov = current_ffmpeg_override(&self.settings.snapshot());
        self.ffmpeg.refresh(ov).await
    }
}

fn current_ffmpeg_override(s: &crate::config::AppSettings) -> FfmpegPathOverride {
    FfmpegPathOverride {
        ffmpeg: s.ffmpeg_path.as_deref().map(PathBuf::from),
        ffprobe: s.ffprobe_path.as_deref().map(PathBuf::from),
        bundled_bin_dir: None, // populated by a future packaging pass
    }
}

/// Compute the effective models directory: `models_dir_override`
/// from settings (if set) wins, otherwise the OS-default
/// `<app_data>/models` computed by `AppPaths`.
fn effective_models_dir(paths: &AppPaths, s: &crate::config::AppSettings) -> PathBuf {
    match s.models_dir_override.as_deref() {
        Some(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => paths.models_dir.clone(),
    }
}

/// Phase 11 — best-effort recursive sweep of `.tmp` sidecar files
/// left behind by a crashed atomic write. Returns the number of
/// files removed. Errors on individual entries are logged but do
/// not abort the sweep — a hostile filesystem shouldn't stop us
/// cleaning up the neighbours.
///
/// Depth is capped so a pathological project tree (symlink loops,
/// millions of nested dirs) can't hang startup.
fn sweep_orphan_temp_files(root: &std::path::Path) -> std::io::Result<usize> {
    const MAX_DEPTH: usize = 8;
    let mut removed: usize = 0;
    if !root.exists() {
        return Ok(0);
    }
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > MAX_DEPTH {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(err) => {
                tracing::debug!(dir = %dir.display(), %err, "read_dir failed during .tmp sweep");
                continue;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push((path, depth + 1));
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            // Match both `foo.tmp` and `foo.json.tmp` style sidecars.
            if !name.ends_with(".tmp") {
                continue;
            }
            if let Err(err) = std::fs::remove_file(&path) {
                tracing::debug!(path = %path.display(), %err, "failed to remove orphan .tmp");
            } else {
                removed += 1;
            }
        }
    }
    Ok(removed)
}
