//! Tauri `#[command]` entry points. Each returns `Result<T, AppError>`;
//! `AppError` is JSON-serialisable and mirrored by `src/ipc/types.ts`.
//!
//! Commands are intentionally thin — validation + call the domain service.

use std::path::PathBuf;

use tauri::State;

use crate::app::AppState;
use crate::audio::{ExtractionRequest, ExtractionStart};
use crate::config::{AppSettings, AppSettingsPatch};
use crate::db::models::{ProjectModelPatch, ProjectRecord, ProjectSummary};
use crate::errors::AppError;
use crate::ffmpeg::extract::AudioExtractParams;
use crate::ffmpeg::{probe, FfmpegAvailability, VideoMetadata};
use crate::ipc::{AppInfo, ProjectMediaState};
use crate::integrations::youtube::{
    YouTubeAccount, YouTubeConnectionState, YouTubeError, YouTubePlaylist, YouTubePublishOptions,
    YouTubePublishingHistoryEntry, YouTubeThumbnailResult, YouTubeUploadSnapshot,
    YouTubeVideoMetadata,
};
use crate::jobs::{JobSnapshot, JobsRepo};
use crate::mix::{
    MixEnv, MixGenerateStart, MixManifest, MixRequest, MixSettings, MixSummary, PreviewMixResult,
};
use crate::models::{
    self as model_mgr, ImportSpec, LocalModel, ModelDirectoryInfo, ModelManagerError,
};
use crate::projects::service::{CreateProjectInput, ImportMediaInput, ImportMediaResult};
use crate::render::{
    RenderEnv, RenderGenerateStart, RenderManifest, RenderRequest, RenderSettings, RenderStatus,
    RenderSummary,
};
use crate::stt::{ModelInfo, SttEnv, SttOptions, TranscribeStart, TranscriptSummary};
use crate::subtitles::{
    ExportKind, ExportSubtitlesResult, ImportSubtitlesResult, SubtitleDoc, SubtitleFormat,
    SubtitleSegmentPatch, SubtitleSummary,
};
use crate::sync::{
    PreviewSyncResult, SyncEnv, SyncGenerateStart, SyncManifest, SyncRequest, SyncSettings,
    SyncSummary,
};
use crate::translation::{
    TranslateOptions, TranslateStart, TranslationDoc, TranslationEnv, TranslationModelInfo,
    TranslationSummary,
};
use crate::tts::{
    CreateVoiceProfileRequest, GenerateRequest, PreviewResult, TtsEnv, TtsGenerateStart,
    TtsManifest, TtsSettings, TtsSummary, VoiceInfo,
};
use crate::worker::{EnvInfo, PingResponse, WorkerStatus};

/// Base URL the frontend points `<video>` and `<audio>` at.
///
/// Media cannot be played through a custom URI scheme in WebKit, so the
/// app serves it over loopback HTTP instead; see `crate::media_server`.
/// The URL carries a per-run token, so it is fetched rather than
/// hard-coded.
#[tauri::command]
pub async fn get_media_base_url() -> Result<String, AppError> {
    let url = crate::media_server::base_url().ok_or_else(|| {
        AppError::new("MEDIA_SERVER_UNAVAILABLE", "the local media server is not running")
            .with_hint("Restart the app. Playback needs it; the rest of the pipeline does not.")
    })?;
    tracing::debug!("frontend requested media server URL");
    Ok(url)
}

#[tauri::command]
pub async fn get_app_info(state: State<'_, AppState>) -> Result<AppInfo, AppError> {
    let p = &state.paths;
    Ok(AppInfo {
        app_name: "Local Movie Translator".into(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        data_dir: p.data_dir.display().to_string(),
        config_dir: p.config_dir.display().to_string(),
        log_dir: p.log_dir.display().to_string(),
        projects_dir: p.projects_dir.display().to_string(),
        // Report the *effective* models dir so the Application panel
        // agrees with the AI Models panel when the user has an
        // override configured (Phase 10).
        models_dir: state.effective_models_dir().display().to_string(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
    })
}

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<AppSettings, AppError> {
    Ok(state.settings.snapshot())
}

#[tauri::command]
pub async fn update_settings(
    state: State<'_, AppState>,
    patch: AppSettingsPatch,
) -> Result<AppSettings, AppError> {
    let updated = state.settings.update(patch.clone())?;
    if patch.offline_mode == Some(true) {
        state.youtube.cancel_all_uploads();
    }
    // Any settings change that touches ffmpeg_path triggers a re-detect.
    if patch.ffmpeg_path.is_some() {
        let _ = state.refresh_ffmpeg().await;
    }
    // Phase 11 — perf knobs (`cpu_threads`, `gpu_acceleration`) are
    // pushed to the running worker so the change applies without an
    // app restart. Providers reload their model on the next call.
    if patch.cpu_threads.is_some() || patch.gpu_acceleration.is_some() {
        if let Err(err) = state
            .worker
            .reinitialize_perf(updated.cpu_threads, updated.gpu_acceleration)
            .await
        {
            tracing::warn!(
                %err,
                "worker refused perf hot-reload; changes take effect after restart",
            );
        }
    }
    Ok(updated)
}

#[tauri::command]
pub async fn list_projects(state: State<'_, AppState>) -> Result<Vec<ProjectSummary>, AppError> {
    Ok(state.projects.list().await?)
}

#[tauri::command]
pub async fn create_project(
    state: State<'_, AppState>,
    input: CreateProjectInput,
) -> Result<ProjectSummary, AppError> {
    Ok(state.projects.create(input).await?)
}

#[tauri::command]
pub async fn open_project(
    state: State<'_, AppState>,
    id: String,
) -> Result<ProjectRecord, AppError> {
    Ok(state.projects.open(id).await?)
}

#[tauri::command]
pub async fn delete_project(state: State<'_, AppState>, id: String) -> Result<(), AppError> {
    Ok(state.projects.delete(id).await?)
}

#[tauri::command]
pub async fn worker_status(state: State<'_, AppState>) -> Result<WorkerStatus, AppError> {
    Ok(state.worker.status().await)
}

#[tauri::command]
pub async fn worker_ping(state: State<'_, AppState>) -> Result<PingResponse, AppError> {
    Ok(state.worker.ping().await?)
}

#[tauri::command]
pub async fn worker_env_info(state: State<'_, AppState>) -> Result<EnvInfo, AppError> {
    Ok(state.worker.env_info().await?)
}

// ---------- Phase 2 ----------

#[tauri::command]
pub async fn get_ffmpeg_availability(
    state: State<'_, AppState>,
) -> Result<FfmpegAvailability, AppError> {
    Ok(state.ffmpeg.availability())
}

#[tauri::command]
pub async fn refresh_ffmpeg(state: State<'_, AppState>) -> Result<FfmpegAvailability, AppError> {
    Ok(state.refresh_ffmpeg().await)
}

#[tauri::command]
pub async fn probe_media(
    state: State<'_, AppState>,
    path: String,
) -> Result<VideoMetadata, AppError> {
    let svc = state.ffmpeg.get().ok_or_else(|| {
        AppError::recoverable(
            "FFMPEG_NOT_FOUND",
            state
                .ffmpeg
                .availability()
                .error
                .unwrap_or_else(|| "FFmpeg not detected".into()),
        )
        .with_stage("media")
        .with_hint("Install FFmpeg or set a custom path in Settings.")
    })?;
    Ok(probe::probe_video(svc.ffprobe(), &PathBuf::from(path)).await?)
}

#[tauri::command]
pub async fn import_media(
    state: State<'_, AppState>,
    input: ImportMediaInput,
) -> Result<ImportMediaResult, AppError> {
    Ok(state.projects.import_media(input).await?)
}

#[tauri::command]
pub async fn get_project_media(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<ProjectMediaState, AppError> {
    let rec = state.projects.open(project_id.clone()).await?;
    let mut out = ProjectMediaState::default();

    // Probe metadata only when a source is imported and ffmpeg exists.
    if let (Some(source), Some(svc)) = (rec.source_media_path.as_ref(), state.ffmpeg.get()) {
        match probe::probe_video(svc.ffprobe(), &PathBuf::from(source)).await {
            Ok(meta) => out.metadata = Some(meta),
            Err(err) => {
                tracing::warn!(%err, "probe failed for {source}");
            }
        }
    }

    // Cache manifest.
    let project_root = PathBuf::from(&rec.root_path);
    if let Ok(cache) = crate::audio::AudioCacheFile::load(&project_root) {
        if let Some(entry) = cache.original_wav.clone() {
            let abs = entry.absolute_output(&project_root);
            if abs.exists() {
                out.audio_absolute_path = Some(abs.display().to_string());
                out.audio = Some(entry);
            }
        }
    }

    // Transcript summary (if any).
    if let Ok(Some(t)) = crate::stt::TranscriptCacheFile::load(&project_root) {
        out.transcript = Some(crate::stt::TranscriptSummary::from_transcript(
            &t,
            crate::stt::TRANSCRIPT_RELATIVE,
        ));
    }

    // Translation summary (if any).
    if let Ok(Some(doc)) = crate::translation::TranslationCacheFile::load(&project_root) {
        out.translation = Some(crate::translation::TranslationSummary::from_doc(
            &doc,
            crate::translation::TRANSLATION_RELATIVE,
        ));
    }

    // Subtitle summary (if any).
    if let Ok(Some(doc)) = crate::subtitles::SubtitleCacheFile::load(&project_root) {
        out.subtitles = Some(crate::subtitles::SubtitleSummary::from_doc(
            &doc,
            crate::subtitles::SUBTITLES_RELATIVE,
        ));

        // TTS summary is only meaningful when subtitles exist — we
        // compare per-segment cache identity against voices.json.
        let manifest = crate::tts::TtsCacheFile::load(&project_root).ok().flatten();
        let (engine, default_voice) = manifest
            .as_ref()
            .map(|m| (m.engine.clone(), m.default_voice_id.clone()))
            .unwrap_or_else(|| ("piper".to_string(), String::new()));
        let summary = crate::tts::service::build_summary(
            &doc,
            manifest.as_ref(),
            &crate::tts::service::SummaryRequest {
                engine,
                default_voice_id: default_voice,
                settings: TtsSettings::default(),
            },
        );
        out.tts = Some(summary);

        // Phase 7 — sync summary. Only meaningful when both subtitles
        // *and* TTS exist; otherwise leave it None so the UI hides the
        // panel.
        let sync_manifest = crate::sync::SyncCacheFile::load(&project_root)
            .ok()
            .flatten();
        let sync_settings = sync_manifest
            .as_ref()
            .map(|m| m.settings)
            .unwrap_or_default();
        out.sync = Some(crate::sync::service::build_summary(
            &doc,
            manifest.as_ref(),
            sync_manifest.as_ref(),
            &sync_settings,
        ));

        // Phase 8 — mix summary. Reads the sync manifest + optional
        // source fingerprint so the UI knows whether the mix is Ready
        // / Stale / Missing without asking the user to press "Refresh".
        let mix_manifest = crate::mix::MixCacheFile::load(&project_root).ok().flatten();
        let mix_settings = mix_manifest
            .as_ref()
            .map(|m| m.settings)
            .unwrap_or_default();
        // Phase 11 — the source fingerprint stored on the project
        // record is authoritative as long as the file's size/mtime
        // haven't drifted. `get_project_media` runs on every job
        // progress tick, so re-hashing a multi-GB movie here (even
        // the 128 KiB sentinel-window hash) is 10-100 ms of pointless
        // IO. We only recompute if the on-disk stat has moved, which
        // is the same trigger the importer already uses.
        let source_fp = rec.source_media_path.as_ref().and_then(|s| {
            reuse_or_recompute_fingerprint(
                &PathBuf::from(s),
                rec.source_hash.as_deref(),
                rec.source_size,
                rec.source_modified_at.as_ref(),
            )
        });
        out.mix = Some(crate::mix::service::build_summary(
            &project_root,
            &doc,
            sync_manifest.as_ref(),
            mix_manifest.as_ref(),
            &mix_settings,
            source_fp.as_ref(),
        ));

        // Phase 9 — render summary. Uses the mix entry (if any) so
        // stale renders can be detected without re-probing the source.
        let render_manifest = crate::render::RenderCacheFile::load(&project_root)
            .ok()
            .flatten();
        let render_settings = render_manifest
            .as_ref()
            .map(|m| m.settings.clone())
            .unwrap_or_default();
        out.render = Some(crate::render::service::build_summary(
            &project_root,
            Some(&doc),
            mix_manifest.as_ref().and_then(|m| m.current.as_ref()),
            render_manifest.as_ref(),
            &render_settings,
            source_fp.as_ref(),
        ));
    }

    // Active jobs from DB.
    let db = state.db.clone();
    out.active_jobs = db
        .run(move |d| JobsRepo::list_active(d, &project_id))
        .await?;
    Ok(out)
}

#[tauri::command]
pub async fn extract_audio(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<ExtractionStart, AppError> {
    let rec = state.projects.open(project_id.clone()).await?;
    let source = rec.source_media_path.as_deref().ok_or_else(|| {
        AppError::recoverable(
            "NO_SOURCE_MEDIA",
            "Import a video into the project before extracting audio.",
        )
        .with_stage("audio_extraction")
    })?;

    let req = ExtractionRequest {
        project_id: project_id.clone(),
        project_root: PathBuf::from(&rec.root_path),
        source_media_path: PathBuf::from(source),
        params: AudioExtractParams::whisper_default(),
    };
    Ok(state.audio.extract(req).await?)
}

#[tauri::command]
pub async fn cancel_job(state: State<'_, AppState>, job_id: String) -> Result<(), AppError> {
    Ok(state.jobs.cancel(&job_id)?)
}

#[tauri::command]
pub async fn list_active_jobs(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Vec<crate::jobs::JobSnapshot>, AppError> {
    let db = state.db.clone();
    Ok(db
        .run(move |d| JobsRepo::list_active(d, &project_id))
        .await?)
}

/// Phase 11 — lightweight runtime snapshot. Cheap to call (a
/// couple of atomic reads + a `sysinfo`-free process memory probe);
/// safe to poll every couple of seconds from the UI.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStats {
    /// Global count of jobs the registry considers "active" (queued
    /// or running) across every project.
    pub active_jobs: usize,
    /// Total unique projects with at least one active job.
    pub active_projects: usize,
    /// Approx. resident-set-size of the Tauri host process, in bytes.
    /// `None` when the platform doesn't expose it cheaply.
    pub host_rss_bytes: Option<u64>,
    /// Approx. resident-set-size of the Python worker process, in
    /// bytes. `None` when the worker isn't running or the platform
    /// doesn't expose it.
    pub worker_rss_bytes: Option<u64>,
    /// Worker uptime in seconds, or `None` when not running.
    pub worker_uptime_secs: Option<u64>,
}

#[tauri::command]
pub async fn get_runtime_stats(state: State<'_, AppState>) -> Result<RuntimeStats, AppError> {
    let active = state.jobs.snapshot_all();
    let active_jobs = active.len();
    let active_projects = {
        let mut set = std::collections::HashSet::<String>::new();
        for j in &active {
            set.insert(j.project_id.clone());
        }
        set.len()
    };
    let worker = state.worker.status().await;
    let worker_running = matches!(worker.state, crate::worker::WorkerState::Running);
    Ok(RuntimeStats {
        active_jobs,
        active_projects,
        host_rss_bytes: process_rss_bytes(std::process::id()),
        worker_rss_bytes: if worker_running {
            worker.pid.and_then(process_rss_bytes)
        } else {
            None
        },
        worker_uptime_secs: if worker_running {
            Some(worker.uptime_ms / 1000)
        } else {
            None
        },
    })
}

/// Best-effort RSS lookup that avoids pulling in `sysinfo`. Returns
/// `None` when the platform doesn't offer a cheap probe.
#[cfg(target_os = "linux")]
fn process_rss_bytes(pid: u32) -> Option<u64> {
    let statm = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // Linux page size is 4 KiB on every architecture we care about.
    Some(pages * 4096)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn process_rss_bytes(pid: u32) -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb * 1024)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn process_rss_bytes(_pid: u32) -> Option<u64> {
    None
}

// ---------- Phase 3 (Speech recognition) ----------

#[tauri::command]
pub async fn get_stt_env(state: State<'_, AppState>) -> Result<SttEnv, AppError> {
    Ok(state.stt.env().await?)
}

#[tauri::command]
pub async fn list_whisper_models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, AppError> {
    Ok(state.stt.list_models().await?)
}

#[tauri::command]
pub async fn download_whisper_model(
    state: State<'_, AppState>,
    name: String,
) -> Result<JobSnapshot, AppError> {
    // Phase 12 — log every download attempt at INFO so we can
    // diagnose "I clicked download and nothing happened" without
    // asking the user to enable debug logging first.
    tracing::info!(model = %name, "download_whisper_model requested");
    // Phase 10 — Offline Mode is a hard gate for the only entry
    // point in the app that speaks HTTP. We refuse the request here
    // rather than at the worker layer so the UI gets a stable
    // structured error code and the download job is never even
    // registered.
    if state.settings.snapshot().offline_mode {
        tracing::info!(
            model = %name,
            "download_whisper_model refused: offline mode enabled",
        );
        return Err(AppError::from(ModelManagerError::NetworkDisabled));
    }
    let result = state.stt.download_model(name.clone()).await;
    match &result {
        Ok(snap) => tracing::info!(
            model = %name,
            job_id = %snap.id,
            "download_whisper_model started",
        ),
        Err(err) => tracing::warn!(
            model = %name,
            error = %err,
            "download_whisper_model failed to start",
        ),
    }
    Ok(result?)
}

#[tauri::command]
pub async fn transcribe(
    state: State<'_, AppState>,
    project_id: String,
    options: Option<SttOptions>,
) -> Result<TranscribeStart, AppError> {
    Ok(state
        .stt
        .transcribe(project_id, options.unwrap_or_default())
        .await?)
}

#[tauri::command]
pub async fn get_project_transcript(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<TranscriptSummary>, AppError> {
    Ok(state.stt.get_transcript_summary(project_id).await?)
}

// ---------- Phase 4 (Local LLM translation) ----------

#[tauri::command]
pub async fn get_translation_env(state: State<'_, AppState>) -> Result<TranslationEnv, AppError> {
    Ok(state.translation.env().await?)
}

#[tauri::command]
pub async fn list_translation_models(
    state: State<'_, AppState>,
) -> Result<Vec<TranslationModelInfo>, AppError> {
    Ok(state.translation.list_models().await?)
}

#[tauri::command]
pub async fn list_recommended_translation_presets(
    state: State<'_, AppState>,
) -> Result<Vec<crate::translation::RecommendedPreset>, AppError> {
    Ok(state.translation.list_recommended().await?)
}

#[tauri::command]
pub async fn download_translation_model(
    state: State<'_, AppState>,
    preset: String,
) -> Result<JobSnapshot, AppError> {
    tracing::info!(preset = %preset, "download_translation_model requested");
    if state.settings.snapshot().offline_mode {
        tracing::info!(
            preset = %preset,
            "download_translation_model refused: offline mode enabled",
        );
        return Err(AppError::from(ModelManagerError::NetworkDisabled));
    }
    let result = state.translation.download_model(preset.clone()).await;
    match &result {
        Ok(snap) => tracing::info!(
            preset = %preset,
            job_id = %snap.id,
            "download_translation_model started",
        ),
        Err(err) => tracing::warn!(
            preset = %preset,
            error = %err,
            "download_translation_model failed to start",
        ),
    }
    Ok(result?)
}

#[tauri::command]
pub async fn translate(
    state: State<'_, AppState>,
    project_id: String,
    options: TranslateOptions,
) -> Result<TranslateStart, AppError> {
    Ok(state.translation.translate(project_id, options).await?)
}

#[tauri::command]
pub async fn get_project_translation(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<TranslationSummary>, AppError> {
    Ok(state
        .translation
        .get_translation_summary(project_id)
        .await?)
}

#[tauri::command]
pub async fn get_project_translation_doc(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<TranslationDoc>, AppError> {
    Ok(state.translation.get_translation_doc(project_id).await?)
}

#[tauri::command]
pub async fn update_translation_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
    translation: String,
) -> Result<TranslationSummary, AppError> {
    Ok(state
        .translation
        .update_segment(project_id, segment_id, translation)
        .await?)
}

// ---------- Phase 5 (Subtitle editor) ----------

#[tauri::command]
pub async fn get_project_subtitles(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<SubtitleSummary>, AppError> {
    Ok(state.subtitles.get_summary(project_id).await?)
}

#[tauri::command]
pub async fn get_project_subtitles_doc(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<SubtitleDoc>, AppError> {
    Ok(state.subtitles.get_doc(project_id).await?)
}

#[tauri::command]
pub async fn rebuild_project_subtitles(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<SubtitleDoc, AppError> {
    Ok(state.subtitles.rebuild_from_sources(project_id).await?)
}

#[tauri::command]
pub async fn update_subtitle_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
    patch: SubtitleSegmentPatch,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .update_segment(project_id, segment_id, patch)
        .await?)
}

#[tauri::command]
pub async fn assign_subtitle_voice_to_speaker(
    state: State<'_, AppState>,
    project_id: String,
    speaker: String,
    voice_id: Option<String>,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .assign_voice_to_speaker(project_id, speaker, voice_id)
        .await?)
}

#[tauri::command]
pub async fn add_subtitle_segment(
    state: State<'_, AppState>,
    project_id: String,
    after_id: Option<u32>,
    start: f64,
    end: f64,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .add_segment(project_id, after_id, start, end)
        .await?)
}

#[tauri::command]
pub async fn delete_subtitle_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .delete_segment(project_id, segment_id)
        .await?)
}

#[tauri::command]
pub async fn split_subtitle_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
    split_time: f64,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .split_segment(project_id, segment_id, split_time)
        .await?)
}

#[tauri::command]
pub async fn merge_subtitle_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
) -> Result<SubtitleDoc, AppError> {
    Ok(state
        .subtitles
        .merge_segment(project_id, segment_id)
        .await?)
}

#[tauri::command]
pub async fn clear_subtitle_dirty(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<SubtitleDoc, AppError> {
    Ok(state.subtitles.clear_dirty(project_id).await?)
}

#[tauri::command]
pub async fn import_subtitles(
    state: State<'_, AppState>,
    project_id: String,
    path: String,
) -> Result<ImportSubtitlesResult, AppError> {
    Ok(state.subtitles.import_from_file(project_id, path).await?)
}

#[tauri::command]
pub async fn export_subtitles(
    state: State<'_, AppState>,
    project_id: String,
    path: String,
    format: SubtitleFormat,
    kind: Option<ExportKind>,
) -> Result<ExportSubtitlesResult, AppError> {
    Ok(state
        .subtitles
        .export_to_file(project_id, path, format, kind.unwrap_or_default())
        .await?)
}

// ---------- Phase 6 (Local TTS / AI dubbing) ----------

#[tauri::command]
pub async fn get_tts_env(state: State<'_, AppState>) -> Result<TtsEnv, AppError> {
    Ok(state.tts.env().await?)
}

#[tauri::command]
pub async fn list_tts_voices(state: State<'_, AppState>) -> Result<Vec<VoiceInfo>, AppError> {
    Ok(state.tts.list_voices().await?)
}

#[tauri::command]
pub async fn create_tts_voice_profile(
    state: State<'_, AppState>,
    request: CreateVoiceProfileRequest,
) -> Result<VoiceInfo, AppError> {
    Ok(state.tts.create_voice_profile(request).await?)
}

#[tauri::command]
pub async fn list_recommended_tts_voices(
    state: State<'_, AppState>,
) -> Result<Vec<crate::tts::RecommendedVoicePreset>, AppError> {
    Ok(state.tts.list_recommended_voices().await?)
}

#[tauri::command]
pub async fn download_tts_voice(
    state: State<'_, AppState>,
    preset: String,
) -> Result<JobSnapshot, AppError> {
    tracing::info!(preset = %preset, "download_tts_voice requested");
    if state.settings.snapshot().offline_mode {
        tracing::info!(
            preset = %preset,
            "download_tts_voice refused: offline mode enabled",
        );
        return Err(AppError::from(ModelManagerError::NetworkDisabled));
    }
    let result = state.tts.download_voice(preset.clone()).await;
    match &result {
        Ok(snap) => tracing::info!(
            preset = %preset,
            job_id = %snap.id,
            "download_tts_voice started",
        ),
        Err(err) => tracing::warn!(
            preset = %preset,
            error = %err,
            "download_tts_voice failed to start",
        ),
    }
    Ok(result?)
}

#[tauri::command]
pub async fn get_project_tts_summary(
    state: State<'_, AppState>,
    project_id: String,
    engine: Option<String>,
    default_voice_id: Option<String>,
    settings: Option<TtsSettings>,
) -> Result<Option<TtsSummary>, AppError> {
    let req = crate::tts::service::SummaryRequest {
        engine: engine.unwrap_or_else(|| "piper".into()),
        default_voice_id: default_voice_id.unwrap_or_default(),
        settings: settings.unwrap_or_default(),
    };
    Ok(state.tts.get_summary(project_id, req).await?)
}

#[tauri::command]
pub async fn get_project_tts_manifest(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<TtsManifest>, AppError> {
    Ok(state.tts.get_manifest(project_id).await?)
}

#[tauri::command]
pub async fn preview_tts_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
    voice_id: Option<String>,
    engine: Option<String>,
    settings: Option<TtsSettings>,
) -> Result<PreviewResult, AppError> {
    Ok(state
        .tts
        .preview_segment(project_id, segment_id, voice_id, settings, engine)
        .await?)
}

#[tauri::command]
pub async fn generate_tts(
    state: State<'_, AppState>,
    project_id: String,
    request: GenerateRequest,
) -> Result<TtsGenerateStart, AppError> {
    Ok(state.tts.generate(project_id, request).await?)
}

// ---------- Phase 7 (Voice synchronisation) ----------

#[tauri::command]
pub async fn get_sync_env(state: State<'_, AppState>) -> Result<SyncEnv, AppError> {
    Ok(state.sync.env().await?)
}

#[tauri::command]
pub async fn get_project_sync_summary(
    state: State<'_, AppState>,
    project_id: String,
    settings: Option<SyncSettings>,
) -> Result<Option<SyncSummary>, AppError> {
    Ok(state
        .sync
        .get_summary(project_id, settings.unwrap_or_default())
        .await?)
}

#[tauri::command]
pub async fn get_project_sync_manifest(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<SyncManifest>, AppError> {
    Ok(state.sync.get_manifest(project_id).await?)
}

#[tauri::command]
pub async fn preview_sync_segment(
    state: State<'_, AppState>,
    project_id: String,
    segment_id: u32,
    settings: Option<SyncSettings>,
) -> Result<PreviewSyncResult, AppError> {
    Ok(state
        .sync
        .preview_segment(project_id, segment_id, settings)
        .await?)
}

#[tauri::command]
pub async fn apply_sync(
    state: State<'_, AppState>,
    project_id: String,
    request: SyncRequest,
) -> Result<SyncGenerateStart, AppError> {
    Ok(state.sync.apply(project_id, request).await?)
}

// ---------- Phase 8 (Audio mixing) ----------

#[tauri::command]
pub async fn get_mix_env(state: State<'_, AppState>) -> Result<MixEnv, AppError> {
    Ok(state.mix.env().await?)
}

#[tauri::command]
pub async fn get_project_mix_summary(
    state: State<'_, AppState>,
    project_id: String,
    settings: Option<MixSettings>,
) -> Result<Option<MixSummary>, AppError> {
    Ok(state
        .mix
        .get_summary(project_id, settings.unwrap_or_default())
        .await?)
}

#[tauri::command]
pub async fn get_project_mix_manifest(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<MixManifest>, AppError> {
    Ok(state.mix.get_manifest(project_id).await?)
}

#[tauri::command]
pub async fn get_project_mix_preview(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<PreviewMixResult>, AppError> {
    Ok(state.mix.get_preview(project_id).await?)
}

#[tauri::command]
pub async fn apply_mix(
    state: State<'_, AppState>,
    project_id: String,
    request: MixRequest,
) -> Result<MixGenerateStart, AppError> {
    Ok(state.mix.apply(project_id, request).await?)
}

// ------------------------------------------------------------ Phase 9

#[tauri::command]
pub async fn get_render_env(state: State<'_, AppState>) -> Result<RenderEnv, AppError> {
    Ok(state.render.env().await?)
}

#[tauri::command]
pub async fn get_project_render_summary(
    state: State<'_, AppState>,
    project_id: String,
    settings: Option<RenderSettings>,
) -> Result<Option<RenderSummary>, AppError> {
    Ok(state
        .render
        .get_summary(project_id, settings.unwrap_or_default())
        .await?)
}

#[tauri::command]
pub async fn get_project_render_manifest(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Option<RenderManifest>, AppError> {
    Ok(state.render.get_manifest(project_id).await?)
}

#[tauri::command]
pub async fn apply_render(
    state: State<'_, AppState>,
    project_id: String,
    request: RenderRequest,
) -> Result<RenderGenerateStart, AppError> {
    Ok(state.render.apply(project_id, request).await?)
}

// ------------------------------------------------------------ Phase 10

#[tauri::command]
pub async fn list_local_models(state: State<'_, AppState>) -> Result<Vec<LocalModel>, AppError> {
    Ok(state.models.list().await)
}

#[tauri::command]
pub async fn rescan_local_models(state: State<'_, AppState>) -> Result<Vec<LocalModel>, AppError> {
    Ok(state.models.rescan().await)
}

#[tauri::command]
pub async fn get_model_directory(
    state: State<'_, AppState>,
) -> Result<ModelDirectoryInfo, AppError> {
    let effective = state.effective_models_dir();
    let default_path = state.paths.models_dir.clone();
    let is_default = effective == default_path;
    let exists = effective.exists();
    Ok(ModelDirectoryInfo {
        path: effective.display().to_string(),
        is_default,
        default_path: default_path.display().to_string(),
        whisper_subdir: effective.join("whisper").display().to_string(),
        translation_subdir: effective.join("translation").display().to_string(),
        tts_subdir: effective.join("tts").display().to_string(),
        exists,
    })
}

#[tauri::command]
pub async fn set_model_directory(
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<ModelDirectoryInfo, AppError> {
    let trimmed = path.map(|p| p.trim().to_string()).filter(|s| !s.is_empty());
    // Cheap sanity check so we don't silently accept a relative path.
    if let Some(ref p) = trimmed {
        let pb = PathBuf::from(p);
        if !pb.is_absolute() {
            return Err(AppError::recoverable(
                "MODEL_DIR_NOT_ABSOLUTE",
                format!("Models directory path must be absolute: {p}"),
            )
            .with_stage("models")
            .with_hint("Pick an absolute directory path."));
        }
        // Probe writability so we fail loudly here rather than at
        // the next `import_local_model` call.
        if let Err(reason) = model_mgr::probe_writable(pb.clone()).await {
            return Err(AppError::from(ModelManagerError::ModelsDirNotWritable {
                path: pb,
                reason,
            }));
        }
    }

    state.settings.update(AppSettingsPatch {
        models_dir_override: Some(trimmed),
        ..Default::default()
    })?;

    // Ask the Python worker to re-configure its per-stage handlers
    // so the change takes effect immediately — no app restart
    // required. Best-effort: on failure we still return the new
    // directory info and let the UI surface a warning.
    let new_dir = state.effective_models_dir();
    if let Err(err) = state.worker.reinitialize_models_root(new_dir.clone()).await {
        tracing::warn!(
            %err,
            "worker refused to reload models root; the change will apply after restart",
        );
    }
    state.models.invalidate();

    get_model_directory(state).await
}

#[tauri::command]
pub async fn import_local_model(
    state: State<'_, AppState>,
    spec: ImportSpec,
) -> Result<LocalModel, AppError> {
    let models_root = state.effective_models_dir();
    let entry = tauri::async_runtime::spawn_blocking(move || {
        model_mgr::import_local_model(&models_root, spec)
    })
    .await
    .map_err(|e| AppError::new("MODEL_JOIN", e.to_string()))??;
    state.models.invalidate();
    Ok(entry)
}

/// Phase 11 — unload just the model(s) associated with a given
/// pipeline stage. Called by the frontend's "auto-unload after idle"
/// timer so freeing Whisper doesn't also unload a still-warm
/// llama.cpp context, and vice versa.
#[tauri::command]
pub async fn unload_stage_models(
    state: State<'_, AppState>,
    stage: String,
) -> Result<Vec<String>, AppError> {
    let mut released: Vec<String> = Vec::new();
    match stage.as_str() {
        "transcribe" => {
            if state.stt.unload().await.unwrap_or(false) {
                released.push("whisper".to_string());
            }
        }
        "translate" => {
            if state.translation.unload().await.unwrap_or(false) {
                released.push("translation".to_string());
            }
        }
        "tts" | "sync" => {
            // Sync piggybacks on the TTS engine when previewing —
            // both share the piper voice cache.
            match state.tts.unload_all().await {
                Ok(engines) => released.extend(engines.into_iter().map(|e| format!("tts:{e}"))),
                Err(err) => tracing::warn!(%err, "tts unload failed (stage-scoped)"),
            }
        }
        // Non-model stages: no-op.
        "extract_audio" | "mix" | "render" => {}
        other => {
            return Err(AppError::recoverable(
                "UNLOAD_UNKNOWN_STAGE",
                format!("unknown stage: {other}"),
            )
            .with_stage("models"));
        }
    }
    Ok(released)
}

#[tauri::command]
pub async fn unload_all_models(state: State<'_, AppState>) -> Result<Vec<String>, AppError> {
    // Phase 11 — every stage now exposes an explicit `unload` so
    // this command actually frees resident memory for Whisper +
    // llama.cpp + Piper in one shot. Errors are logged and skipped
    // so a slow-to-respond worker never blocks the "release
    // everything" surface.
    let mut released: Vec<String> = Vec::new();
    match state.stt.unload().await {
        Ok(true) => released.push("whisper".to_string()),
        Ok(false) => {}
        Err(err) => tracing::warn!(%err, "stt unload failed"),
    }
    match state.translation.unload().await {
        Ok(true) => released.push("translation".to_string()),
        Ok(false) => {}
        Err(err) => tracing::warn!(%err, "translation unload failed"),
    }
    match state.tts.unload_all().await {
        Ok(engines) => released.extend(engines.into_iter().map(|e| format!("tts:{e}"))),
        Err(err) => tracing::warn!(%err, "tts unload failed"),
    }
    Ok(released)
}

#[tauri::command]
pub async fn update_project_models(
    state: State<'_, AppState>,
    project_id: String,
    patch: ProjectModelPatch,
) -> Result<ProjectRecord, AppError> {
    Ok(state.projects.update_models(project_id, patch).await?)
}

// ---------- Phase 12 (Storage / logs / crash recovery surface) ----------

/// Named directory the UI can ask the OS to reveal (`open_path`) or
/// report a size for (`get_storage_stats`). Restricted to app-owned
/// roots so the frontend can't ask us to open, size, or clear an
/// arbitrary filesystem location.
#[derive(Debug, Clone, Copy)]
enum PathKind {
    Data,
    Config,
    Log,
    Cache,
    Projects,
    Models,
}

impl PathKind {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "data" => Self::Data,
            "config" => Self::Config,
            "log" | "logs" => Self::Log,
            "cache" => Self::Cache,
            "projects" => Self::Projects,
            "models" => Self::Models,
            _ => return None,
        })
    }

    fn resolve(self, state: &AppState) -> PathBuf {
        match self {
            Self::Data => state.paths.data_dir.clone(),
            Self::Config => state.paths.config_dir.clone(),
            Self::Log => state.paths.log_dir.clone(),
            Self::Cache => state.paths.cache_dir.clone(),
            Self::Projects => state.paths.projects_dir.clone(),
            Self::Models => state.effective_models_dir(),
        }
    }
}

fn parse_path_kind(kind: &str) -> Result<PathKind, AppError> {
    PathKind::parse(kind).ok_or_else(|| {
        AppError::recoverable(
            "STORAGE_UNKNOWN_PATH_KIND",
            format!("unknown path kind: {kind}"),
        )
        .with_stage("storage")
        .with_hint("Supported: data, config, log, cache, projects, models.")
    })
}

/// Phase 12 — reveal one of the app-owned directories in the OS
/// file explorer. Restricted to the known-safe roots above; the
/// frontend can't pass an arbitrary path.
#[tauri::command]
pub async fn open_app_path(state: State<'_, AppState>, kind: String) -> Result<String, AppError> {
    let target = parse_path_kind(&kind)?.resolve(&state);
    // Best-effort mkdir so opening a not-yet-populated dir (e.g.
    // logs right after install) still works.
    let _ = std::fs::create_dir_all(&target);
    let path_str = target.display().to_string();
    let spawn = spawn_open_command(&target);
    match spawn {
        Ok(_) => Ok(path_str),
        Err(err) => Err(AppError::recoverable(
            "STORAGE_OPEN_FAILED",
            format!("could not open {path_str}: {err}"),
        )
        .with_stage("storage")),
    }
}

fn spawn_open_command(target: &std::path::Path) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(target)
            .status()
            .map(|_| ())
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(target)
            .status()
            .map(|_| ())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(target)
            .status()
            .map(|_| ())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", unix)))]
    {
        let _ = target;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "no file-manager launcher on this platform",
        ))
    }
}

/// Storage snapshot the Settings › Storage panel renders. Sizes are
/// computed with a bounded recursive walk (Phase 12: `MAX_WALK_DEPTH`
/// = 6) so a pathological directory can't hang the UI.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageStats {
    pub data_dir: String,
    pub data_bytes: u64,
    pub cache_dir: String,
    pub cache_bytes: u64,
    pub log_dir: String,
    pub log_bytes: u64,
    pub projects_dir: String,
    pub projects_bytes: u64,
    pub models_dir: String,
    pub models_bytes: u64,
}

#[tauri::command]
pub async fn get_storage_stats(state: State<'_, AppState>) -> Result<StorageStats, AppError> {
    let paths = state.paths.clone();
    let models_dir = state.effective_models_dir();
    // Run the walks off the async runtime so the UI stays responsive
    // even on a slow disk.
    let stats = tokio::task::spawn_blocking(move || StorageStats {
        data_dir: paths.data_dir.display().to_string(),
        data_bytes: dir_size_bounded(&paths.data_dir),
        cache_dir: paths.cache_dir.display().to_string(),
        cache_bytes: dir_size_bounded(&paths.cache_dir),
        log_dir: paths.log_dir.display().to_string(),
        log_bytes: dir_size_bounded(&paths.log_dir),
        projects_dir: paths.projects_dir.display().to_string(),
        projects_bytes: dir_size_bounded(&paths.projects_dir),
        models_dir: models_dir.display().to_string(),
        models_bytes: dir_size_bounded(&models_dir),
    })
    .await
    .map_err(|e| AppError::new("STORAGE_JOIN", e.to_string()).with_stage("storage"))?;
    Ok(stats)
}

const MAX_WALK_DEPTH: usize = 6;

fn dir_size_bounded(root: &std::path::Path) -> u64 {
    if !root.exists() {
        return 0;
    }
    let mut total: u64 = 0;
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > MAX_WALK_DEPTH {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_file() {
                if let Ok(meta) = entry.metadata() {
                    total = total.saturating_add(meta.len());
                }
            } else if ft.is_dir() {
                stack.push((entry.path(), depth + 1));
            }
        }
    }
    total
}

/// Phase 12 — clear the app cache directory. Only removes files that
/// live under `<cache>/…`; per-project data, models and outputs
/// (which live under `<data>/…`) are never touched. The log
/// subdirectory is also preserved so the tracing appender's file
/// handle stays valid — use `clear_logs` for that.
#[tauri::command]
pub async fn clear_cache(state: State<'_, AppState>) -> Result<u64, AppError> {
    let cache = state.paths.cache_dir.clone();
    let log = state.paths.log_dir.clone();
    let bytes = tokio::task::spawn_blocking(move || {
        let mut freed: u64 = 0;
        let entries = match std::fs::read_dir(&cache) {
            Ok(e) => e,
            Err(_) => return 0u64,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // Never touch the log subdir here — logs are handled by
            // `clear_logs` so the active file handle stays valid.
            if path == log {
                continue;
            }
            let size = entry
                .metadata()
                .map(|m| if m.is_file() { m.len() } else { 0 })
                .unwrap_or(0);
            let ft = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            let removed = if ft.is_dir() {
                std::fs::remove_dir_all(&path).is_ok()
            } else {
                std::fs::remove_file(&path).is_ok()
            };
            if removed {
                freed = freed.saturating_add(size);
            }
        }
        freed
    })
    .await
    .map_err(|e| AppError::new("STORAGE_JOIN", e.to_string()).with_stage("storage"))?;
    Ok(bytes)
}

/// Phase 12 — truncate active log files and prune rotations older
/// than the retention window. Files are truncated rather than
/// deleted so the running tracing appender keeps writing to the
/// same file descriptor.
#[tauri::command]
pub async fn clear_logs(state: State<'_, AppState>) -> Result<usize, AppError> {
    let log_dir = state.paths.log_dir.clone();
    let cleared = tokio::task::spawn_blocking(move || {
        crate::logging::prune_old_logs(&log_dir, 0);
        crate::logging::clear_active_logs(&log_dir).unwrap_or(0)
    })
    .await
    .map_err(|e| AppError::new("STORAGE_JOIN", e.to_string()).with_stage("storage"))?;
    Ok(cleared)
}

/// Phase 12 — the Dashboard's "Something went wrong on the last
/// run" surface asks Rust for the terminal jobs the crash-reaper
/// (`reap_orphans`, called at startup) marked as
/// `error_code = JOB_ORPHANED`. The UI shows them so the user
/// can retry from where they left off rather than wondering why
/// half a translation vanished.
#[tauri::command]
pub async fn list_orphaned_jobs(state: State<'_, AppState>) -> Result<Vec<JobSnapshot>, AppError> {
    let db = state.db.clone();
    Ok(db.run(crate::jobs::JobsRepo::list_orphaned).await?)
}

// ---------- Phase 13 (optional YouTube integration) ----------

#[tauri::command]
pub async fn get_youtube_state(
    state: State<'_, AppState>,
) -> Result<YouTubeConnectionState, AppError> {
    let offline = state.settings.snapshot().offline_mode;
    Ok(state.youtube.connection_state(offline))
}

#[tauri::command]
pub async fn connect_youtube(
    state: State<'_, AppState>,
) -> Result<YouTubeConnectionState, AppError> {
    if state.settings.snapshot().offline_mode {
        return Err(AppError::from(YouTubeError::Offline));
    }
    Ok(state.youtube.connect().await?)
}

#[tauri::command]
pub async fn disconnect_youtube(
    state: State<'_, AppState>,
) -> Result<YouTubeConnectionState, AppError> {
    let offline = state.settings.snapshot().offline_mode;
    Ok(state.youtube.disconnect(offline)?)
}

#[tauri::command]
pub async fn list_youtube_accounts(
    state: State<'_, AppState>,
) -> Result<Vec<YouTubeAccount>, AppError> {
    Ok(state.youtube.list_accounts())
}

#[tauri::command]
pub async fn select_youtube_account(
    state: State<'_, AppState>,
    account_id: String,
) -> Result<YouTubeConnectionState, AppError> {
    let offline = state.settings.snapshot().offline_mode;
    Ok(state.youtube.select_account(&account_id, offline)?)
}

#[tauri::command]
pub async fn list_youtube_playlists(
    state: State<'_, AppState>,
) -> Result<Vec<YouTubePlaylist>, AppError> {
    if state.settings.snapshot().offline_mode {
        return Err(AppError::from(YouTubeError::Offline));
    }
    Ok(state.youtube.list_playlists().await?)
}

#[tauri::command]
pub async fn start_youtube_upload(
    state: State<'_, AppState>,
    project_id: String,
    metadata: YouTubeVideoMetadata,
    options: Option<YouTubePublishOptions>,
) -> Result<YouTubeUploadSnapshot, AppError> {
    if state.settings.snapshot().offline_mode {
        return Err(AppError::from(YouTubeError::Offline));
    }
    // The frontend never supplies an unrestricted file path. Resolve the
    // current, validated Phase 9 render from the selected project instead.
    let manifest = state
        .render
        .get_manifest(project_id.clone())
        .await?
        .ok_or_else(|| YouTubeError::InvalidVideo("No rendered movie is available.".into()))?;
    let summary = state
        .render
        .get_summary(project_id.clone(), manifest.settings.clone())
        .await?
        .ok_or_else(|| YouTubeError::InvalidVideo("No rendered movie is available.".into()))?;
    if summary.status != RenderStatus::Ready {
        return Err(AppError::from(YouTubeError::InvalidVideo(
            "The current project render is missing or stale. Render it again before publishing."
                .into(),
        )));
    }
    let rendered = manifest.current.ok_or_else(|| {
        YouTubeError::InvalidVideo("The project has no completed render to upload.".into())
    })?;
    if summary.absolute_path.as_deref() != Some(rendered.file_absolute.as_str()) {
        return Err(AppError::from(YouTubeError::InvalidVideo(
            "The ready render summary does not match the current render manifest.".into(),
        )));
    }
    let project = state.projects.open(project_id.clone()).await?;
    let options = options.unwrap_or_default();
    let subtitle_doc = if options.publish_translated_subtitles
        || options.publish_original_subtitles
    {
        state.subtitles.get_doc(project_id.clone()).await?
    } else {
        None
    };
    Ok(state
        .youtube
        .start_upload(
            project_id,
            PathBuf::from(project.root_path),
            PathBuf::from(rendered.file_absolute),
            metadata,
            options,
            subtitle_doc,
        )?)
}

#[tauri::command]
pub async fn list_youtube_uploads(
    state: State<'_, AppState>,
) -> Result<Vec<YouTubeUploadSnapshot>, AppError> {
    Ok(state.youtube.list_uploads())
}

#[tauri::command]
pub async fn cancel_youtube_upload(
    state: State<'_, AppState>,
    upload_id: String,
) -> Result<(), AppError> {
    Ok(state.youtube.cancel_upload(&upload_id)?)
}

#[tauri::command]
pub async fn retry_youtube_upload(
    state: State<'_, AppState>,
    upload_id: String,
) -> Result<YouTubeUploadSnapshot, AppError> {
    if state.settings.snapshot().offline_mode {
        return Err(AppError::from(YouTubeError::Offline));
    }
    Ok(state.youtube.retry_upload(&upload_id)?)
}

#[tauri::command]
pub async fn open_youtube_video(
    state: State<'_, AppState>,
    video_id: String,
) -> Result<(), AppError> {
    Ok(state.youtube.open_video(&video_id)?)
}

#[tauri::command]
pub async fn generate_youtube_thumbnail(
    state: State<'_, AppState>,
    project_id: String,
    time_seconds: f64,
) -> Result<YouTubeThumbnailResult, AppError> {
    let project = state.projects.open(project_id.clone()).await?;
    let manifest = state
        .render
        .get_manifest(project_id)
        .await?
        .ok_or_else(|| YouTubeError::InvalidVideo("No rendered movie is available.".into()))?;
    let rendered = manifest.current.ok_or_else(|| {
        YouTubeError::InvalidVideo("The project has no completed render.".into())
    })?;
    Ok(state
        .youtube
        .generate_thumbnail(
            &PathBuf::from(project.root_path),
            &PathBuf::from(rendered.file_absolute),
            time_seconds,
        )
        .await?)
}

#[tauri::command]
pub async fn validate_youtube_thumbnail(
    state: State<'_, AppState>,
    path: String,
) -> Result<YouTubeThumbnailResult, AppError> {
    Ok(state
        .youtube
        .validate_thumbnail_selection(&PathBuf::from(path))?)
}

#[tauri::command]
pub async fn list_youtube_history(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Vec<YouTubePublishingHistoryEntry>, AppError> {
    let project = state.projects.open(project_id).await?;
    Ok(state
        .youtube
        .list_history(&PathBuf::from(project.root_path))?)
}

/// Phase 11 — cheap "did the file move?" check that reuses the
/// fingerprint the importer wrote to `ProjectRecord`. Only if the
/// stat has drifted do we hit the disk to recompute the sentinel
/// window hash. `None` in, `None` out (no source imported yet).
fn reuse_or_recompute_fingerprint(
    path: &PathBuf,
    known_hash: Option<&str>,
    known_size: Option<i64>,
    known_modified: Option<&chrono::DateTime<chrono::Utc>>,
) -> Option<crate::media::SourceFingerprint> {
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    let modified: chrono::DateTime<chrono::Utc> = meta
        .modified()
        .ok()
        .map(chrono::DateTime::<chrono::Utc>::from)?;

    // Fast path: stat still matches the record and we've got the
    // stored hash. This covers 99% of `get_project_media` calls.
    if let (Some(hash), Some(stored_size), Some(stored_mtime)) =
        (known_hash, known_size, known_modified)
    {
        if stored_size as u64 == size && *stored_mtime == modified {
            return Some(crate::media::SourceFingerprint {
                hash: hash.to_string(),
                size_bytes: size,
                modified_at: modified,
            });
        }
    }

    // Slow path: the file's been replaced (mtime/size drift), so the
    // cached hash is meaningless. Recompute the cheap sentinel-window
    // hash and let the caller decide what to invalidate.
    crate::media::fingerprint::fingerprint_file(path).ok()
}
