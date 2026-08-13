//! Local Movie Translator — Tauri backend entry point.
//!
//! Responsibilities of this crate (see ARCHITECTURE.md §2):
//!   * Application lifecycle & window management
//!   * OS-appropriate filesystem paths (data/config/cache/logs)
//!   * SQLite persistence (projects, jobs, settings, subtitles)
//!   * Project on-disk layout creation and validation
//!   * Python worker supervisor + JSON-RPC over stdio
//!   * FFmpeg process lifecycle (probe + audio extraction)   [Phase 2]
//!   * Typed IPC surface exposed to the React frontend
//!
//! This crate must never perform AI inference, subtitle math, or video work.

pub mod app;
pub mod audio;
pub mod commands;
pub mod config;
pub mod db;
pub mod errors;
pub mod ffmpeg;
pub mod ipc;
pub mod integrations;
pub mod jobs;
pub mod logging;
pub mod media;
pub mod media_server;
pub mod mix;
pub mod models;
pub mod paths;
pub mod projects;
pub mod render;
pub mod stt;
pub mod subtitles;
pub mod sync;
pub mod translation;
pub mod tts;
pub mod worker;

use tauri::Manager;

use crate::app::AppState;

/// Public entrypoint invoked from `main.rs` (and from `mobile` in the future).
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let paths = paths::AppPaths::detect().expect("failed to detect app paths");
    let _log_guard = logging::init(&paths).expect("failed to initialise logging");

    tracing::info!(
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        data_dir = %paths.data_dir.display(),
        "starting local-movie-translator"
    );

    // Phase 12 — we install our own JSON `tracing_subscriber` above
    // (`logging::init`), so we intentionally do NOT register
    // `tauri_plugin_log` here. Both compete for the process-wide
    // `log` dispatcher and only one can win; keeping our subscriber
    // means logs land in the rotated JSON file at `<cache>/logs/`
    // where the Settings › Storage & Logs panel expects them.
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            let handle = app.handle().clone();
            // Phase 12 — `AppState::bootstrap` schedules background
            // `tokio::spawn` tasks (worker supervisor, ffmpeg probe,
            // orphan `.tmp` sweep). Tauri's `setup` closure runs on
            // the main thread outside any tokio context, so we
            // enter Tauri's shared async runtime before bootstrapping
            // to give those `tokio::spawn` calls a reactor to attach
            // to.
            let state = tauri::async_runtime::block_on(async {
                AppState::bootstrap(handle, paths.clone())
            })
            .map_err(|e| Box::new(std::io::Error::other(e.to_string())))?;
            app.manage(state);

            // The video preview loads over loopback HTTP rather than a
            // custom URI scheme, which WebKit refuses to play media from.
            // A failure here costs the preview, not the app, so it is
            // logged instead of aborting startup.
            // The URL is deliberately not logged: it carries the access
            // token. `media_server` logs the port on its own.
            tauri::async_runtime::block_on(async {
                if let Err(err) = media_server::start().await {
                    tracing::error!(%err, "media server failed to start");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_app_info,
            commands::get_settings,
            commands::update_settings,
            commands::list_projects,
            commands::create_project,
            commands::open_project,
            commands::delete_project,
            commands::worker_status,
            commands::worker_ping,
            commands::worker_env_info,
            // Phase 2
            commands::get_ffmpeg_availability,
            commands::refresh_ffmpeg,
            commands::probe_media,
            commands::import_media,
            commands::get_project_media,
            commands::extract_audio,
            commands::cancel_job,
            commands::list_active_jobs,
            // Phase 3
            commands::get_stt_env,
            commands::list_whisper_models,
            commands::download_whisper_model,
            commands::transcribe,
            commands::get_project_transcript,
            // Phase 4
            commands::get_translation_env,
            commands::list_translation_models,
            commands::list_recommended_translation_presets,
            commands::download_translation_model,
            commands::translate,
            commands::get_project_translation,
            commands::get_project_translation_doc,
            commands::update_translation_segment,
            // Phase 5
            commands::get_project_subtitles,
            commands::get_project_subtitles_doc,
            commands::rebuild_project_subtitles,
            commands::update_subtitle_segment,
            commands::assign_subtitle_voice_to_speaker,
            commands::add_subtitle_segment,
            commands::delete_subtitle_segment,
            commands::split_subtitle_segment,
            commands::merge_subtitle_segment,
            commands::clear_subtitle_dirty,
            commands::import_subtitles,
            commands::export_subtitles,
            // Phase 6
            commands::get_tts_env,
            commands::list_tts_voices,
            commands::create_tts_voice_profile,
            commands::list_recommended_tts_voices,
            commands::download_tts_voice,
            commands::get_project_tts_summary,
            commands::get_project_tts_manifest,
            commands::preview_tts_segment,
            commands::generate_tts,
            // Phase 7
            commands::get_sync_env,
            commands::get_project_sync_summary,
            commands::get_project_sync_manifest,
            commands::preview_sync_segment,
            commands::apply_sync,
            // Phase 8
            commands::get_mix_env,
            commands::get_project_mix_summary,
            commands::get_project_mix_manifest,
            commands::get_project_mix_preview,
            commands::apply_mix,
            // Phase 9
            commands::get_render_env,
            commands::get_project_render_summary,
            commands::get_project_render_manifest,
            commands::apply_render,
            // Phase 10 — Local Model Manager
            commands::list_local_models,
            commands::rescan_local_models,
            commands::import_local_model,
            commands::get_model_directory,
            commands::set_model_directory,
            commands::unload_all_models,
            commands::unload_stage_models,
            commands::get_runtime_stats,
            commands::update_project_models,
            // Phase 12 — production/storage surface
            commands::open_app_path,
            commands::get_storage_stats,
            commands::clear_cache,
            commands::clear_logs,
            commands::list_orphaned_jobs,
            // Phase 13 — optional YouTube integration
            commands::get_youtube_state,
            commands::connect_youtube,
            commands::disconnect_youtube,
            commands::list_youtube_accounts,
            commands::select_youtube_account,
            commands::list_youtube_playlists,
            commands::start_youtube_upload,
            commands::list_youtube_uploads,
            commands::cancel_youtube_upload,
            commands::retry_youtube_upload,
            commands::open_youtube_video,
            commands::generate_youtube_thumbnail,
            commands::validate_youtube_thumbnail,
            commands::list_youtube_history,
            commands::get_media_base_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
