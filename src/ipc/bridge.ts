// Thin, typed wrapper around Tauri IPC. All frontend ↔ backend traffic
// goes through this file so we can add tracing/mocking in one place.

import { invoke as tauriInvoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import {
  type AppError,
  type AppInfo,
  type AppSettings,
  type AppSettingsPatch,
  type CreateProjectInput,
  type EnvInfo,
  type ExportKind,
  type ExportSubtitlesResult,
  type ExtractionStart,
  type FfmpegAvailability,
  type GenerateRequest,
  type ImportMediaInput,
  type ImportMediaResult,
  type ImportModelSpec,
  type ImportSubtitlesResult,
  type JobProgressEvent,
  type JobSnapshot,
  type LocalModel,
  type ModelDirectoryInfo,
  type ProjectModelPatch,
  type MixEnv,
  type MixGenerateStart,
  type MixManifest,
  type MixRequest,
  type MixSettings,
  type MixSummary,
  type PingResponse,
  type OutputFormat,
  type PreviewMixResult,
  type PreviewResult,
  type PreviewSyncResult,
  type RenderEnv,
  type RenderGenerateStart,
  type RenderManifest,
  type RenderRequest,
  type RenderSettings,
  type RenderSummary,
  type RuntimeStats,
  type AppPathKind,
  type StorageStats,
  type Project,
  type ProjectMediaState,
  type ProjectSummary,
  type SttEnv,
  type SttOptions,
  type SubtitleDoc,
  type SubtitleFormat,
  type SubtitleSegmentPatch,
  type SubtitleSummary,
  type SyncEnv,
  type SyncGenerateStart,
  type SyncManifest,
  type SyncRequest,
  type SyncSegmentCompletedEvent,
  type SyncSettings,
  type SyncSummary,
  type TranscribeStart,
  type TranscriptSummary,
  type TranslateOptions,
  type TranslateStart,
  type TranslationChunkCompletedEvent,
  type TranslationDoc,
  type TranslationEnv,
  type TranslationModelInfo,
  type TranslationRecommendedPreset,
  type TranslationSummary,
  type TtsEnv,
  type TtsGenerateStart,
  type TtsManifest,
  type TtsRecommendedVoicePreset,
  type TtsSegmentCompletedEvent,
  type TtsSettings,
  type TtsSummary,
  type VideoMetadata,
  type VoiceInfo,
  type WhisperModelInfo,
  type WorkerStatus,
  type YouTubeConnectionState,
  type YouTubeAccount,
  type YouTubePlaylist,
  type YouTubePublishingHistoryEntry,
  type YouTubePublishOptions,
  type YouTubeThumbnailResult,
  type YouTubeUploadProgressEvent,
  type YouTubeUploadSnapshot,
  type YouTubeVideoMetadata,
  isAppError,
} from "./types";

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return (await tauriInvoke<T>(cmd, args)) as T;
  } catch (raw) {
    if (isAppError(raw)) throw raw;
    const err: AppError = {
      code: "IPC_UNKNOWN",
      stage: null,
      message: typeof raw === "string" ? raw : JSON.stringify(raw),
      recoverable: false,
      hint: null,
    };
    throw err;
  }
}

export const api = {
  getAppInfo: () => invoke<AppInfo>("get_app_info"),

  getSettings: () => invoke<AppSettings>("get_settings"),
  updateSettings: (patch: AppSettingsPatch) =>
    invoke<AppSettings>("update_settings", { patch }),

  listProjects: () => invoke<ProjectSummary[]>("list_projects"),
  createProject: (input: CreateProjectInput) =>
    invoke<ProjectSummary>("create_project", { input }),
  openProject: (id: string) => invoke<Project>("open_project", { id }),
  deleteProject: (id: string) => invoke<void>("delete_project", { id }),

  workerStatus: () => invoke<WorkerStatus>("worker_status"),
  workerPing: () => invoke<PingResponse>("worker_ping"),
  workerEnvInfo: () => invoke<EnvInfo>("worker_env_info"),

  // Phase 2
  getFfmpegAvailability: () =>
    invoke<FfmpegAvailability>("get_ffmpeg_availability"),
  refreshFfmpeg: () => invoke<FfmpegAvailability>("refresh_ffmpeg"),

  probeMedia: (path: string) => invoke<VideoMetadata>("probe_media", { path }),
  importMedia: (input: ImportMediaInput) =>
    invoke<ImportMediaResult>("import_media", { input }),
  getProjectMedia: (projectId: string) =>
    invoke<ProjectMediaState>("get_project_media", { projectId }),
  extractAudio: (projectId: string) =>
    invoke<ExtractionStart>("extract_audio", { projectId }),
  cancelJob: (jobId: string) => invoke<void>("cancel_job", { jobId }),
  listActiveJobs: (projectId: string) =>
    invoke<JobSnapshot[]>("list_active_jobs", { projectId }),

  // Phase 3 (Speech recognition)
  getSttEnv: () => invoke<SttEnv>("get_stt_env"),
  listWhisperModels: () => invoke<WhisperModelInfo[]>("list_whisper_models"),
  downloadWhisperModel: (name: string) =>
    invoke<JobSnapshot>("download_whisper_model", { name }),
  transcribe: (projectId: string, options?: SttOptions) =>
    invoke<TranscribeStart>("transcribe", { projectId, options }),
  getProjectTranscript: (projectId: string) =>
    invoke<TranscriptSummary | null>("get_project_transcript", { projectId }),

  // Phase 4 (Local LLM translation)
  getTranslationEnv: () => invoke<TranslationEnv>("get_translation_env"),
  listTranslationModels: () =>
    invoke<TranslationModelInfo[]>("list_translation_models"),
  // Phase 12 — mirror of `listWhisperModels` / `downloadWhisperModel`.
  // The presets list is small and static (~3 rows sourced from
  // `translation/registry.py::_RECOMMENDED_MODELS`); the download
  // command spawns a background job and returns the snapshot
  // immediately so the UI can render a progress bar.
  listRecommendedTranslationPresets: () =>
    invoke<TranslationRecommendedPreset[]>(
      "list_recommended_translation_presets",
    ),
  downloadTranslationModel: (preset: string) =>
    invoke<JobSnapshot>("download_translation_model", { preset }),
  translate: (projectId: string, options: TranslateOptions) =>
    invoke<TranslateStart>("translate", { projectId, options }),
  getProjectTranslation: (projectId: string) =>
    invoke<TranslationSummary | null>("get_project_translation", { projectId }),
  getProjectTranslationDoc: (projectId: string) =>
    invoke<TranslationDoc | null>("get_project_translation_doc", { projectId }),
  updateTranslationSegment: (
    projectId: string,
    segmentId: number,
    translation: string,
  ) =>
    invoke<TranslationSummary>("update_translation_segment", {
      projectId,
      segmentId,
      translation,
    }),

  // Phase 5 (Subtitle editor)
  getProjectSubtitles: (projectId: string) =>
    invoke<SubtitleSummary | null>("get_project_subtitles", { projectId }),
  getProjectSubtitlesDoc: (projectId: string) =>
    invoke<SubtitleDoc | null>("get_project_subtitles_doc", { projectId }),
  rebuildProjectSubtitles: (projectId: string) =>
    invoke<SubtitleDoc>("rebuild_project_subtitles", { projectId }),
  updateSubtitleSegment: (
    projectId: string,
    segmentId: number,
    patch: SubtitleSegmentPatch,
  ) =>
    invoke<SubtitleDoc>("update_subtitle_segment", {
      projectId,
      segmentId,
      patch,
    }),
  addSubtitleSegment: (
    projectId: string,
    afterId: number | null,
    start: number,
    end: number,
  ) =>
    invoke<SubtitleDoc>("add_subtitle_segment", {
      projectId,
      afterId,
      start,
      end,
    }),
  deleteSubtitleSegment: (projectId: string, segmentId: number) =>
    invoke<SubtitleDoc>("delete_subtitle_segment", { projectId, segmentId }),
  splitSubtitleSegment: (
    projectId: string,
    segmentId: number,
    splitTime: number,
  ) =>
    invoke<SubtitleDoc>("split_subtitle_segment", {
      projectId,
      segmentId,
      splitTime,
    }),
  mergeSubtitleSegment: (projectId: string, segmentId: number) =>
    invoke<SubtitleDoc>("merge_subtitle_segment", { projectId, segmentId }),
  clearSubtitleDirty: (projectId: string) =>
    invoke<SubtitleDoc>("clear_subtitle_dirty", { projectId }),
  importSubtitles: (projectId: string, path: string) =>
    invoke<ImportSubtitlesResult>("import_subtitles", { projectId, path }),
  exportSubtitles: (
    projectId: string,
    path: string,
    format: SubtitleFormat,
    kind: ExportKind = "translated",
  ) =>
    invoke<ExportSubtitlesResult>("export_subtitles", {
      projectId,
      path,
      format,
      kind,
    }),

  // Phase 6 (Local TTS / AI dubbing)
  getTtsEnv: () => invoke<TtsEnv>("get_tts_env"),
  listTtsVoices: () => invoke<VoiceInfo[]>("list_tts_voices"),
  // Phase 12 — TTS voice auto-download, mirrors the STT +
  // translation download flow. Presets live in the worker at
  // ``tts/registry.py::_RECOMMENDED_VOICES``; the download itself
  // returns a job snapshot and streams progress via
  // ``job://update`` + ``job://progress``.
  listRecommendedTtsVoices: () =>
    invoke<TtsRecommendedVoicePreset[]>("list_recommended_tts_voices"),
  downloadTtsVoice: (preset: string) =>
    invoke<JobSnapshot>("download_tts_voice", { preset }),
  getProjectTtsSummary: (
    projectId: string,
    engine: string,
    defaultVoiceId: string,
    settings: TtsSettings,
  ) =>
    invoke<TtsSummary | null>("get_project_tts_summary", {
      projectId,
      engine,
      defaultVoiceId,
      settings,
    }),
  getProjectTtsManifest: (projectId: string) =>
    invoke<TtsManifest | null>("get_project_tts_manifest", { projectId }),
  previewTtsSegment: (
    projectId: string,
    segmentId: number,
    voiceId: string | null,
    engine: string | null,
    settings: TtsSettings | null,
  ) =>
    invoke<PreviewResult>("preview_tts_segment", {
      projectId,
      segmentId,
      voiceId,
      engine,
      settings,
    }),
  generateTts: (projectId: string, request: GenerateRequest) =>
    invoke<TtsGenerateStart>("generate_tts", { projectId, request }),

  // Phase 7 (Voice synchronisation)
  getSyncEnv: () => invoke<SyncEnv>("get_sync_env"),
  getProjectSyncSummary: (projectId: string, settings: SyncSettings) =>
    invoke<SyncSummary | null>("get_project_sync_summary", {
      projectId,
      settings,
    }),
  getProjectSyncManifest: (projectId: string) =>
    invoke<SyncManifest | null>("get_project_sync_manifest", { projectId }),
  previewSyncSegment: (
    projectId: string,
    segmentId: number,
    settings: SyncSettings | null,
  ) =>
    invoke<PreviewSyncResult>("preview_sync_segment", {
      projectId,
      segmentId,
      settings,
    }),
  applySync: (projectId: string, request: SyncRequest) =>
    invoke<SyncGenerateStart>("apply_sync", { projectId, request }),

  // Phase 8 (Audio mixing)
  getMixEnv: () => invoke<MixEnv>("get_mix_env"),
  getProjectMixSummary: (projectId: string, settings: MixSettings) =>
    invoke<MixSummary | null>("get_project_mix_summary", {
      projectId,
      settings,
    }),
  getProjectMixManifest: (projectId: string) =>
    invoke<MixManifest | null>("get_project_mix_manifest", { projectId }),
  getProjectMixPreview: (projectId: string) =>
    invoke<PreviewMixResult | null>("get_project_mix_preview", { projectId }),
  applyMix: (projectId: string, request: MixRequest) =>
    invoke<MixGenerateStart>("apply_mix", { projectId, request }),

  // Phase 9 (Final video rendering)
  getRenderEnv: () => invoke<RenderEnv>("get_render_env"),
  getProjectRenderSummary: (projectId: string, settings: RenderSettings) =>
    invoke<RenderSummary | null>("get_project_render_summary", {
      projectId,
      settings,
    }),
  getProjectRenderManifest: (projectId: string) =>
    invoke<RenderManifest | null>("get_project_render_manifest", {
      projectId,
    }),
  applyRender: (projectId: string, request: RenderRequest) =>
    invoke<RenderGenerateStart>("apply_render", { projectId, request }),

  // Phase 10 (Local Model Manager)
  listLocalModels: () => invoke<LocalModel[]>("list_local_models"),
  rescanLocalModels: () => invoke<LocalModel[]>("rescan_local_models"),
  getModelDirectory: () => invoke<ModelDirectoryInfo>("get_model_directory"),
  setModelDirectory: (path: string | null) =>
    invoke<ModelDirectoryInfo>("set_model_directory", { path }),
  importLocalModel: (spec: ImportModelSpec) =>
    invoke<LocalModel>("import_local_model", { spec }),
  unloadAllModels: () => invoke<string[]>("unload_all_models"),
  /** Phase 11 — release just the model(s) for a given pipeline stage.
   *  Used by the frontend's per-stage auto-unload timer. */
  unloadStageModels: (stage: string) =>
    invoke<string[]>("unload_stage_models", { stage }),
  /** Phase 11 — lightweight runtime snapshot for the resource monitor
   *  strip. Safe to poll every 2–3 seconds. */
  getRuntimeStats: () => invoke<RuntimeStats>("get_runtime_stats"),

  // ---------- Phase 12 — Storage / crash recovery surface ----------

  /** Reveal an app-owned directory in the OS file explorer. `kind`
   *  is validated by Rust; passing an unknown value returns a
   *  `STORAGE_UNKNOWN_PATH_KIND` error. */
  openAppPath: (kind: AppPathKind) =>
    invoke<string>("open_app_path", { kind }),
  /** Compute on-disk sizes for the app-owned directories. Bounded
   *  walk — never blocks on pathological trees. */
  getStorageStats: () => invoke<StorageStats>("get_storage_stats"),
  /** Best-effort cache cleanup. Preserves logs, projects, models
   *  and all other data. Returns the number of bytes freed. */
  clearCache: () => invoke<number>("clear_cache"),
  /** Truncate active log files and prune rotations. Returns the
   *  number of active log files that were truncated. */
  clearLogs: () => invoke<number>("clear_logs"),
  /** Jobs the last startup sweep marked as interrupted by a crash
   *  or forced restart. The Dashboard surfaces this so the user
   *  can resume the affected project. */
  listOrphanedJobs: () => invoke<JobSnapshot[]>("list_orphaned_jobs"),
  updateProjectModels: (projectId: string, patch: ProjectModelPatch) =>
    invoke<Project>("update_project_models", { projectId, patch }),

  // Phase 13 — OAuth and uploads stay in the Rust host. Tokens and
  // resumable session URLs never cross IPC into React.
  getYouTubeState: () =>
    invoke<YouTubeConnectionState>("get_youtube_state"),
  connectYouTube: () =>
    invoke<YouTubeConnectionState>("connect_youtube"),
  disconnectYouTube: () =>
    invoke<YouTubeConnectionState>("disconnect_youtube"),
  listYouTubeAccounts: () =>
    invoke<YouTubeAccount[]>("list_youtube_accounts"),
  selectYouTubeAccount: (accountId: string) =>
    invoke<YouTubeConnectionState>("select_youtube_account", { accountId }),
  listYouTubePlaylists: () =>
    invoke<YouTubePlaylist[]>("list_youtube_playlists"),
  startYouTubeUpload: (
    projectId: string,
    metadata: YouTubeVideoMetadata,
    options?: YouTubePublishOptions,
  ) =>
    invoke<YouTubeUploadSnapshot>("start_youtube_upload", {
      projectId,
      metadata,
      options,
    }),
  listYouTubeUploads: () =>
    invoke<YouTubeUploadSnapshot[]>("list_youtube_uploads"),
  cancelYouTubeUpload: (uploadId: string) =>
    invoke<void>("cancel_youtube_upload", { uploadId }),
  retryYouTubeUpload: (uploadId: string) =>
    invoke<YouTubeUploadSnapshot>("retry_youtube_upload", { uploadId }),
  openYouTubeVideo: (videoId: string) =>
    invoke<void>("open_youtube_video", { videoId }),
  generateYouTubeThumbnail: (projectId: string, timeSeconds: number) =>
    invoke<YouTubeThumbnailResult>("generate_youtube_thumbnail", {
      projectId,
      timeSeconds,
    }),
  validateYouTubeThumbnail: (path: string) =>
    invoke<YouTubeThumbnailResult>("validate_youtube_thumbnail", { path }),
  listYouTubeHistory: (projectId: string) =>
    invoke<YouTubePublishingHistoryEntry[]>("list_youtube_history", {
      projectId,
    }),
};

/**
 * Native "save final movie" dialog. Returns the absolute path or null.
 */
export async function pickRenderOutputPath(
  defaultName: string,
  format: OutputFormat,
): Promise<string | null> {
  const result = await saveDialog({
    defaultPath: defaultName,
    filters: [
      { name: format.toUpperCase(), extensions: [format] },
    ],
  });
  return typeof result === "string" ? result : null;
}

export function onWorkerStatus(
  cb: (status: WorkerStatus) => void,
): Promise<UnlistenFn> {
  return listen<WorkerStatus>("worker://status", (evt) => cb(evt.payload));
}

export function onJobProgress(
  cb: (evt: JobProgressEvent) => void,
): Promise<UnlistenFn> {
  return listen<JobProgressEvent>("job://progress", (evt) => cb(evt.payload));
}

export function onJobUpdate(
  cb: (snap: JobSnapshot) => void,
): Promise<UnlistenFn> {
  return listen<JobSnapshot>("job://update", (evt) => cb(evt.payload));
}

export function onTranslationChunkCompleted(
  cb: (evt: TranslationChunkCompletedEvent) => void,
): Promise<UnlistenFn> {
  return listen<TranslationChunkCompletedEvent>(
    "translation://chunk_completed",
    (evt) => cb(evt.payload),
  );
}

export function onTtsSegmentCompleted(
  cb: (evt: TtsSegmentCompletedEvent) => void,
): Promise<UnlistenFn> {
  return listen<TtsSegmentCompletedEvent>(
    "tts://segment_completed",
    (evt) => cb(evt.payload),
  );
}

export function onSyncSegmentCompleted(
  cb: (evt: SyncSegmentCompletedEvent) => void,
): Promise<UnlistenFn> {
  return listen<SyncSegmentCompletedEvent>(
    "sync://segment_completed",
    (evt) => cb(evt.payload),
  );
}

export function onYouTubeUpload(
  cb: (upload: YouTubeUploadProgressEvent["upload"]) => void,
): Promise<UnlistenFn> {
  return listen<YouTubeUploadProgressEvent>("youtube://upload", (evt) =>
    cb(evt.payload.upload),
  );
}

/**
 * Open the native file picker constrained to the supported video
 * extensions. Returns the selected absolute path, or `null` if the
 * user cancels.
 */
export async function pickMediaFile(): Promise<string | null> {
  const result = await openDialog({
    multiple: false,
    filters: [
      {
        name: "Video",
        extensions: ["mp4", "m4v", "mkv", "mov", "avi", "webm"],
      },
    ],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : (result as { path: string }).path;
}

/** Explicit local image selection for a YouTube custom thumbnail. */
export async function pickYouTubeThumbnail(): Promise<string | null> {
  const result = await openDialog({
    multiple: false,
    filters: [
      {
        name: "Thumbnail image",
        extensions: ["jpg", "jpeg", "png", "webp"],
      },
    ],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : (result as { path: string }).path;
}

/**
 * Convert an absolute local path into an `asset:` URL that the
 * WebView's `<video>` / `<img>` elements can load without shuffling
 * bytes through IPC. Requires the asset protocol to be enabled in
 * `tauri.conf.json`.
 */
export function assetUrl(absolutePath: string): string {
  return convertFileSrc(absolutePath);
}

/**
 * Base URL of the loopback media server, cached for the session.
 *
 * Held in a module variable because {@link mediaUrl} is called during
 * render and cannot await anything.
 */
let mediaBaseUrl: string | null = null;

async function probeMediaElement(baseUrl: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const audio = new Audio();
    const cleanup = () => {
      window.clearTimeout(timer);
      audio.onloadedmetadata = null;
      audio.onerror = null;
      audio.removeAttribute("src");
      audio.load();
    };
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("media element probe timed out"));
    }, 5_000);
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      cleanup();
      resolve();
    };
    audio.onerror = () => {
      const code = audio.error?.code ?? "unknown";
      cleanup();
      reject(new Error(`media element probe failed with code ${code}`));
    };
    audio.src = `${baseUrl}&probe=1`;
    audio.load();
  });
}

/**
 * Fetch the media server's base URL once, at startup.
 *
 * The URL carries a token minted per run, so it cannot be hard-coded.
 * Failure is not fatal: only playback depends on it.
 */
export async function initMediaBaseUrl(): Promise<void> {
  try {
    mediaBaseUrl = await invoke<string>("get_media_base_url");
    await probeMediaElement(mediaBaseUrl);
  } catch (err) {
    console.error("media preview startup check failed", err);
  }
}

/**
 * Convert an absolute local path into a URL that `<video>` and `<audio>`
 * can actually play.
 *
 * WebKit will not play media from a custom URI scheme — neither Tauri's
 * `asset:` nor one of our own — because its media pipeline never calls the
 * scheme handler (webkit.org/b/146351, webkit.org/b/119469). So media goes
 * over loopback HTTP instead, where ranges and seeking work; see
 * `src-tauri/src/media_server.rs`. Keep using {@link assetUrl} for images,
 * which have no such restriction.
 *
 * Returns null before the server URL has been fetched, so callers can
 * hold off rather than render an element that is bound to fail.
 */
export function mediaUrl(absolutePath: string): string | null {
  if (!mediaBaseUrl) return null;
  return `${mediaBaseUrl}&path=${encodeURIComponent(absolutePath)}`;
}

/** Native "open subtitle file" dialog. Returns the absolute path or null. */
export async function pickSubtitleFile(): Promise<string | null> {
  const result = await openDialog({
    multiple: false,
    filters: [
      { name: "Subtitle", extensions: ["srt", "ass", "ssa"] },
    ],
  });
  if (result == null) return null;
  return typeof result === "string" ? result : (result as { path: string }).path;
}

/**
 * Native directory picker used by the Model Manager to change the
 * models directory. Returns the absolute path or null on cancel.
 */
export async function pickDirectory(
  title = "Select folder",
): Promise<string | null> {
  const result = await openDialog({
    directory: true,
    multiple: false,
    title,
  });
  if (result == null) return null;
  return typeof result === "string" ? result : (result as { path: string }).path;
}

/**
 * Native picker for the "Add Local Model" flow. `directory` = true
 * expects a folder (Whisper snapshot / Piper voice). `extensions`
 * filters to specific file types (GGUF).
 */
export async function pickModelSource(
  directory: boolean,
  filters?: { name: string; extensions: string[] }[],
): Promise<string | null> {
  const result = await openDialog({
    directory,
    multiple: false,
    filters: filters ?? undefined,
  });
  if (result == null) return null;
  return typeof result === "string" ? result : (result as { path: string }).path;
}

/** Native "save subtitle" dialog. Returns the absolute path or null. */
export async function pickSubtitleSavePath(
  defaultName: string,
  format: SubtitleFormat,
): Promise<string | null> {
  const result = await saveDialog({
    defaultPath: defaultName,
    filters: [
      { name: format.toUpperCase(), extensions: [format] },
    ],
  });
  return typeof result === "string" ? result : null;
}
