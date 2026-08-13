import { create } from "zustand";
import {
  api,
  initMediaBaseUrl,
  onJobProgress,
  onJobUpdate,
  onSyncSegmentCompleted,
  onTranslationChunkCompleted,
  onTtsSegmentCompleted,
  onWorkerStatus,
} from "@/ipc/bridge";
import type {
  AppInfo,
  AppSettings,
  AppSettingsPatch,
  CreateProjectInput,
  ExportKind,
  ExportSubtitlesResult,
  FfmpegAvailability,
  GenerateMode,
  ImportMediaInput,
  ImportMediaResult,
  ImportModelSpec,
  ImportSubtitlesResult,
  JobSnapshot,
  LocalModel,
  ModelDirectoryInfo,
  ProjectModelPatch,
  MixEnv,
  MixManifest,
  MixMode,
  MixRequest,
  MixSettings,
  MixSummary,
  PreviewMixResult,
  PreviewResult,
  PreviewSyncResult,
  Project,
  ProjectMediaState,
  ProjectSummary,
  RenderEnv,
  RenderManifest,
  RenderRequest,
  RenderSettings,
  RenderSummary,
  SttEnv,
  SttOptions,
  SubtitleDoc,
  SubtitleFormat,
  SubtitleSegmentPatch,
  SyncEnv,
  SyncManifest,
  SyncMode,
  SyncSettings,
  SyncSummary,
  TranscriptSummary,
  TranslateOptions,
  TranslationDoc,
  TranslationEnv,
  TranslationModelInfo,
  TranslationRecommendedPreset,
  TranslationSummary,
  TtsEnv,
  TtsManifest,
  TtsRecommendedVoicePreset,
  TtsSettings,
  TtsSummary,
  VoiceInfo,
  WhisperModelInfo,
  WorkerStatus,
} from "@/ipc/types";
import {
  defaultMixSettings,
  defaultRenderSettings,
  defaultSyncSettings,
  defaultTtsSettings,
} from "@/ipc/types";

// Phase 11 — trailing debounce per key so the flood of
// `translation.chunk_completed` / `tts.segment_completed` /
// `sync.segment_completed` events don't each trigger a full
// doc/manifest IPC round-trip. 350 ms is fast enough that the UI
// visibly updates within a chunk boundary but slow enough that a
// 30-chunk-per-second translation burst produces one refresh, not
// thirty.
const REFRESH_DEBOUNCE_MS = 350;
const debounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};
const PROGRESS_UI_INTERVAL_MS = 100;
const progressLastCommitted = new Map<string, number>();
let bootstrapStarted = false;

function debouncedInvoke(key: string, fn: () => void, ms = REFRESH_DEBOUNCE_MS): void {
  const existing = debounceTimers[key];
  if (existing) clearTimeout(existing);
  debounceTimers[key] = setTimeout(() => {
    delete debounceTimers[key];
    try {
      fn();
    } catch (err) {
      console.warn(`debounced refresh ${key} failed`, err);
    }
  }, ms);
}

/** Keep `subtitleMode` to something this FFmpeg can actually do.
 *
 *  Burning needs libass, which many builds ship without, and the default
 *  is to burn — so a project can arrive here (from its manifest, or from
 *  the defaults) asking for something that would only fail at render time.
 *  Fall back to the sidecar instead, which at least lands beside the movie.
 *  Written as a guard rather than a one-time default so it corrects itself
 *  no matter which of the two loads wins the race. */
function withSupportedSubtitleMode(
  settings: RenderSettings,
  env: RenderEnv | null,
): RenderSettings {
  if (!env || env.subtitleBurnAvailable) return settings;
  if (settings.subtitleMode !== "burned") return settings;
  return { ...settings, subtitleMode: "external" };
}

function translationSummaryFromDoc(doc: TranslationDoc): TranslationSummary {
  let translatedCount = 0;
  let editedCount = 0;
  for (const segment of doc.segments) {
    if (segment.translation.trim()) translatedCount += 1;
    if (segment.edited) editedCount += 1;
  }
  return {
    sourceLanguage: doc.sourceLanguage,
    targetLanguage: doc.targetLanguage,
    model: doc.model,
    promptVersion: doc.promptVersion,
    segmentCount: doc.segments.length,
    translatedCount,
    editedCount,
    cacheKey: doc.cacheKey,
    transcriptCacheKey: doc.transcriptCacheKey,
    createdAt: doc.createdAt,
    updatedAt: doc.updatedAt,
    relativePath: "translation/translation.json",
  };
}

// Phase 11 — per-stage auto-unload timers. `scheduleStageUnload`
// fires the `unload_stage_models` command after `autoUnloadAfterSecs`
// seconds of no new activity on that stage; `cancelStageUnload`
// short-circuits it when a follow-up job arrives (typical: translate
// finishes → tts starts within a second, no need to unload the LLM
// only to reload it if the user re-translates).
const unloadTimers: Record<string, ReturnType<typeof setTimeout>> = {};
function cancelStageUnload(stage: string): void {
  const t = unloadTimers[stage];
  if (t) {
    clearTimeout(t);
    delete unloadTimers[stage];
  }
}
function scheduleStageUnload(stage: string, settings: AppSettings | null): void {
  cancelStageUnload(stage);
  const grace = settings?.autoUnloadAfterSecs ?? 0;
  if (!grace || grace <= 0) return;
  // Only model-holding stages are worth timing.
  if (
    stage !== "transcribe" &&
    stage !== "translate" &&
    stage !== "tts" &&
    stage !== "sync"
  ) {
    return;
  }
  unloadTimers[stage] = setTimeout(() => {
    delete unloadTimers[stage];
    void api.unloadStageModels(stage).catch((err) => {
      console.warn(`stage unload ${stage} failed`, err);
    });
  }, grace * 1000);
}

async function unloadIdleStage(
  stage: string,
  state: Pick<AppState, "jobsById" | "settings">,
): Promise<void> {
  // Respect the user's explicit "keep models resident" choice.
  if (state.settings?.autoUnloadAfterSecs == null) return;
  const active = Object.values(state.jobsById).some(
    (job) =>
      job.stage === stage &&
      (job.status === "queued" || job.status === "running"),
  );
  if (active) return;
  cancelStageUnload(stage);
  try {
    await api.unloadStageModels(stage);
  } catch (err) {
    console.warn(`releasing idle ${stage} model failed`, err);
  }
}

interface AppState {
  bootReady: boolean;
  bootError: string | null;

  appInfo: AppInfo | null;
  settings: AppSettings | null;

  projects: ProjectSummary[];
  projectsLoading: boolean;

  currentProject: Project | null;
  currentProjectLoading: boolean;
  currentMedia: ProjectMediaState | null;
  currentMediaLoading: boolean;

  ffmpeg: FfmpegAvailability | null;

  /** Live progress per job id (0..1). Removed on terminal update. */
  jobProgress: Record<string, number>;
  /** Latest snapshot per job id. */
  jobsById: Record<string, JobSnapshot>;

  workerUnlisten: (() => void) | null;
  jobUnlisten: (() => void) | null;

  worker: WorkerStatus;

  // Phase 3: speech-to-text
  sttEnv: SttEnv | null;
  whisperModels: WhisperModelInfo[];
  whisperModelsLoading: boolean;

  // Phase 4: local LLM translation
  translationEnv: TranslationEnv | null;
  translationModels: TranslationModelInfo[];
  translationModelsLoading: boolean;
  // Phase 12: curated auto-download presets (mirrors whisperModels).
  translationRecommendedPresets: TranslationRecommendedPreset[];
  translationRecommendedLoading: boolean;
  currentTranslationDoc: TranslationDoc | null;

  // Phase 5: subtitle editor
  currentSubtitleDoc: SubtitleDoc | null;
  currentSubtitleLoading: boolean;

  // Phase 6: local TTS / AI dubbing
  ttsEnv: TtsEnv | null;
  ttsVoices: VoiceInfo[];
  ttsVoicesLoading: boolean;
  // Phase 12 — curated auto-download presets, mirrors
  // `translationRecommendedPresets` / `whisperModels`.
  ttsRecommendedVoices: TtsRecommendedVoicePreset[];
  ttsRecommendedLoading: boolean;
  currentTtsManifest: TtsManifest | null;
  ttsEngine: string;
  ttsVoiceId: string;
  ttsSettings: TtsSettings;
  lastTtsPreview: PreviewResult | null;

  // Phase 7: voice synchronisation
  syncEnv: SyncEnv | null;
  currentSyncManifest: SyncManifest | null;
  syncSettings: SyncSettings;
  lastSyncPreview: PreviewSyncResult | null;

  // Phase 8: audio mixing
  mixEnv: MixEnv | null;
  currentMixManifest: MixManifest | null;
  mixSettings: MixSettings;
  lastMixPreview: PreviewMixResult | null;

  // Phase 9: final video rendering
  renderEnv: RenderEnv | null;
  currentRenderManifest: RenderManifest | null;
  renderSettings: RenderSettings;

  // Phase 10: local model manager
  localModels: LocalModel[];
  localModelsLoading: boolean;
  modelDirectory: ModelDirectoryInfo | null;
  modelImportBusy: boolean;

  bootstrap: () => Promise<void>;
  refreshProjects: () => Promise<void>;
  createProject: (input: CreateProjectInput) => Promise<ProjectSummary>;
  openProject: (id: string) => Promise<Project>;
  deleteProject: (id: string) => Promise<void>;
  updateSettings: (patch: AppSettingsPatch) => Promise<void>;
  refreshFfmpeg: () => Promise<FfmpegAvailability>;

  // Phase 2
  importMedia: (input: ImportMediaInput) => Promise<ImportMediaResult>;
  refreshMedia: (projectId: string) => Promise<ProjectMediaState>;
  extractAudio: (projectId: string) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;

  // Phase 3
  refreshSttEnv: () => Promise<SttEnv | null>;
  refreshWhisperModels: () => Promise<WhisperModelInfo[]>;
  downloadWhisperModel: (name: string) => Promise<void>;
  startTranscribe: (
    projectId: string,
    options?: SttOptions,
  ) => Promise<TranscriptSummary | null>;

  // Phase 4
  refreshTranslationEnv: () => Promise<TranslationEnv | null>;
  refreshTranslationModels: () => Promise<TranslationModelInfo[]>;
  refreshTranslationRecommendedPresets: () => Promise<
    TranslationRecommendedPreset[]
  >;
  downloadTranslationModel: (preset: string) => Promise<void>;
  loadTranslationDoc: (projectId: string) => Promise<TranslationDoc | null>;
  startTranslate: (
    projectId: string,
    options: TranslateOptions,
  ) => Promise<TranslationSummary | null>;
  updateTranslationSegment: (
    projectId: string,
    segmentId: number,
    translation: string,
  ) => Promise<void>;

  // Phase 5
  loadSubtitles: (projectId: string) => Promise<SubtitleDoc | null>;
  rebuildSubtitles: (projectId: string) => Promise<SubtitleDoc>;
  updateSubtitleSegment: (
    projectId: string,
    segmentId: number,
    patch: SubtitleSegmentPatch,
  ) => Promise<SubtitleDoc>;
  addSubtitleSegment: (
    projectId: string,
    afterId: number | null,
    start: number,
    end: number,
  ) => Promise<SubtitleDoc>;
  deleteSubtitleSegment: (
    projectId: string,
    segmentId: number,
  ) => Promise<SubtitleDoc>;
  splitSubtitleSegment: (
    projectId: string,
    segmentId: number,
    splitTime: number,
  ) => Promise<SubtitleDoc>;
  mergeSubtitleSegment: (
    projectId: string,
    segmentId: number,
  ) => Promise<SubtitleDoc>;
  importSubtitles: (
    projectId: string,
    path: string,
  ) => Promise<ImportSubtitlesResult>;
  exportSubtitles: (
    projectId: string,
    path: string,
    format: SubtitleFormat,
    kind?: ExportKind,
  ) => Promise<ExportSubtitlesResult>;

  // Phase 6
  refreshTtsEnv: () => Promise<TtsEnv | null>;
  refreshTtsVoices: () => Promise<VoiceInfo[]>;
  refreshTtsRecommendedVoices: () => Promise<TtsRecommendedVoicePreset[]>;
  downloadTtsVoice: (preset: string) => Promise<void>;
  loadTtsManifest: (projectId: string) => Promise<TtsManifest | null>;
  refreshTtsSummary: (projectId: string) => Promise<TtsSummary | null>;
  setTtsEngine: (engine: string) => void;
  setTtsVoiceId: (voiceId: string) => void;
  setTtsSettings: (settings: Partial<TtsSettings>) => void;
  previewTts: (
    projectId: string,
    segmentId: number,
    overrides?: { voiceId?: string | null; settings?: TtsSettings | null },
  ) => Promise<PreviewResult>;
  startGenerateTts: (
    projectId: string,
    mode: GenerateMode,
  ) => Promise<TtsSummary | null>;

  // Phase 7
  refreshSyncEnv: () => Promise<SyncEnv | null>;
  loadSyncManifest: (projectId: string) => Promise<SyncManifest | null>;
  refreshSyncSummary: (projectId: string) => Promise<SyncSummary | null>;
  setSyncSettings: (settings: Partial<SyncSettings>) => void;
  previewSync: (
    projectId: string,
    segmentId: number,
    overrides?: { settings?: SyncSettings | null },
  ) => Promise<PreviewSyncResult>;
  startApplySync: (
    projectId: string,
    mode: SyncMode,
  ) => Promise<SyncSummary | null>;

  // Phase 8
  refreshMixEnv: () => Promise<MixEnv | null>;
  loadMixManifest: (projectId: string) => Promise<MixManifest | null>;
  refreshMixSummary: (projectId: string) => Promise<MixSummary | null>;
  loadMixPreview: (projectId: string) => Promise<PreviewMixResult | null>;
  setMixSettings: (settings: Partial<MixSettings>) => void;
  startApplyMix: (
    projectId: string,
    overrides?: { mode?: MixMode },
  ) => Promise<MixSummary | null>;

  // Phase 9
  refreshRenderEnv: () => Promise<RenderEnv | null>;
  loadRenderManifest: (projectId: string) => Promise<RenderManifest | null>;
  refreshRenderSummary: (projectId: string) => Promise<RenderSummary | null>;
  setRenderSettings: (settings: Partial<RenderSettings>) => void;
  startApplyRender: (
    projectId: string,
    overrides?: { force?: boolean },
  ) => Promise<RenderSummary | null>;

  // Phase 10 (Local Model Manager)
  refreshLocalModels: (force?: boolean) => Promise<LocalModel[]>;
  refreshModelDirectory: () => Promise<ModelDirectoryInfo | null>;
  setModelDirectory: (path: string | null) => Promise<ModelDirectoryInfo>;
  importLocalModel: (spec: ImportModelSpec) => Promise<LocalModel>;
  unloadAllModels: () => Promise<string[]>;
  updateProjectModels: (
    projectId: string,
    patch: ProjectModelPatch,
  ) => Promise<Project>;
  markFirstRunCompleted: () => Promise<void>;
}

const initialWorker: WorkerStatus = {
  state: "starting",
  pid: null,
  uptimeMs: 0,
  lastError: null,
};

export const useAppStore = create<AppState>((set, get) => ({
  bootReady: false,
  bootError: null,

  appInfo: null,
  settings: null,

  projects: [],
  projectsLoading: false,

  currentProject: null,
  currentProjectLoading: false,
  currentMedia: null,
  currentMediaLoading: false,

  ffmpeg: null,

  jobProgress: {},
  jobsById: {},

  workerUnlisten: null,
  jobUnlisten: null,

  worker: initialWorker,

  sttEnv: null,
  whisperModels: [],
  whisperModelsLoading: false,

  translationEnv: null,
  translationModels: [],
  translationModelsLoading: false,
  translationRecommendedPresets: [],
  translationRecommendedLoading: false,
  currentTranslationDoc: null,

  currentSubtitleDoc: null,
  currentSubtitleLoading: false,

  ttsEnv: null,
  ttsVoices: [],
  ttsVoicesLoading: false,
  ttsRecommendedVoices: [],
  ttsRecommendedLoading: false,
  currentTtsManifest: null,
  ttsEngine: "piper",
  ttsVoiceId: "",
  ttsSettings: defaultTtsSettings(),
  lastTtsPreview: null,

  syncEnv: null,
  currentSyncManifest: null,
  syncSettings: defaultSyncSettings(),
  lastSyncPreview: null,

  mixEnv: null,
  currentMixManifest: null,
  mixSettings: defaultMixSettings(),
  lastMixPreview: null,

  renderEnv: null,
  currentRenderManifest: null,
  renderSettings: defaultRenderSettings(),

  localModels: [],
  localModelsLoading: false,
  modelDirectory: null,
  modelImportBusy: false,

  bootstrap: async () => {
    // React StrictMode invokes mount effects twice in development. Keep
    // bootstrap idempotent so it cannot register duplicate Tauri event
    // listeners or duplicate every progress/terminal update.
    if (bootstrapStarted || get().bootReady) return;
    bootstrapStarted = true;
    try {
      const [appInfo, settings, projects, workerStatus, ffmpeg] =
        await Promise.all([
          api.getAppInfo(),
          api.getSettings(),
          api.listProjects(),
          api.workerStatus(),
          api.getFfmpegAvailability(),
          // Playback needs the loopback media server's URL before any
          // <video> mounts. It never throws, so it cannot fail bootstrap.
          initMediaBaseUrl(),
        ]);

      const workerUnlisten = await onWorkerStatus((status) =>
        set({ worker: status }),
      );

      // Combined listener disposer for job events.
      const [progUnlisten, updUnlisten, chunkUnlisten, ttsUnlisten, syncUnlisten] =
        await Promise.all([
          onJobProgress((evt) => {
            const now = Date.now();
            const last = progressLastCommitted.get(evt.id) ?? 0;
            if (
              evt.progress < 1 &&
              now - last < PROGRESS_UI_INTERVAL_MS
            ) {
              return;
            }
            progressLastCommitted.set(evt.id, now);
            set((state) => ({
              jobProgress: { ...state.jobProgress, [evt.id]: evt.progress },
            }));
          }),
          onJobUpdate((snap) => {
            // Phase 11 — auto-unload the stage's model after a grace
            // period so long idle stretches don't hold a multi-GB
            // Whisper/LLM/Piper resident in RAM. `startsAtStage`
            // cancels a pending unload the moment a new job of the
            // same stage arrives (typical pipeline: translate →
            // tts → sync each within seconds).
            if (snap.status === "running" || snap.status === "queued") {
              cancelStageUnload(snap.stage);
            } else if (
              snap.status === "completed" ||
              snap.status === "failed" ||
              snap.status === "cancelled"
            ) {
              scheduleStageUnload(snap.stage, get().settings);
            }
            set((state) => {
              const jobsById = { ...state.jobsById, [snap.id]: snap };
              const jobProgress = { ...state.jobProgress };
              if (
                snap.status === "completed" ||
                snap.status === "failed" ||
                snap.status === "cancelled"
              ) {
                delete jobProgress[snap.id];
                progressLastCommitted.delete(snap.id);
                // If this is the current project's job, refresh media so the
                // UI picks up new audio/transcript/translation manifests.
                if (
                  state.currentProject &&
                  snap.projectId === state.currentProject.id
                ) {
                  void get().refreshMedia(snap.projectId);
                  if (snap.stage === "translate") {
                    void get().loadTranslationDoc(snap.projectId);
                    // Phase 12 UX — auto-build subtitles the
                    // moment translation finishes. Every downstream
                    // stage (TTS, sync, mix, render, export) is
                    // gated on the subtitle doc existing, so
                    // making the user click "Build subtitles"
                    // manually was the #2 friction point after
                    // "Extract audio". `rebuildSubtitles` is safe
                    // to call even if a doc already exists —
                    // Rust-side it merges the fresh translation
                    // into the current subtitle timing.
                    if (snap.status === "completed") {
                      void get()
                        .rebuildSubtitles(snap.projectId)
                        .catch((err) =>
                          console.warn(
                            "auto-rebuild subtitles after translate failed",
                            err,
                          ),
                        );
                    }
                  }
                  if (snap.stage === "tts") {
                    void get().loadTtsManifest(snap.projectId);
                    void get().loadSubtitles(snap.projectId);
                  }
                  if (snap.stage === "sync") {
                    void get().loadSyncManifest(snap.projectId);
                    void get().refreshSyncSummary(snap.projectId);
                    void get().loadSubtitles(snap.projectId);
                  }
                  if (snap.stage === "mix") {
                    void get().loadMixManifest(snap.projectId);
                    void get().refreshMixSummary(snap.projectId);
                    void get().loadMixPreview(snap.projectId);
                    void get().loadSubtitles(snap.projectId);
                  }
                  if (snap.stage === "render") {
                    void get().loadRenderManifest(snap.projectId);
                    void get().refreshRenderSummary(snap.projectId);
                    void get().loadSubtitles(snap.projectId);
                  }
                }
              }
              return { jobsById, jobProgress };
            });
          }),
          onTranslationChunkCompleted((evt) => {
            const state = get();
            if (
              !state.currentProject ||
              state.currentProject.id !== evt.projectId
            ) {
              return;
            }
            // Update the cheap counters from the event itself. Loading
            // the translation doc is still debounced, but do not call
            // get_project_media here: that command also collects every
            // downstream summary and used to launch ffprobe repeatedly
            // throughout a long translation job.
            set((current) => {
              const media = current.currentMedia;
              const summary = media?.translation;
              if (!media || !summary) return {};
              return {
                currentMedia: {
                  ...media,
                  translation: {
                    ...summary,
                    translatedCount: evt.translatedCount,
                    segmentCount: evt.segmentCount,
                  },
                },
              };
            });
            debouncedInvoke(`translation:${evt.projectId}`, () => {
              void get().loadTranslationDoc(evt.projectId);
            });
          }),
          onTtsSegmentCompleted((evt) => {
            const state = get();
            if (
              !state.currentProject ||
              state.currentProject.id !== evt.projectId
            ) {
              return;
            }
            debouncedInvoke(`tts:${evt.projectId}`, () => {
              void get().loadTtsManifest(evt.projectId);
            });
          }),
          onSyncSegmentCompleted((evt) => {
            const state = get();
            if (
              !state.currentProject ||
              state.currentProject.id !== evt.projectId
            ) {
              return;
            }
            debouncedInvoke(`sync:${evt.projectId}`, () => {
              void get().loadSyncManifest(evt.projectId);
            });
          }),
        ]);
      const jobUnlisten = () => {
        progUnlisten();
        updUnlisten();
        chunkUnlisten();
        ttsUnlisten();
        syncUnlisten();
      };

      set({
        appInfo,
        settings,
        projects,
        worker: workerStatus,
        ffmpeg,
        workerUnlisten,
        jobUnlisten,
        bootReady: true,
        bootError: null,
      });
    } catch (err) {
      set({
        bootReady: true,
        bootError: err instanceof Error ? err.message : JSON.stringify(err),
      });
    }
  },

  refreshProjects: async () => {
    set({ projectsLoading: true });
    try {
      const projects = await api.listProjects();
      set({ projects });
    } finally {
      set({ projectsLoading: false });
    }
  },

  createProject: async (input) => {
    const project = await api.createProject(input);
    await get().refreshProjects();
    return project;
  },

  openProject: async (id) => {
    set({
      currentProjectLoading: true,
      currentProject: null,
      currentMedia: null,
    });
    try {
      const project = await api.openProject(id);
      set({ currentProject: project });
      // Kick off media load in the background — cheap enough to await here
      // so the caller sees the fully-populated screen when we resolve.
      await get().refreshMedia(id);
      return project;
    } finally {
      set({ currentProjectLoading: false });
    }
  },

  deleteProject: async (id) => {
    await api.deleteProject(id);
    await get().refreshProjects();
    if (get().currentProject?.id === id) {
      set({ currentProject: null, currentMedia: null });
    }
  },

  updateSettings: async (patch) => {
    const settings = await api.updateSettings(patch);
    set({ settings });
    // ffmpeg_path changes trigger a re-detect on the backend; refresh
    // our cached availability so Settings reflects it immediately.
    if ("ffmpegPath" in patch || "ffprobePath" in patch) {
      await get().refreshFfmpeg();
    }
  },

  refreshFfmpeg: async () => {
    const ffmpeg = await api.refreshFfmpeg();
    set({ ffmpeg });
    return ffmpeg;
  },

  importMedia: async (input) => {
    set({ currentMediaLoading: true });
    try {
      const result = await api.importMedia(input);
      set({ currentProject: result.project });
      const media = await get().refreshMedia(input.projectId);
      set({ currentMedia: media });
      await get().refreshProjects();
      return result;
    } finally {
      set({ currentMediaLoading: false });
    }
  },

  refreshMedia: async (projectId) => {
    set({ currentMediaLoading: true });
    try {
      const media = await api.getProjectMedia(projectId);
      set({ currentMedia: media });
      // Seed job map with any live jobs the backend already tracks.
      if (media.activeJobs.length > 0) {
        set((state) => ({
          jobsById: media.activeJobs.reduce(
            (acc, j) => ({ ...acc, [j.id]: j }),
            state.jobsById,
          ),
        }));
      }
      return media;
    } finally {
      set({ currentMediaLoading: false });
    }
  },

  extractAudio: async (projectId) => {
    const start = await api.extractAudio(projectId);
    if (start.kind === "cacheHit") {
      // Nothing to do — the cache manifest is authoritative. Refresh
      // media state so the UI shows the cached path.
      await get().refreshMedia(projectId);
      return;
    }
    // Started: seed the jobs map so the UI has something to render
    // before the first `job://progress` event fires.
    const { kind: _kind, ...snap } = start;
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
  },

  cancelJob: async (jobId) => {
    await api.cancelJob(jobId);
  },

  refreshSttEnv: async () => {
    try {
      const env = await api.getSttEnv();
      set({ sttEnv: env });
      return env;
    } catch (err) {
      // Worker may not be up yet; leave the previous snapshot in
      // place and let the caller retry.
      console.warn("stt env refresh failed", err);
      return null;
    }
  },

  refreshWhisperModels: async () => {
    set({ whisperModelsLoading: true });
    try {
      const models = await api.listWhisperModels();
      set({ whisperModels: models });
      return models;
    } finally {
      set({ whisperModelsLoading: false });
    }
  },

  downloadWhisperModel: async (name) => {
    const snap = await api.downloadWhisperModel(name);
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
  },

  startTranscribe: async (projectId, options) => {
    const start = await api.transcribe(projectId, options);
    // Phase 10 — remember the model in the project file so
    // reopening it uses the same choice, and a missing model is
    // reported explicitly rather than silently substituted.
    const chosenModel = options?.model;
    if (chosenModel) {
      void get()
        .updateProjectModels(projectId, { whisperModel: chosenModel })
        .catch((err) =>
          console.warn("persisting whisper model failed", err),
        );
    }
    if (start.kind === "cacheHit") {
      await get().refreshMedia(projectId);
      return start.transcript;
    }
    const { kind: _kind, ...snap } = start;
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
    return null;
  },

  refreshTranslationEnv: async () => {
    try {
      const env = await api.getTranslationEnv();
      set({ translationEnv: env });
      return env;
    } catch (err) {
      console.warn("translation env refresh failed", err);
      return null;
    }
  },

  refreshTranslationModels: async () => {
    set({ translationModelsLoading: true });
    try {
      const models = await api.listTranslationModels();
      set({ translationModels: models });
      return models;
    } catch (err) {
      console.warn("translation models refresh failed", err);
      return [];
    } finally {
      set({ translationModelsLoading: false });
    }
  },

  refreshTranslationRecommendedPresets: async () => {
    set({ translationRecommendedLoading: true });
    try {
      const presets = await api.listRecommendedTranslationPresets();
      set({ translationRecommendedPresets: presets });
      return presets;
    } catch (err) {
      console.warn("translation recommended presets refresh failed", err);
      return [];
    } finally {
      set({ translationRecommendedLoading: false });
    }
  },

  downloadTranslationModel: async (preset) => {
    // Phase 12 — same job-registration pattern as
    // `downloadWhisperModel`. The Rust side emits `job://update` +
    // `job://progress` events keyed on the snapshot's id, so
    // seeding both maps here makes the download bar render
    // immediately without waiting for the first progress tick.
    const snap = await api.downloadTranslationModel(preset);
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
  },

  loadTranslationDoc: async (projectId) => {
    const doc = await api.getProjectTranslationDoc(projectId);
    set((state) => ({
      currentTranslationDoc: doc,
      currentMedia:
        doc && state.currentMedia
          ? {
              ...state.currentMedia,
              translation: translationSummaryFromDoc(doc),
            }
          : state.currentMedia,
    }));
    return doc;
  },

  startTranslate: async (projectId, options) => {
    // A completed large-v3 transcription can otherwise overlap in RAM
    // with the GGUF model for the full idle grace period.
    await unloadIdleStage("transcribe", get());
    const start = await api.translate(projectId, options);
    // Phase 10 — persist per-project translation model choice.
    void get()
      .updateProjectModels(projectId, { translationModel: options.model })
      .catch((err) =>
        console.warn("persisting translation model failed", err),
      );
    if (start.kind === "cacheHit") {
      await get().refreshMedia(projectId);
      await get().loadTranslationDoc(projectId);
      return start.summary;
    }
    const { kind: _kind, ...snap } = start;
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
    // Load the (possibly seeded, mostly-empty) doc so the editor
    // renders skeleton rows while chunks stream in.
    await get().loadTranslationDoc(projectId);
    return null;
  },

  updateTranslationSegment: async (projectId, segmentId, translation) => {
    const summary = await api.updateTranslationSegment(
      projectId,
      segmentId,
      translation,
    );
    // Optimistically patch the doc in memory so the input isn't
    // debounced by another refresh round-trip.
    set((state) => {
      const doc = state.currentTranslationDoc;
      if (!doc) return {};
      const segments = doc.segments.map((s) =>
        s.id === segmentId
          ? { ...s, translation, edited: true }
          : s,
      );
      return {
        currentTranslationDoc: { ...doc, segments },
        currentMedia: state.currentMedia
          ? { ...state.currentMedia, translation: summary }
          : state.currentMedia,
      };
    });
  },

  loadSubtitles: async (projectId) => {
    set({ currentSubtitleLoading: true });
    try {
      const doc = await api.getProjectSubtitlesDoc(projectId);
      set({ currentSubtitleDoc: doc });
      return doc;
    } finally {
      set({ currentSubtitleLoading: false });
    }
  },

  rebuildSubtitles: async (projectId) => {
    const doc = await api.rebuildProjectSubtitles(projectId);
    set({ currentSubtitleDoc: doc });
    await get().refreshMedia(projectId);
    return doc;
  },

  updateSubtitleSegment: async (projectId, segmentId, patch) => {
    const doc = await api.updateSubtitleSegment(projectId, segmentId, patch);
    set({ currentSubtitleDoc: doc });
    // Summary counters (dirty flags) live on the media summary.
    void get().refreshMedia(projectId);
    return doc;
  },

  addSubtitleSegment: async (projectId, afterId, start, end) => {
    const doc = await api.addSubtitleSegment(projectId, afterId, start, end);
    set({ currentSubtitleDoc: doc });
    void get().refreshMedia(projectId);
    return doc;
  },

  deleteSubtitleSegment: async (projectId, segmentId) => {
    const doc = await api.deleteSubtitleSegment(projectId, segmentId);
    set({ currentSubtitleDoc: doc });
    void get().refreshMedia(projectId);
    return doc;
  },

  splitSubtitleSegment: async (projectId, segmentId, splitTime) => {
    const doc = await api.splitSubtitleSegment(projectId, segmentId, splitTime);
    set({ currentSubtitleDoc: doc });
    void get().refreshMedia(projectId);
    return doc;
  },

  mergeSubtitleSegment: async (projectId, segmentId) => {
    const doc = await api.mergeSubtitleSegment(projectId, segmentId);
    set({ currentSubtitleDoc: doc });
    void get().refreshMedia(projectId);
    return doc;
  },

  importSubtitles: async (projectId, path) => {
    const result = await api.importSubtitles(projectId, path);
    set({ currentSubtitleDoc: result.doc });
    void get().refreshMedia(projectId);
    return result;
  },

  exportSubtitles: async (projectId, path, format, kind) => {
    return api.exportSubtitles(projectId, path, format, kind);
  },

  // ---------- Phase 6 (TTS / dubbing) ----------

  refreshTtsEnv: async () => {
    try {
      const env = await api.getTtsEnv();
      set({ ttsEnv: env });
      if (!get().ttsEngine && env.defaultEngine) {
        set({ ttsEngine: env.defaultEngine });
      }
      return env;
    } catch (err) {
      console.warn("tts env refresh failed", err);
      return null;
    }
  },

  refreshTtsVoices: async () => {
    set({ ttsVoicesLoading: true });
    try {
      const voices = await api.listTtsVoices();
      set({ ttsVoices: voices });
      // Pre-select the first voice matching the current engine when
      // the user hasn't picked one yet.
      const state = get();
      if (!state.ttsVoiceId && voices.length > 0) {
        const first =
          voices.find((v) => v.engine === state.ttsEngine) ?? voices[0];
        set({ ttsVoiceId: first.id, ttsEngine: first.engine });
      }
      return voices;
    } catch (err) {
      console.warn("tts voices refresh failed", err);
      return [];
    } finally {
      set({ ttsVoicesLoading: false });
    }
  },

  refreshTtsRecommendedVoices: async () => {
    set({ ttsRecommendedLoading: true });
    try {
      const presets = await api.listRecommendedTtsVoices();
      set({ ttsRecommendedVoices: presets });
      return presets;
    } catch (err) {
      console.warn("tts recommended voices refresh failed", err);
      return [];
    } finally {
      set({ ttsRecommendedLoading: false });
    }
  },

  downloadTtsVoice: async (preset) => {
    // Phase 12 — same job-registration pattern as
    // `downloadWhisperModel` and `downloadTranslationModel`. Seed
    // both maps so the progress bar shows immediately without
    // waiting for the first progress tick from the worker.
    const snap = await api.downloadTtsVoice(preset);
    set((state) => ({
      jobsById: { ...state.jobsById, [snap.id]: snap },
      jobProgress: { ...state.jobProgress, [snap.id]: 0 },
    }));
  },

  loadTtsManifest: async (projectId) => {
    const manifest = await api.getProjectTtsManifest(projectId);
    set({ currentTtsManifest: manifest });
    return manifest;
  },

  refreshTtsSummary: async (projectId) => {
    const state = get();
    try {
      const summary = await api.getProjectTtsSummary(
        projectId,
        state.ttsEngine,
        state.ttsVoiceId,
        state.ttsSettings,
      );
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, tts: summary } };
      });
      return summary;
    } catch (err) {
      console.warn("tts summary refresh failed", err);
      return null;
    }
  },

  setTtsEngine: (engine) => {
    set({ ttsEngine: engine });
    // Re-select the first voice on this engine so we never leave the
    // UI pointing at an invalid combination.
    const state = get();
    if (!state.ttsVoices.some((v) => v.id === state.ttsVoiceId && v.engine === engine)) {
      const fallback = state.ttsVoices.find((v) => v.engine === engine);
      set({ ttsVoiceId: fallback?.id ?? "" });
    }
  },

  setTtsVoiceId: (voiceId) => set({ ttsVoiceId: voiceId }),

  setTtsSettings: (settings) =>
    set((state) => ({ ttsSettings: { ...state.ttsSettings, ...settings } })),

  previewTts: async (projectId, segmentId, overrides) => {
    const state = get();
    const voiceId = overrides?.voiceId ?? state.ttsVoiceId ?? null;
    const settings = overrides?.settings ?? state.ttsSettings;
    const preview = await api.previewTtsSegment(
      projectId,
      segmentId,
      voiceId,
      state.ttsEngine,
      settings,
    );
    set({ lastTtsPreview: preview });
    // Refresh manifest so the row that was just cached shows a tick.
    void get().loadTtsManifest(projectId);
    void get().refreshTtsSummary(projectId);
    return preview;
  },

  startGenerateTts: async (projectId, mode) => {
    const state = get();
    // TTS does not need the Whisper or translation model. Release idle
    // predecessors before loading the voice to avoid cross-stage peaks.
    await unloadIdleStage("transcribe", state);
    await unloadIdleStage("translate", get());
    const start = await api.generateTts(projectId, {
      engine: state.ttsEngine,
      defaultVoiceId: state.ttsVoiceId,
      settings: state.ttsSettings,
      mode,
    });
    // Phase 10 — persist per-project TTS engine + voice.
    void get()
      .updateProjectModels(projectId, {
        ttsEngine: state.ttsEngine,
        ttsVoiceId: state.ttsVoiceId,
      })
      .catch((err) => console.warn("persisting tts choice failed", err));
    if (start.kind === "upToDate") {
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, tts: start.summary } };
      });
      return start.summary;
    }
    const { kind: _kind, ...snap } = start;
    set((s) => ({
      jobsById: { ...s.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...s.jobProgress, [snap.id]: 0 },
    }));
    return null;
  },

  // ---------- Phase 7 (Voice synchronisation) ----------

  refreshSyncEnv: async () => {
    try {
      const env = await api.getSyncEnv();
      set({ syncEnv: env });
      return env;
    } catch (err) {
      console.warn("sync env refresh failed", err);
      return null;
    }
  },

  loadSyncManifest: async (projectId) => {
    const manifest = await api.getProjectSyncManifest(projectId);
    set({ currentSyncManifest: manifest });
    return manifest;
  },

  refreshSyncSummary: async (projectId) => {
    const state = get();
    try {
      const summary = await api.getProjectSyncSummary(
        projectId,
        state.syncSettings,
      );
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, sync: summary } };
      });
      return summary;
    } catch (err) {
      console.warn("sync summary refresh failed", err);
      return null;
    }
  },

  setSyncSettings: (settings) =>
    set((state) => ({ syncSettings: { ...state.syncSettings, ...settings } })),

  previewSync: async (projectId, segmentId, overrides) => {
    const state = get();
    const settings = overrides?.settings ?? state.syncSettings;
    const preview = await api.previewSyncSegment(projectId, segmentId, settings);
    set({ lastSyncPreview: preview });
    void get().loadSyncManifest(projectId);
    void get().refreshSyncSummary(projectId);
    return preview;
  },

  startApplySync: async (projectId, mode) => {
    const state = get();
    const start = await api.applySync(projectId, {
      settings: state.syncSettings,
      mode,
    });
    if (start.kind === "upToDate") {
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, sync: start.summary } };
      });
      return start.summary;
    }
    const { kind: _kind, ...snap } = start;
    set((s) => ({
      jobsById: { ...s.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...s.jobProgress, [snap.id]: 0 },
    }));
    return null;
  },

  // ---------- Phase 8 (Audio mixing) ----------

  refreshMixEnv: async () => {
    try {
      const env = await api.getMixEnv();
      set({ mixEnv: env });
      return env;
    } catch (err) {
      console.warn("mix env refresh failed", err);
      return null;
    }
  },

  loadMixManifest: async (projectId) => {
    const manifest = await api.getProjectMixManifest(projectId);
    set({ currentMixManifest: manifest });
    // Keep the sliders in sync with what was last generated so the
    // UI doesn't show stale defaults after a project reopen.
    if (manifest?.settings) {
      set({ mixSettings: manifest.settings });
    }
    return manifest;
  },

  refreshMixSummary: async (projectId) => {
    const state = get();
    try {
      const summary = await api.getProjectMixSummary(
        projectId,
        state.mixSettings,
      );
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, mix: summary } };
      });
      return summary;
    } catch (err) {
      console.warn("mix summary refresh failed", err);
      return null;
    }
  },

  loadMixPreview: async (projectId) => {
    try {
      const preview = await api.getProjectMixPreview(projectId);
      set({ lastMixPreview: preview });
      return preview;
    } catch (err) {
      console.warn("mix preview load failed", err);
      return null;
    }
  },

  setMixSettings: (settings) =>
    set((state) => ({ mixSettings: { ...state.mixSettings, ...settings } })),

  startApplyMix: async (projectId, overrides) => {
    const state = get();
    const request: MixRequest = {
      settings: state.mixSettings,
      mode: overrides?.mode ?? { kind: "all" },
    };
    const start = await api.applyMix(projectId, request);
    if (start.kind === "upToDate") {
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, mix: start.summary } };
      });
      void get().loadMixPreview(projectId);
      return start.summary;
    }
    const { kind: _kind, ...snap } = start;
    set((s) => ({
      jobsById: { ...s.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...s.jobProgress, [snap.id]: 0 },
    }));
    return null;
  },

  // ---------- Phase 9 (Final video rendering) ----------

  refreshRenderEnv: async () => {
    try {
      const env = await api.getRenderEnv();
      set((s) => ({
        renderEnv: env,
        renderSettings: withSupportedSubtitleMode(s.renderSettings, env),
      }));
      return env;
    } catch (err) {
      console.warn("render env refresh failed", err);
      return null;
    }
  },

  loadRenderManifest: async (projectId) => {
    const manifest = await api.getProjectRenderManifest(projectId);
    set({ currentRenderManifest: manifest });
    // Keep the panel controls in sync with what was last generated
    // so a project reopen doesn't wipe the user's settings back to
    // the defaults.
    if (manifest?.settings) {
      set((s) => ({
        renderSettings: withSupportedSubtitleMode(
          manifest.settings,
          s.renderEnv,
        ),
      }));
    }
    return manifest;
  },

  refreshRenderSummary: async (projectId) => {
    const state = get();
    try {
      const summary = await api.getProjectRenderSummary(
        projectId,
        state.renderSettings,
      );
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, render: summary } };
      });
      return summary;
    } catch (err) {
      console.warn("render summary refresh failed", err);
      return null;
    }
  },

  setRenderSettings: (settings) =>
    set((state) => ({
      renderSettings: { ...state.renderSettings, ...settings },
    })),

  startApplyRender: async (projectId, overrides) => {
    const state = get();
    const request: RenderRequest = {
      settings: state.renderSettings,
      force: overrides?.force ?? false,
    };
    const start = await api.applyRender(projectId, request);
    if (start.kind === "upToDate") {
      set((s) => {
        if (!s.currentMedia) return {};
        return { currentMedia: { ...s.currentMedia, render: start.summary } };
      });
      return start.summary;
    }
    const { kind: _kind, ...snap } = start;
    set((s) => ({
      jobsById: { ...s.jobsById, [snap.id]: snap as JobSnapshot },
      jobProgress: { ...s.jobProgress, [snap.id]: 0 },
    }));
    return null;
  },

  // ---------- Phase 10 (Local Model Manager) ----------

  refreshLocalModels: async (force = false) => {
    set({ localModelsLoading: true });
    try {
      const models = force
        ? await api.rescanLocalModels()
        : await api.listLocalModels();
      set({ localModels: models });
      return models;
    } catch (err) {
      console.warn("local models refresh failed", err);
      return [];
    } finally {
      set({ localModelsLoading: false });
    }
  },

  refreshModelDirectory: async () => {
    try {
      const info = await api.getModelDirectory();
      set({ modelDirectory: info });
      return info;
    } catch (err) {
      console.warn("model directory refresh failed", err);
      return null;
    }
  },

  setModelDirectory: async (path) => {
    const info = await api.setModelDirectory(path);
    set({ modelDirectory: info });
    // The rescan is cheap and immediately reflects the new dir.
    await get().refreshLocalModels(true);
    return info;
  },

  importLocalModel: async (spec) => {
    set({ modelImportBusy: true });
    try {
      const entry = await api.importLocalModel(spec);
      // The registry is now stale — pull the fresh list so the UI
      // shows the new row without the user hitting Rescan.
      await get().refreshLocalModels(true);
      // Give the stage-specific caches a nudge too so downstream
      // dropdowns (Whisper model list, Translation model list, TTS
      // voices) pick up the addition.
      if (spec.kind === "whisper") {
        void get().refreshWhisperModels();
      } else if (spec.kind === "translation") {
        void get().refreshTranslationModels();
      } else {
        void get().refreshTtsVoices();
      }
      return entry;
    } finally {
      set({ modelImportBusy: false });
    }
  },

  unloadAllModels: async () => {
    const released = await api.unloadAllModels();
    return released;
  },

  updateProjectModels: async (projectId, patch) => {
    const rec = await api.updateProjectModels(projectId, patch);
    set((state) => {
      if (state.currentProject?.id === projectId) {
        return { currentProject: rec };
      }
      return {};
    });
    return rec;
  },

  markFirstRunCompleted: async () => {
    if (get().settings?.firstRunCompleted) return;
    const settings = await api.updateSettings({ firstRunCompleted: true });
    set({ settings });
  },
}));
