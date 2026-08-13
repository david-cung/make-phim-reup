import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  mediaUrl,
  pickMediaFile,
  pickRenderOutputPath,
  pickSubtitleFile,
  pickSubtitleSavePath,
  pickTtsReferenceAudio,
} from "@/ipc/bridge";
import { useAppStore } from "@/state/store";
import { humanBytes } from "@/utils/format";
import { TopBar } from "../components/TopBar";
import { YouTubePanel } from "../components/YouTubePanel";
import {
  IconClose,
  IconExport,
  IconFolder,
  IconLayers,
  IconMedia,
  IconMic,
  IconPause,
  IconPlay,
  IconSettings,
  IconSparkles,
  IconSubtitles,
  IconWaveform,
  IconZoomIn,
  IconZoomOut,
} from "../components/icons";
import {
  defaultSttOptions,
  defaultTranslateOptions,
  isAppError,
  QUALITY_PROFILE_PRESETS,
  TRANSLATION_LANGUAGES,
  type AppError,
  type ExportKind,
  type JobSnapshot,
  type MixEnv,
  type MixSettings,
  type MixSummary,
  type OutputFormat,
  type PreviewMixResult,
  type PreviewResult,
  type PreviewSyncResult,
  type Project,
  type ProjectMediaState,
  type QualityProfile,
  type RenderEnv,
  type RenderSettings,
  type RenderSummary,
  type SubtitleMode,
  type SttEnv,
  type SttOptions,
  type SubtitleDoc,
  type SubtitleFormat,
  type SubtitleSegment,
  type SubtitleSegmentPatch,
  type SubtitleSummary,
  type SyncEnv,
  type SyncManifest,
  type SyncSegmentEntry,
  type SyncSettings,
  type SyncSummary,
  type TranscriptSummary,
  type TranslateOptions,
  type TranslationDoc,
  type TranslationEnv,
  type TranslationModelInfo,
  type TranslationRecommendedPreset,
  type TranslationSummary,
  type TtsEnv,
  type TtsManifest,
  type TtsRecommendedVoicePreset,
  type TtsSegmentEntry,
  type TtsSettings,
  type TtsSummary,
  type VideoMetadata,
  type VoiceInfo,
  type WhisperModelInfo,
} from "@/ipc/types";

// Whisper source-language picker. Same shape as
// `TRANSLATION_LANGUAGES` but prepended with "Auto detect" (Whisper
// listens to the first few seconds and infers), and Cantonese
// (`yue`) is listed because Whisper large-v3 supports it distinctly
// from Mandarin.
const LANGUAGE_OPTIONS: { code: string | null; label: string }[] = [
  { code: null, label: "Auto detect" },
  ...TRANSLATION_LANGUAGES.map((l) => ({ code: l.code, label: l.label })),
];

const STT_PROFILE_META: {
  id: QualityProfile;
  label: string;
  blurb: string;
}[] = [
  { id: "fast", label: "Fast", blurb: "Faster processing" },
  { id: "balanced", label: "Balanced", blurb: "Good quality, reasonable speed" },
  { id: "quality", label: "Quality", blurb: "Highest transcription accuracy" },
];

function whisperDisplayName(name: string): string {
  if (name === "large-v3" || name === "large") return "Whisper large-v3";
  if (!name) return "Whisper";
  return `Whisper ${name.charAt(0).toUpperCase()}${name.slice(1)}`;
}

function currentSttProfile(options: SttOptions): QualityProfile | "custom" {
  const p = options.qualityProfile;
  if (p === "fast" || p === "balanced" || p === "quality") return p;
  for (const id of STT_PROFILE_META) {
    if (QUALITY_PROFILE_PRESETS[id.id].model === options.model) return id.id;
  }
  return "custom";
}

// Phase 12 UX — after a model download is kicked off from a stage
// panel we need to know which job snapshot to poll. Download jobs
// are the ones without a `projectId` in the given stage; we pick
// the newest one whose `createdAt` is at or after the click. Falls
// back to any pending/running download if timestamps are unavailable.
// The stage argument lets Whisper (transcribe) and GGUF translation
// (translate) downloads share the same infrastructure.
function findLatestDownloadJobId(
  sinceMs: number,
  stage: JobSnapshot["stage"] = "transcribe",
): string | null {
  const jobs = Object.values(useAppStore.getState().jobsById);
  const candidates = jobs.filter((j) => !j.projectId && j.stage === stage);
  if (candidates.length === 0) return null;
  const parseTs = (s: string | null | undefined) => {
    if (!s) return 0;
    const t = Date.parse(s);
    return Number.isFinite(t) ? t : 0;
  };
  const withinWindow = candidates.filter(
    (j) => parseTs(j.createdAt) >= sinceMs - 5_000,
  );
  const pool = withinWindow.length > 0 ? withinWindow : candidates;
  pool.sort((a, b) => parseTs(b.createdAt) - parseTs(a.createdAt));
  return pool[0]?.id ?? null;
}

// Phase 12 UX — resolve when the given job reaches a terminal state
// (`completed` | `failed` | `cancelled`). Uses the zustand subscription
// so we don't spin on `setInterval`.
//
// The guard is an inactivity watchdog, not a wall-clock deadline: a
// total cap can't distinguish "still working" from "dead", and the
// honest upper bound is hours — translating a feature film with a
// CPU-bound quantised LLM, or rendering a long export, legitimately
// runs that long. A worker that dies silently, by contrast, stops
// reporting immediately. So we only give up after a long stretch with
// no progress event and no status change at all.
function waitForJobTerminal(
  jobId: string,
  inactivityMs = 15 * 60_000,
): Promise<JobSnapshot> {
  return new Promise((resolve, reject) => {
    const initial = useAppStore.getState().jobsById[jobId];
    if (initial && isTerminalStatus(initial.status)) {
      resolve(initial);
      return;
    }

    let timer: ReturnType<typeof setTimeout>;
    let lastStatus = initial?.status;
    let lastProgress = useAppStore.getState().jobProgress[jobId];

    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        unsub();
        reject(
          new Error(
            `No progress from this job for ${Math.round(
              inactivityMs / 60_000,
            )} minutes — the worker may have stopped. Check the logs, then try again.`,
          ),
        );
      }, inactivityMs);
    };

    const unsub = useAppStore.subscribe((state) => {
      const snap = state.jobsById[jobId];
      if (!snap) return;
      if (isTerminalStatus(snap.status)) {
        clearTimeout(timer);
        unsub();
        resolve(snap);
        return;
      }
      const progress = state.jobProgress[jobId];
      if (snap.status !== lastStatus || progress !== lastProgress) {
        lastStatus = snap.status;
        lastProgress = progress;
        arm();
      }
    });

    arm();
  });
}

function isTerminalStatus(status: JobSnapshot["status"]): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

export default function ProjectView() {
  const { id } = useParams<{ id: string }>();
  const openProject = useAppStore((s) => s.openProject);
  const project = useAppStore((s) => s.currentProject);
  const media = useAppStore((s) => s.currentMedia);
  const mediaLoading = useAppStore((s) => s.currentMediaLoading);
  const ffmpeg = useAppStore((s) => s.ffmpeg);
  const jobProgress = useAppStore((s) => s.jobProgress);
  const ttsProgressDetailByJob = useAppStore(
    (s) => s.ttsProgressDetailByJob,
  );
  const jobsById = useAppStore((s) => s.jobsById);
  const importMedia = useAppStore((s) => s.importMedia);
  const extractAudio = useAppStore((s) => s.extractAudio);
  const cancelJob = useAppStore((s) => s.cancelJob);

  const sttEnv = useAppStore((s) => s.sttEnv);
  const whisperModels = useAppStore((s) => s.whisperModels);
  const refreshSttEnv = useAppStore((s) => s.refreshSttEnv);
  const refreshWhisperModels = useAppStore((s) => s.refreshWhisperModels);
  const downloadWhisperModel = useAppStore((s) => s.downloadWhisperModel);
  const startTranscribe = useAppStore((s) => s.startTranscribe);
  // Phase 12 UX — needed for the "Offline Mode is blocking" recovery
  // path inside `handleTranscribe`. When a model download is refused
  // because Offline Mode is on, we prompt the user and, if they
  // agree, flip the setting off and retry the download in-place.
  const updateSettings = useAppStore((s) => s.updateSettings);

  const translationEnv = useAppStore((s) => s.translationEnv);
  const translationModels = useAppStore((s) => s.translationModels);
  const translationDoc = useAppStore((s) => s.currentTranslationDoc);
  const refreshTranslationEnv = useAppStore((s) => s.refreshTranslationEnv);
  const refreshTranslationModels = useAppStore((s) => s.refreshTranslationModels);
  const loadTranslationDoc = useAppStore((s) => s.loadTranslationDoc);
  const startTranslate = useAppStore((s) => s.startTranslate);
  const updateTranslationSegment = useAppStore(
    (s) => s.updateTranslationSegment,
  );
  // Phase 12 UX — mirrors the `refreshWhisperModels` +
  // `downloadWhisperModel` pair used by `handleTranscribe`. Users
  // with an empty translation directory shouldn't have to hunt for
  // a GGUF; clicking the primary CTA auto-pulls the recommended
  // model from HuggingFace and then runs translation.
  const translationRecommendedPresets = useAppStore(
    (s) => s.translationRecommendedPresets,
  );
  const refreshTranslationRecommendedPresets = useAppStore(
    (s) => s.refreshTranslationRecommendedPresets,
  );
  const downloadTranslationModel = useAppStore(
    (s) => s.downloadTranslationModel,
  );

  const subtitleDoc = useAppStore((s) => s.currentSubtitleDoc);
  const subtitleLoading = useAppStore((s) => s.currentSubtitleLoading);
  const loadSubtitles = useAppStore((s) => s.loadSubtitles);
  const rebuildSubtitles = useAppStore((s) => s.rebuildSubtitles);
  const updateSubtitleSegment = useAppStore((s) => s.updateSubtitleSegment);
  const assignSubtitleVoiceToSpeaker = useAppStore(
    (s) => s.assignSubtitleVoiceToSpeaker,
  );
  const addSubtitleSegment = useAppStore((s) => s.addSubtitleSegment);
  const deleteSubtitleSegment = useAppStore((s) => s.deleteSubtitleSegment);
  const splitSubtitleSegment = useAppStore((s) => s.splitSubtitleSegment);
  const mergeSubtitleSegment = useAppStore((s) => s.mergeSubtitleSegment);
  const importSubtitlesAction = useAppStore((s) => s.importSubtitles);
  const exportSubtitlesAction = useAppStore((s) => s.exportSubtitles);

  const ttsEnv = useAppStore((s) => s.ttsEnv);
  const ttsVoices = useAppStore((s) => s.ttsVoices);
  const ttsManifest = useAppStore((s) => s.currentTtsManifest);
  const ttsEngine = useAppStore((s) => s.ttsEngine);
  const ttsQualityMode = useAppStore((s) => s.ttsQualityMode);
  const ttsVoiceId = useAppStore((s) => s.ttsVoiceId);
  const ttsSettings = useAppStore((s) => s.ttsSettings);
  const lastTtsPreview = useAppStore((s) => s.lastTtsPreview);
  const refreshTtsEnv = useAppStore((s) => s.refreshTtsEnv);
  const refreshTtsVoices = useAppStore((s) => s.refreshTtsVoices);
  const loadTtsManifest = useAppStore((s) => s.loadTtsManifest);
  const refreshTtsSummary = useAppStore((s) => s.refreshTtsSummary);
  const setTtsEngine = useAppStore((s) => s.setTtsEngine);
  const setTtsQualityMode = useAppStore((s) => s.setTtsQualityMode);
  const setTtsVoiceId = useAppStore((s) => s.setTtsVoiceId);
  const setTtsSettings = useAppStore((s) => s.setTtsSettings);
  const previewTts = useAppStore((s) => s.previewTts);
  const startGenerateTts = useAppStore((s) => s.startGenerateTts);
  // Phase 12 UX — mirrors `translationRecommendedPresets` /
  // `downloadTranslationModel`. Lets "Generate all" auto-pull a
  // Piper voice matching the project's target language when the
  // user hasn't dropped one into `<models>/tts/piper/` yet.
  const ttsRecommendedVoices = useAppStore((s) => s.ttsRecommendedVoices);
  const refreshTtsRecommendedVoices = useAppStore(
    (s) => s.refreshTtsRecommendedVoices,
  );
  const downloadTtsVoice = useAppStore((s) => s.downloadTtsVoice);
  const createTtsVoiceProfile = useAppStore(
    (s) => s.createTtsVoiceProfile,
  );

  const syncEnv = useAppStore((s) => s.syncEnv);
  const syncManifest = useAppStore((s) => s.currentSyncManifest);
  const syncSettings = useAppStore((s) => s.syncSettings);
  const lastSyncPreview = useAppStore((s) => s.lastSyncPreview);
  const refreshSyncEnv = useAppStore((s) => s.refreshSyncEnv);
  const loadSyncManifest = useAppStore((s) => s.loadSyncManifest);
  const refreshSyncSummary = useAppStore((s) => s.refreshSyncSummary);
  const setSyncSettings = useAppStore((s) => s.setSyncSettings);
  const previewSync = useAppStore((s) => s.previewSync);
  const startApplySync = useAppStore((s) => s.startApplySync);

  const mixEnv = useAppStore((s) => s.mixEnv);
  const mixSettings = useAppStore((s) => s.mixSettings);
  const lastMixPreview = useAppStore((s) => s.lastMixPreview);
  const refreshMixEnv = useAppStore((s) => s.refreshMixEnv);
  const loadMixManifest = useAppStore((s) => s.loadMixManifest);
  const refreshMixSummary = useAppStore((s) => s.refreshMixSummary);
  const loadMixPreview = useAppStore((s) => s.loadMixPreview);
  const setMixSettings = useAppStore((s) => s.setMixSettings);
  const startApplyMix = useAppStore((s) => s.startApplyMix);

  const renderEnv = useAppStore((s) => s.renderEnv);
  const renderSettings = useAppStore((s) => s.renderSettings);
  const refreshRenderEnv = useAppStore((s) => s.refreshRenderEnv);
  const loadRenderManifest = useAppStore((s) => s.loadRenderManifest);
  const refreshRenderSummary = useAppStore((s) => s.refreshRenderSummary);
  const setRenderSettings = useAppStore((s) => s.setRenderSettings);
  const startApplyRender = useAppStore((s) => s.startApplyRender);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const environmentRefreshStartedRef = useRef(false);
  const [videoTime, setVideoTime] = useState(0);
  const seekVideo = useCallback((time: number) => {
    const el = videoRef.current;
    if (el) {
      el.currentTime = Math.max(0, time);
    }
  }, []);

  // Two kinds of failure, deliberately kept apart. `loadError` means the
  // project itself couldn't be opened, so there is no editor to show.
  // `error` is an operational failure from one stage — the editor stays
  // up and reports it inline, because replacing the whole workspace
  // (losing the video, the subtitles, the timeline) over a single failed
  // render is far more disruptive than the failure itself.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Success/no-op feedback. Without it, finishing a pipeline that had
  // nothing left to do looks identical to a button that does nothing.
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sttOptions, setSttOptions] = useState<SttOptions>(() =>
    defaultSttOptions(),
  );
  const [translateOptions, setTranslateOptions] = useState<TranslateOptions>(
    () =>
      defaultTranslateOptions({
        sourceLanguage: project?.sourceLanguage ?? "en",
        targetLanguage: project?.targetLanguage ?? "vi",
      }),
  );

  useEffect(() => {
    if (!id) return;
    setLoadError(null);
    setError(null);
    openProject(id).catch((e) => setLoadError(formatError(e)));
  }, [id, openProject]);

  useEffect(() => {
    if (environmentRefreshStartedRef.current) return;
    environmentRefreshStartedRef.current = true;
    // Environment/model inventories are process-wide, not project-wide.
    // Reuse snapshots already held by the store when navigating between
    // Settings and projects; each panel still exposes an explicit rescan.
    if (!sttEnv) {
      void refreshSttEnv();
      void refreshWhisperModels();
    }
    if (!translationEnv) {
      void refreshTranslationEnv();
      void refreshTranslationModels();
      void refreshTranslationRecommendedPresets();
    }
    if (!ttsEnv) {
      void refreshTtsEnv();
      void refreshTtsVoices();
      void refreshTtsRecommendedVoices();
    }
    if (!syncEnv) void refreshSyncEnv();
    if (!mixEnv) void refreshMixEnv();
    if (!renderEnv) void refreshRenderEnv();
  }, [
    sttEnv,
    refreshSttEnv,
    refreshWhisperModels,
    translationEnv,
    refreshTranslationEnv,
    refreshTranslationModels,
    refreshTranslationRecommendedPresets,
    ttsEnv,
    refreshTtsEnv,
    refreshTtsVoices,
    refreshTtsRecommendedVoices,
    syncEnv,
    refreshSyncEnv,
    mixEnv,
    refreshMixEnv,
    renderEnv,
    refreshRenderEnv,
  ]);

  useEffect(() => {
    if (!id) return;
    void loadTranslationDoc(id);
    void loadSubtitles(id);
    void loadTtsManifest(id);
    void loadSyncManifest(id);
    void loadMixManifest(id);
    void loadMixPreview(id);
    void loadRenderManifest(id);
  }, [
    id,
    loadTranslationDoc,
    loadSubtitles,
    loadTtsManifest,
    loadSyncManifest,
    loadMixManifest,
    loadMixPreview,
    loadRenderManifest,
  ]);

  // When the user changes engine/voice/settings, refresh the summary so
  // the "N/M generated" counters reflect the *current* cache identity.
  useEffect(() => {
    if (!id) return;
    const timer = window.setTimeout(() => {
      void refreshTtsSummary(id);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [id, ttsEngine, ttsVoiceId, ttsSettings, refreshTtsSummary]);

  // Same for sync settings — coverage counts depend on min/max speed.
  useEffect(() => {
    if (!id) return;
    const timer = window.setTimeout(() => {
      void refreshSyncSummary(id);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [id, syncSettings, refreshSyncSummary]);

  // And for mix settings — status (Ready/Stale/Missing) depends on
  // volume + ducking, since those feed into the cache key.
  useEffect(() => {
    if (!id) return;
    const timer = window.setTimeout(() => {
      void refreshMixSummary(id);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [id, mixSettings, refreshMixSummary]);

  // Render settings feed into the cache key too — subtitle mode,
  // output format and codecs all move the Ready/Stale/Missing needle.
  useEffect(() => {
    if (!id) return;
    const timer = window.setTimeout(() => {
      void refreshRenderSummary(id);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [id, renderSettings, refreshRenderSummary]);

  useEffect(() => {
    if (!sttEnv) return;
    setSttOptions((prev) => {
      if (prev.device != null) return prev;
      return { ...prev, device: sttEnv.defaultDevice };
    });
  }, [sttEnv]);

  // Pre-select an installed GGUF the first time we see them.
  useEffect(() => {
    if (!translationModels.length) return;
    setTranslateOptions((prev) => {
      if (prev.model) return prev;
      const pick =
        translationModels.find((m) => m.isDefault) ?? translationModels[0];
      return { ...prev, model: pick.name };
    });
  }, [translationModels]);

  // Pass the project's source language to Whisper unless the user
  // already picked a language (including Auto detect after load).
  useEffect(() => {
    if (!project) return;
    const lang = project.sourceLanguage?.trim();
    if (!lang || lang === "auto" || lang === "und") return;
    setSttOptions((prev) => {
      if (prev.language != null) return prev;
      return { ...prev, language: lang };
    });
  }, [project?.id, project?.sourceLanguage]);

  // Sync language dropdowns from the project defaults when they land.
  useEffect(() => {
    if (!project) return;
    setTranslateOptions((prev) => ({
      ...prev,
      sourceLanguage: prev.sourceLanguage || project.sourceLanguage,
      targetLanguage: prev.targetLanguage || project.targetLanguage,
    }));
  }, [project]);

  const activeExtractionJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "extract_audio" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeTranscribeJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "transcribe" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeDownloadJob: JobSnapshot | null = useMemo(() => {
    for (const snap of Object.values(jobsById)) {
      if (
        !snap.projectId &&
        snap.stage === "transcribe" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById]);

  // Phase 12 — translation model downloads live entirely in-memory
  // (no `projectId`) but share the `translate` stage with real
  // translation jobs. Filtering on `!snap.projectId` disambiguates
  // the two so the download bar and the translate-progress bar
  // don't stomp on each other in the UI.
  const activeTranslateDownloadJob: JobSnapshot | null = useMemo(() => {
    for (const snap of Object.values(jobsById)) {
      if (
        !snap.projectId &&
        snap.stage === "translate" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById]);

  // Phase 12 — same disambiguation for TTS: an in-memory download
  // job carries `stage: "tts"` and empty `projectId`, so it must
  // not be confused with a real per-project synthesis job.
  const activeTtsDownloadJob: JobSnapshot | null = useMemo(() => {
    for (const snap of Object.values(jobsById)) {
      if (
        !snap.projectId &&
        snap.stage === "tts" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById]);

  const activeTranslateJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "translate" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeTtsJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "tts" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeSyncJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "sync" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeMixJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "mix" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  const activeRenderJob: JobSnapshot | null = useMemo(() => {
    if (!project) return null;
    for (const snap of Object.values(jobsById)) {
      if (
        snap.projectId === project.id &&
        snap.stage === "render" &&
        (snap.status === "running" || snap.status === "queued")
      ) {
        return snap;
      }
    }
    return null;
  }, [jobsById, project]);

  // -----------------------------------------------------------------------
  // UI-only state (introduced by the professional editor redesign).
  //
  // These must stay above the `!project` early return below: React
  // requires an identical hook sequence on every render, and `project`
  // is null on the first paint while `openProject` resolves.
  //
  // The rest of ProjectView still owns every backend action and store
  // subscription; these flags only drive which panels are visible in
  // the workspace and which subtitle the inspector describes.
  // -----------------------------------------------------------------------
  const [section, setSection] = useState<EditorSection>("media");
  const [selectedSubtitleId, setSelectedSubtitleId] = useState<number | null>(
    null,
  );

  // "Run all" pipeline state. `pipelineStep` doubles as the running
  // flag (non-null while the chain is in flight) and as the label the
  // topbar button shows. The abort flag is a ref rather than state so
  // the async chain reads the latest value without being restarted by
  // a re-render.
  const [pipelineStep, setPipelineStep] = useState<string | null>(null);
  const pipelineAbortRef = useRef(false);

  const subtitleById = useMemo(
    () =>
      new Map(
        (subtitleDoc?.segments ?? []).map((segment) => [segment.id, segment]),
      ),
    [subtitleDoc],
  );

  // Resolve the playhead once. The old render path linearly scanned the
  // same subtitle array for the overlay, inspector, and editor.
  const activeSubtitleFromTime = useMemo(
    () => findSubtitleAtTime(subtitleDoc, videoTime),
    [subtitleDoc, videoTime],
  );
  const effectiveSubtitleId =
    selectedSubtitleId ?? activeSubtitleFromTime?.id ?? null;
  const handleSelectSubtitle = useCallback(
    (subtitleId: number) => {
      setSelectedSubtitleId(subtitleId);
      const segment = subtitleById.get(subtitleId);
      if (segment) seekVideo(segment.start);
    },
    [seekVideo, subtitleById],
  );

  const sourceMediaPath = project?.sourceMediaPath ?? null;

  const workflow = useMemo(
    () =>
      computeWorkflow({
        hasSource: !!sourceMediaPath,
        hasAudio: !!media?.audioAbsolutePath,
        hasTranscript: !!media?.transcript,
        translationRatio:
          media?.translation && media?.transcript
            ? media.translation.translatedCount /
              Math.max(1, media.translation.segmentCount)
            : 0,
        ttsRatio:
          media?.tts && media.tts.subtitleCount > 0
            ? media.tts.generatedCount / Math.max(1, media.tts.subtitleCount)
            : 0,
        syncRatio:
          media?.sync && media.sync.subtitleCount > 0
            ? media.sync.syncedCount / Math.max(1, media.sync.subtitleCount)
            : 0,
        mixReady: media?.mix?.status === "ready",
        renderReady: media?.render?.status === "ready",
        active: {
          extract: !!activeExtractionJob,
          transcribe: !!activeTranscribeJob,
          translate: !!activeTranslateJob,
          tts: !!activeTtsJob,
          sync: !!activeSyncJob,
          mix: !!activeMixJob,
          render: !!activeRenderJob,
        },
      }),
    [
      sourceMediaPath,
      media,
      activeExtractionJob,
      activeTranscribeJob,
      activeTranslateJob,
      activeTtsJob,
      activeSyncJob,
      activeMixJob,
      activeRenderJob,
    ],
  );

  const activeProcessing = useMemo(
    () =>
      [
        activeExtractionJob && {
          label: "Extracting audio",
          job: activeExtractionJob,
        },
        activeTranscribeJob && {
          label: "Transcribing",
          job: activeTranscribeJob,
        },
        activeTranslateJob && {
          label: "Translating",
          job: activeTranslateJob,
        },
        activeTtsJob && { label: "Generating voice", job: activeTtsJob },
        activeSyncJob && { label: "Syncing voice", job: activeSyncJob },
        activeMixJob && { label: "Mixing audio", job: activeMixJob },
        activeRenderJob && { label: "Rendering movie", job: activeRenderJob },
        activeDownloadJob && {
          label: "Downloading Whisper model",
          job: activeDownloadJob,
        },
        activeTranslateDownloadJob && {
          label: "Downloading translation model",
          job: activeTranslateDownloadJob,
        },
        activeTtsDownloadJob && {
          label: "Downloading voice",
          job: activeTtsDownloadJob,
        },
      ].filter(Boolean) as { label: string; job: JobSnapshot }[],
    [
      activeExtractionJob,
      activeTranscribeJob,
      activeTranslateJob,
      activeTtsJob,
      activeSyncJob,
      activeMixJob,
      activeRenderJob,
      activeDownloadJob,
      activeTranslateDownloadJob,
      activeTtsDownloadJob,
    ],
  );

  // Video preview helpers — the dark stage toolbar owns play/pause and
  // drives the underlying <video> ref directly, so we don't have to
  // fork the existing VideoPreview component.
  const [isPlaying, setIsPlaying] = useState(false);

  // Which file the stage plays. The source carries the original
  // soundtrack and no subtitles, so it can never show whether the dub or
  // the subtitles came out right — that only lives in the render. Offer
  // both and default to the finished film once there is one.
  const [previewTarget, setPreviewTarget] = useState<"source" | "result">(
    "result",
  );
  const renderedPath =
    media?.render?.status === "ready" ? media.render.absolutePath : null;
  const showingResult = previewTarget === "result" && !!renderedPath;
  const previewPath = showingResult
    ? renderedPath
    : (project?.sourceMediaPath ?? null);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("ended", onPause);
    return () => {
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("ended", onPause);
    };
  }, [previewPath]);
  const togglePlayback = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  if (loadError) {
    return (
      <div className="app-shell">
        <TopBar showBackToDashboard showDefaultTools={false} />
        <main className="app-body plain">
          <div className="error-panel">
            <h2>Cannot open project</h2>
            <pre>{loadError}</pre>
            <Link to="/" className="btn" style={{ marginTop: 12 }}>
              Back to dashboard
            </Link>
          </div>
        </main>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="app-shell">
        <TopBar showBackToDashboard showDefaultTools={false} />
        <main className="app-body plain">
          <div className="loading">Loading project…</div>
        </main>
      </div>
    );
  }

  const handleImport = async (copy: boolean) => {
    setError(null);
    const path = await pickMediaFile();
    if (!path) return;
    setBusy(true);
    try {
      await importMedia({
        projectId: project.id,
        sourcePath: path,
        copyIntoProject: copy,
      });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleExtract = async () => {
    setError(null);
    setBusy(true);
    try {
      await extractAudio(project.id);
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCancel = async () => {
    if (!activeExtractionJob) return;
    try {
      await cancelJob(activeExtractionJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  /// Phase 12 UX — one-click prerequisite: extract the WAV before
  /// running STT so the user doesn't have to bounce between the
  /// Audio and Speech recognition panels. Returns `true` if audio
  /// is ready to use, `false` if the user cancelled the extract, or
  /// throws on hard failure (FFmpeg missing, disk full, ...). The
  /// helper is intentionally cheap and idempotent — `refreshMedia`
  /// on cache-hit costs one IPC round-trip.
  const ensureAudioExtracted = async (): Promise<boolean> => {
    if (!project) return false;
    const currentMedia = useAppStore.getState().currentMedia;
    if (currentMedia?.audioAbsolutePath) return true;
    await extractAudio(project.id);
    // Pick the freshest extract_audio job for this project — if
    // `extractAudio` hit the cache the store already refreshed
    // `currentMedia` and we skip the wait entirely.
    const jobs = Object.values(useAppStore.getState().jobsById);
    const parseTs = (s: string | null | undefined) =>
      s ? Date.parse(s) || 0 : 0;
    const extract = jobs
      .filter(
        (j) => j.projectId === project.id && j.stage === "extract_audio",
      )
      .sort((a, b) => parseTs(b.createdAt) - parseTs(a.createdAt))[0];
    if (extract && !isTerminalStatus(extract.status)) {
      const result = await waitForJobTerminal(extract.id);
      if (result.status === "cancelled") return false;
      if (result.status === "failed") {
        throw new Error(
          result.errorMessage ||
            `Audio extraction failed${
              result.errorCode ? ` — ${result.errorCode}` : ""
            }.`,
        );
      }
      // `onJobUpdate` refreshes media on completion, but fire-and-forget
      // — and `waitForJobTerminal` resolves off that same event. Reading
      // `currentMedia` now would race the un-awaited IPC round-trip and
      // see a stale snapshot, so refresh again and await it.
      await useAppStore.getState().refreshMedia(project.id);
    }
    const after = useAppStore.getState().currentMedia;
    if (!after?.audioAbsolutePath) {
      throw new Error(
        "Audio extraction finished but no audio file was produced. Check that FFmpeg is installed and the video has an audio track.",
      );
    }
    return true;
  };

  /// Phase 12 UX — make sure the Whisper model the user picked is on
  /// disk before we ask the worker to transcribe with it. Returns
  /// `true` when the model is ready, `false` when the user declined
  /// the download (or cancelled it), and throws on hard failure.
  const ensureWhisperModelReady = async (): Promise<boolean> => {
    const chosen = whisperModels.find((m) => m.name === sttOptions.model);
    if (!chosen || chosen.installed) return true;

    const startedAt = Date.now();
    try {
      await downloadWhisperModel(chosen.name);
    } catch (dlErr) {
      // Offline Mode blocks the ONE HTTP entry point in the app.
      // Rather than showing the raw error and sending the user off to
      // Settings, offer to flip it off in-place and retry — otherwise
      // this is a dead end.
      if (isAppError(dlErr) && dlErr.code === "MODEL_NETWORK_DISABLED") {
        const proceed = window.confirm(
          `To download the "${chosen.name}" model, the app needs one-time network access.\n\n` +
            `Offline Mode is currently ON, which is blocking the download.\n\n` +
            `Turn Offline Mode OFF and download now?\n` +
            `(You can turn it back on in Settings once the download finishes — the model works fully offline afterwards.)`,
        );
        if (!proceed) return false;
        await updateSettings({ offlineMode: false });
        await downloadWhisperModel(chosen.name);
      } else {
        throw dlErr;
      }
    }

    const dlJobId = findLatestDownloadJobId(startedAt);
    if (dlJobId) {
      const result = await waitForJobTerminal(dlJobId);
      if (result.status === "cancelled") return false;
      if (result.status === "failed") {
        throw new Error(
          result.errorMessage ||
            `Model download failed (${chosen.name}${
              result.errorCode ? ` — ${result.errorCode}` : ""
            }).`,
        );
      }
    }
    await refreshWhisperModels();
    const after = useAppStore
      .getState()
      .whisperModels.find((m) => m.name === chosen.name);
    if (!after?.installed) {
      throw new Error(
        `Model ${chosen.name} did not appear installed after download.`,
      );
    }
    return true;
  };

  const handleTranscribe = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      // Phase 12 UX — chain the two mandatory prerequisites so the
      // user only has to click "Transcribe": extract the 16 kHz mono
      // WAV Whisper needs, then pull the model if it isn't on disk.
      // Both steps surface through the existing progress UI.
      const audioReady = await ensureAudioExtracted();
      if (!audioReady) return;
      const modelReady = await ensureWhisperModelReady();
      if (!modelReady) return;
      await startTranscribe(project.id, sttOptions);
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCancelTranscribe = async () => {
    if (!activeTranscribeJob) return;
    try {
      await cancelJob(activeTranscribeJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleDownloadModel = async (name: string) => {
    setError(null);
    setBusy(true);
    try {
      // Phase 12 UX — mirror the recovery/awaiting logic from
      // `handleTranscribe`. Without awaiting the download job's
      // terminal state a background failure (missing
      // `huggingface_hub`, wrong repo, disk full, etc.) never
      // surfaces to the user — the primary button just silently
      // becomes clickable again. We surface those failures as an
      // error banner and, for the Offline-Mode block specifically,
      // offer the one-click "turn off & retry" path.
      const startedAt = Date.now();
      const runDownload = async () => {
        await downloadWhisperModel(name);
        const dlJobId = findLatestDownloadJobId(startedAt);
        if (!dlJobId) return;
        const result = await waitForJobTerminal(dlJobId);
        if (result.status === "failed") {
          throw new Error(
            result.errorMessage ||
              `Model download failed (${name}${
                result.errorCode ? ` — ${result.errorCode}` : ""
              }).`,
          );
        }
      };
      try {
        await runDownload();
      } catch (e) {
        if (isAppError(e) && e.code === "MODEL_NETWORK_DISABLED") {
          const proceed = window.confirm(
            `To download the "${name}" model, the app needs one-time network access.\n\n` +
              `Offline Mode is currently ON, which is blocking the download.\n\n` +
              `Turn Offline Mode OFF and download now?`,
          );
          if (!proceed) return;
          await updateSettings({ offlineMode: false });
          await runDownload();
        } else {
          throw e;
        }
      }
      await refreshWhisperModels();
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCancelDownload = async () => {
    if (!activeDownloadJob) return;
    try {
      await cancelJob(activeDownloadJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  /// Phase 12 UX — mirror of `ensureWhisperModelReady` for the GGUF
  /// side. If the user has no translation model installed (or the one
  /// they picked has disappeared from disk) we pull the recommended
  /// preset from HuggingFace. Returns the `TranslateOptions` to run
  /// with — the caller must use these rather than `translateOptions`,
  /// because a freshly downloaded model has to be wired into `.model`
  /// and React state won't have committed yet. Returns `null` when
  /// the user cancelled.
  const ensureTranslationModelReady =
    async (): Promise<TranslateOptions | null> => {
      const chosenModelPresent =
        !!translateOptions.model &&
        translationModels.some((m) => m.name === translateOptions.model);
      if (chosenModelPresent) return translateOptions;

      // `is_default` in the Python registry marks exactly one entry as
      // the preferred first-time download (a ~2 GB Qwen 2.5 3B, a good
      // quality/size balance on any modern Mac).
      const preset =
        translationRecommendedPresets.find((p) => p.isDefault) ??
        translationRecommendedPresets[0];
      if (!preset) {
        throw new Error(
          "No translation model is installed and no recommended presets are available. Drop a GGUF file into the translation models directory manually.",
        );
      }

      const startedAt = Date.now();
      const runDownload = async () => {
        await downloadTranslationModel(preset.preset);
        const dlJobId = findLatestDownloadJobId(startedAt, "translate");
        if (!dlJobId) return;
        const result = await waitForJobTerminal(dlJobId);
        if (result.status === "cancelled") {
          // Stable sentinel so the caller can bail silently instead of
          // turning a deliberate cancel into a red error banner.
          const err = new Error("cancelled");
          (err as { code?: string }).code = "USER_CANCELLED";
          throw err;
        }
        if (result.status === "failed") {
          throw new Error(
            result.errorMessage ||
              `Model download failed (${preset.label}${
                result.errorCode ? ` — ${result.errorCode}` : ""
              }).`,
          );
        }
      };

      try {
        try {
          await runDownload();
        } catch (dlErr) {
          if (isAppError(dlErr) && dlErr.code === "MODEL_NETWORK_DISABLED") {
            const proceed = window.confirm(
              `To download the translation model "${preset.label}", the app needs one-time network access.\n\n` +
                `Offline Mode is currently ON, which is blocking the download.\n\n` +
                `Turn Offline Mode OFF and download now?\n` +
                `(You can turn it back on in Settings once the download finishes — the model works fully offline afterwards.)`,
            );
            if (!proceed) return null;
            await updateSettings({ offlineMode: false });
            await runDownload();
          } else {
            throw dlErr;
          }
        }
      } catch (e) {
        if ((e as { code?: string })?.code === "USER_CANCELLED") return null;
        throw e;
      }

      const fresh = await refreshTranslationModels();
      const installed =
        fresh.find((m) => m.name === preset.filename) ??
        fresh.find((m) => m.isDefault) ??
        fresh[0];
      if (!installed) {
        throw new Error(
          `Model ${preset.filename} did not appear installed after download.`,
        );
      }
      const nextOpts = { ...translateOptions, model: installed.name };
      setTranslateOptions(nextOpts);
      return nextOpts;
    };

  const handleTranslate = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      const opts = await ensureTranslationModelReady();
      if (!opts) return;
      await startTranslate(project.id, opts);
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCancelTranslate = async () => {
    if (!activeTranslateJob) return;
    try {
      await cancelJob(activeTranslateJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleCancelTranslateDownload = async () => {
    if (!activeTranslateDownloadJob) return;
    try {
      await cancelJob(activeTranslateDownloadJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleUpdateSegment = async (segmentId: number, value: string) => {
    if (!project) return;
    try {
      await updateTranslationSegment(project.id, segmentId, value);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleRebuildSubtitles = async () => {
    if (!project) return;
    setError(null);
    try {
      await rebuildSubtitles(project.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitlePatch = async (
    segmentId: number,
    patch: SubtitleSegmentPatch,
  ) => {
    if (!project) return;
    try {
      await updateSubtitleSegment(project.id, segmentId, patch);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleAdd = async (
    afterId: number | null,
    start: number,
    end: number,
  ) => {
    if (!project) return;
    try {
      await addSubtitleSegment(project.id, afterId, start, end);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleDelete = async (segmentId: number) => {
    if (!project) return;
    try {
      await deleteSubtitleSegment(project.id, segmentId);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleSplit = async (segmentId: number, splitTime: number) => {
    if (!project) return;
    try {
      await splitSubtitleSegment(project.id, segmentId, splitTime);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleMerge = async (segmentId: number) => {
    if (!project) return;
    try {
      await mergeSubtitleSegment(project.id, segmentId);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleImport = async () => {
    if (!project) return;
    setError(null);
    const path = await pickSubtitleFile();
    if (!path) return;
    try {
      await importSubtitlesAction(project.id, path);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSubtitleExport = async (
    format: SubtitleFormat,
    kind: ExportKind,
  ) => {
    if (!project) return;
    setError(null);
    const suggested = `${project.name.replace(/[^A-Za-z0-9._-]+/g, "_") || "subtitles"}.${format}`;
    const path = await pickSubtitleSavePath(suggested, format);
    if (!path) return;
    try {
      await exportSubtitlesAction(project.id, path, format, kind);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleTtsPreview = async (segmentId: number) => {
    if (!project) return;
    setError(null);
    try {
      const result = await previewTts(project.id, segmentId);
      // Autoplay the resulting file. The <audio> element is rendered
      // by the TTS panel so we just point it at the fresh URL.
      const audio = document.getElementById("tts-preview-audio");
      const url = mediaUrl(result.absolutePath);
      if (audio instanceof HTMLAudioElement && url) {
        audio.src = url;
        audio.currentTime = 0;
        void audio.play();
      }
    } catch (e) {
      setError(formatError(e));
    }
  };

  /// Phase 12 UX — parity with `handleTranscribe` /
  /// `handleTranslate`. If the user has no Piper voice installed
  /// (fresh project, no manual GGUF drop) we transparently pull the
  /// preset that best matches the project's target language before
  /// starting synthesis. Returns `true` iff callers should proceed
  /// with generation, `false` if the user cancelled or nothing is
  /// installable.
  const ensureTtsVoiceReady = async (): Promise<boolean> => {
    if (
      ttsVoiceId &&
      ttsVoices.some((v) => v.id === ttsVoiceId && v.engine === ttsEngine)
    ) {
      return true;
    }
    if (ttsVoiceId) {
      throw new Error(
        `The selected voice "${ttsVoiceId}" is not installed for ${ttsEngine}. Rescan or choose another voice; it will not be replaced automatically.`,
      );
    }
    if (ttsEngine === "f5-vietnamese") {
      if (!ttsEnv?.f5RuntimeInstalled) {
        throw new Error(
          "F5-TTS runtime is not installed. Run scripts/setup-f5.ps1 first.",
        );
      }
      if (!ttsEnv.f5Model.installed) {
        throw new Error(
          "F5-TTS QUALITY model is not installed. Install it explicitly in Voice Over settings.",
        );
      }
      throw new Error(
        "Create and select an F5 reference voice profile before generating. QUALITY never falls back to Piper automatically.",
      );
    }
    const targetLang = (project?.targetLanguage || "vi").toLowerCase();
    const enginePresets = ttsRecommendedVoices.filter(
      (candidate) => candidate.engine === ttsEngine,
    );
    const preset =
      enginePresets.find((p) => p.targetLanguages.includes(targetLang)) ??
      enginePresets.find((p) => p.isDefault) ??
      enginePresets[0];
    if (!preset) {
      throw new Error(
        "No TTS voice is installed and no recommended presets are available. Drop a Piper .onnx voice into <models>/tts/piper/ manually.",
      );
    }
    const startedAt = Date.now();
    const runDownload = async () => {
      await downloadTtsVoice(preset.preset);
      const dlJobId = findLatestDownloadJobId(startedAt, "tts");
      if (!dlJobId) return;
      const result = await waitForJobTerminal(dlJobId);
      if (result.status === "cancelled") {
        const err = new Error("cancelled");
        (err as { code?: string }).code = "USER_CANCELLED";
        throw err;
      }
      if (result.status === "failed") {
        throw new Error(
          result.errorMessage ||
            `Voice download failed (${preset.label}${
              result.errorCode ? ` — ${result.errorCode}` : ""
            }).`,
        );
      }
    };
    try {
      try {
        await runDownload();
      } catch (dlErr) {
        if (isAppError(dlErr) && dlErr.code === "MODEL_NETWORK_DISABLED") {
          const proceed = window.confirm(
            `To download the voice "${preset.label}", the app needs one-time network access.\n\n` +
              `Offline Mode is currently ON, which is blocking the download.\n\n` +
              `Turn Offline Mode OFF and download now?\n` +
              `(You can turn it back on in Settings once the download finishes — the voice works fully offline afterwards.)`,
          );
          if (!proceed) return false;
          await updateSettings({ offlineMode: false });
          await runDownload();
        } else {
          throw dlErr;
        }
      }
    } catch (e) {
      if ((e as { code?: string })?.code === "USER_CANCELLED") return false;
      throw e;
    }
    const fresh = await refreshTtsVoices();
    const installed =
      fresh.find((v) => v.id === preset.voiceId) ??
      fresh.find((v) => v.engine === preset.engine) ??
      fresh[0];
    if (!installed) {
      throw new Error(
        `Voice ${preset.voiceId} did not appear installed after download.`,
      );
    }
    // The store's `refreshTtsVoices` already auto-selects the first
    // voice when `ttsVoiceId` is empty, but we set it explicitly
    // too so a subsequent generate call sees the right id even in
    // the edge case where the store ran the auto-select against an
    // older snapshot.
    if (installed.id !== ttsVoiceId) {
      setTtsVoiceId(installed.id);
    }
    if (installed.engine !== ttsEngine) {
      setTtsEngine(installed.engine);
    }
    return true;
  };

  const handleTtsGenerateMissing = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      const ok = await ensureTtsVoiceReady();
      if (!ok) return;
      await startGenerateTts(project.id, { kind: "missing" });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleTtsGenerateAll = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      const ok = await ensureTtsVoiceReady();
      if (!ok) return;
      await startGenerateTts(project.id, { kind: "all" });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCancelTtsDownload = async () => {
    if (!activeTtsDownloadJob) return;
    try {
      await cancelJob(activeTtsDownloadJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleTtsRegenerate = async (segmentId: number) => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      // Phase 12 UX — per-row "Regenerate voice" is the fastest
      // path a user has to preview a single subtitle change, and
      // it must Just Work. Mirror the panel-level auto-download so
      // regenerating a single line on a fresh install downloads
      // the recommended Piper voice on-the-fly rather than
      // silently no-op'ing.
      const voiceReady = await ensureTtsVoiceReady();
      if (!voiceReady) return;
      await startGenerateTts(project.id, {
        kind: "selected",
        ids: [segmentId],
      });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleTtsCancel = async () => {
    if (!activeTtsJob) return;
    try {
      await cancelJob(activeTtsJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSyncPreview = async (segmentId: number) => {
    if (!project) return;
    setError(null);
    try {
      const result = await previewSync(project.id, segmentId);
      const audio = document.getElementById("sync-preview-audio");
      const url = mediaUrl(result.absolutePath);
      if (audio instanceof HTMLAudioElement && url) {
        audio.src = url;
        audio.currentTime = 0;
        void audio.play();
      }
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSyncApplyMissing = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      await startApplySync(project.id, { kind: "missing" });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSyncApplyAll = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      await startApplySync(project.id, { kind: "all" });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSyncRegenerate = async (segmentId: number) => {
    if (!project) return;
    setError(null);
    try {
      await startApplySync(project.id, {
        kind: "selected",
        ids: [segmentId],
      });
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleSyncCancel = async () => {
    if (!activeSyncJob) return;
    try {
      await cancelJob(activeSyncJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleMixApply = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      await startApplyMix(project.id);
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleMixCancel = async () => {
    if (!activeMixJob) return;
    try {
      await cancelJob(activeMixJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleMixPlay = () => {
    const audio = document.getElementById("mix-preview-audio");
    const url = lastMixPreview ? mediaUrl(lastMixPreview.absolutePath) : null;
    if (audio instanceof HTMLAudioElement && url) {
      audio.src = url;
      audio.currentTime = 0;
      void audio.play();
    }
  };

  const handleRenderApply = async (force = false) => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      await startApplyRender(project.id, { force });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRenderCancel = async () => {
    if (!activeRenderJob) return;
    try {
      await cancelJob(activeRenderJob.id);
    } catch (e) {
      setError(formatError(e));
    }
  };

  const chooseRenderOutputPath = async (): Promise<string | null> => {
    const ext: OutputFormat = renderSettings.outputFormat;
    const defaultName = `movie_vi.${ext}`;
    const path = await pickRenderOutputPath(
      renderSettings.outputPath ?? defaultName,
      ext,
    );
    if (path) {
      const normalized = path.toLowerCase().endsWith(`.${ext}`)
        ? path
        : `${path}.${ext}`;
      setRenderSettings({ outputPath: normalized });
      return normalized;
    }
    return null;
  };

  const handleRenderPickOutput = async () => {
    try {
      await chooseRenderOutputPath();
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleRenderClearOutput = () => {
    setRenderSettings({ outputPath: null });
  };

  // Export CTA in the topbar. Pick the destination first; cancelling the
  // native dialog must not start an expensive pipeline. Once confirmed,
  // store the absolute path in render settings before driving every
  // missing stage through to the final movie.
  const handleExport = async () => {
    if (!project.sourceMediaPath) {
      setError("Import a video before exporting.");
      setSection("media");
      return;
    }
    try {
      const outputPath = await chooseRenderOutputPath();
      if (!outputPath) return;

      setSection("render");
      requestAnimationFrame(() => {
        document
          .getElementById("panel-render")
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      await runPipeline({ upTo: "render" });
    } catch (e) {
      setError(formatError(e));
    }
  };

  // -----------------------------------------------------------------------
  // "Run all" — drive the whole pipeline from one click.
  //
  // Each `start*` store action shares the same contract: it resolves to
  // a summary when the work was already cached, or to `null` after
  // seeding a job snapshot. `runStage` uses that to decide whether it
  // needs to wait, so stages that are already up to date cost one IPC
  // round-trip and are skipped instantly. That makes the whole chain
  // resumable — re-running after a failure picks up where it stopped
  // rather than redoing finished work.
  // -----------------------------------------------------------------------
  const PIPELINE_CANCELLED = "PIPELINE_CANCELLED";

  const throwIfAborted = () => {
    if (pipelineAbortRef.current) {
      const err = new Error("cancelled");
      (err as { code?: string }).code = PIPELINE_CANCELLED;
      throw err;
    }
  };

  /// Start one stage and block until its job reaches a terminal state.
  /// A non-null result from `start` means the backend served it from
  /// cache and no job exists to wait on. Resolves `true` when real work
  /// ran, `false` when the stage was already up to date.
  const runStage = async (
    stage: JobSnapshot["stage"],
    start: () => Promise<unknown>,
  ): Promise<boolean> => {
    throwIfAborted();
    const cached = await start();
    if (cached != null) return false;

    const parseTs = (s: string | null | undefined) =>
      s ? Date.parse(s) || 0 : 0;
    const job = Object.values(useAppStore.getState().jobsById)
      .filter((j) => j.projectId === project.id && j.stage === stage)
      .sort((a, b) => parseTs(b.createdAt) - parseTs(a.createdAt))[0];
    if (!job || isTerminalStatus(job.status)) return false;

    const result = await waitForJobTerminal(job.id);
    if (result.status === "cancelled") {
      const err = new Error("cancelled");
      (err as { code?: string }).code = PIPELINE_CANCELLED;
      throw err;
    }
    if (result.status === "failed") {
      throw new Error(
        result.errorMessage ||
          `${stage} failed${result.errorCode ? ` — ${result.errorCode}` : ""}.`,
      );
    }
    // Same race as in `ensureAudioExtracted`: the media refresh that
    // `onJobUpdate` triggers on completion isn't awaited, and this
    // promise resolves off that same event. Every "is this stage
    // already done?" check below depends on the refreshed manifest, so
    // settle it here before returning to the caller.
    await useAppStore.getState().refreshMedia(project.id);
    return true;
  };

  const runPipeline = async (opts?: { upTo?: "render" }) => {
    if (!project.sourceMediaPath) {
      setError("Import a video first, then run the pipeline.");
      setSection("media");
      return;
    }
    if (pipelineStep) return;
    pipelineAbortRef.current = false;
    setError(null);
    setNotice(null);
    setBusy(true);
    // Distinguishes "nothing was left to do" from "the click did
    // nothing", so we can always report an outcome.
    let ranSomething = false;
    try {
      const fresh = () => useAppStore.getState().currentMedia;

      setPipelineStep("Extracting audio");
      setSection("media");
      const audioReady = await ensureAudioExtracted();
      if (!audioReady) return;
      throwIfAborted();

      if (!fresh()?.transcript) {
        setPipelineStep("Transcribing");
        setSection("transcription");
        const modelReady = await ensureWhisperModelReady();
        if (!modelReady) return;
        if (await runStage("transcribe", () =>
          startTranscribe(project.id, sttOptions),
        )) {
          ranSomething = true;
        }
      }

      const tr = fresh()?.translation;
      const needsTranslation =
        !tr || tr.translatedCount < tr.segmentCount || tr.segmentCount === 0;
      if (needsTranslation) {
        setPipelineStep("Translating");
        setSection("translation");
        const opts = await ensureTranslationModelReady();
        if (!opts) return;
        if (await runStage("translate", () => startTranslate(project.id, opts))) {
          ranSomething = true;
        }
      }

      // `onJobUpdate` auto-rebuilds subtitles once translation lands,
      // but that fires off a promise we can't await from here — and it
      // never runs at all when translation was served from cache. Load
      // the current doc first so we don't kick off a second, concurrent
      // rebuild, then build only if there genuinely isn't one.
      setPipelineStep("Building subtitles");
      setSection("subtitles");
      await loadSubtitles(project.id);
      if (!useAppStore.getState().currentSubtitleDoc) {
        await rebuildSubtitles(project.id);
      }
      throwIfAborted();

      setPipelineStep("Generating voice");
      setSection("voices");
      const voiceReady = await ensureTtsVoiceReady();
      if (!voiceReady) return;
      if (
        await runStage("tts", () =>
          startGenerateTts(project.id, { kind: "missing" }),
        )
      ) {
        ranSomething = true;
      }

      setPipelineStep("Syncing voice");
      if (
        await runStage("sync", () =>
          startApplySync(project.id, { kind: "missing" }),
        )
      ) {
        ranSomething = true;
      }

      setPipelineStep("Mixing audio");
      setSection("mix");
      ranSomething = (await runStage("mix", () => startApplyMix(project.id)))
        ? true
        : ranSomething;

      setPipelineStep("Rendering movie");
      setSection("render");
      ranSomething = (await runStage("render", () =>
        startApplyRender(project.id, {}),
      ))
        ? true
        : ranSomething;

      setPipelineStep(null);
      await useAppStore.getState().refreshMedia(project.id);

      const out = useAppStore.getState().currentMedia?.render;
      if (out?.absolutePath) {
        setNotice(
          ranSomething
            ? `Movie ready: ${out.absolutePath}`
            : `Already up to date — the movie is at ${out.absolutePath}`,
        );
      } else if (opts?.upTo === "render") {
        setNotice(
          "The pipeline finished but no output file was reported. Check the render panel below.",
        );
      }
    } catch (e) {
      if ((e as { code?: string })?.code !== PIPELINE_CANCELLED) {
        setError(formatError(e));
      }
    } finally {
      pipelineAbortRef.current = false;
      setPipelineStep(null);
      setBusy(false);
    }
  };

  /// Stop the chain after the current stage. We also cancel whatever
  /// job is in flight so the user isn't left waiting on a long FFmpeg
  /// pass they already abandoned.
  const handleCancelRunAll = async () => {
    pipelineAbortRef.current = true;
    const inFlight = activeProcessing[0]?.job;
    if (inFlight) {
      try {
        await cancelJob(inFlight.id);
      } catch (e) {
        setError(formatError(e));
      }
    }
  };

  return (
    <div className="app-shell">
      <TopBar
        showBackToDashboard
        subject={
          <div className="topbar-project">
            <span className="pname">{project.name}</span>
            <span className="pmeta">
              {project.sourceLanguage.toUpperCase()} →{" "}
              {project.targetLanguage.toUpperCase()} ·{" "}
              <span className={`status status--${project.status}`}>
                {project.status}
              </span>
            </span>
          </div>
        }
        actions={
          pipelineStep ? (
            <button className="btn danger" onClick={handleCancelRunAll}>
              Stop
            </button>
          ) : (
            <button
              className="btn"
              onClick={() => void handleExport()}
              title="Choose an output location, then finish and render the movie"
            >
              <IconExport size={14} />
              <span>Export</span>
            </button>
          )
        }
        onExport={() => void runPipeline()}
        exportLabel={pipelineStep ?? "Run all"}
        exportBusy={!!pipelineStep}
        exportTitle={
          pipelineStep
            ? `Pipeline running — ${pipelineStep}`
            : "Run every remaining stage through to the final movie"
        }
      />

      <main className="app-body">
        <EditorRail active={section} onChange={setSection} />

        <aside className="sidepane" aria-label="Section browser">
          <div className="sidepane-header">
            <span className="sidepane-title">
              {SECTION_META[section].label}
            </span>
          </div>
          <div className="sidepane-body">
            <SectionBrowser
              section={section}
              project={project}
              media={media ?? null}
              subtitleDoc={subtitleDoc}
              selectedSubtitleId={effectiveSubtitleId}
              onImport={() => void handleImport(false)}
              onImportCopy={() => void handleImport(true)}
              onSelectSubtitle={handleSelectSubtitle}
              workflow={workflow}
            />
          </div>
        </aside>

        <div className="workspace">
          <div className="workspace-center">
            <div className="workspace-split">
              <Stage
                sourcePath={project.sourceMediaPath ?? null}
                onImport={() => void handleImport(false)}
                busy={busy}
              >
                {previewPath ? (
                  <VideoPreview
                    key={previewPath}
                    absolutePath={previewPath}
                    metadata={
                      showingResult ? null : (media?.metadata ?? null)
                    }
                    videoRef={videoRef}
                    onTimeUpdate={setVideoTime}
                    // The render already carries its subtitles, burned in
                    // or otherwise; drawing ours on top would double them.
                    overlayText={
                      showingResult
                        ? null
                        : activeSubtitleFromTime?.translatedText ||
                          activeSubtitleFromTime?.sourceText ||
                          null
                    }
                  />
                ) : null}
                {renderedPath && (
                  <div className="stage-target">
                    <button
                      className={`btn tiny ${showingResult ? "" : "primary"}`}
                      onClick={() => setPreviewTarget("source")}
                      title="The imported file: original audio, no subtitles"
                    >
                      Original
                    </button>
                    <button
                      className={`btn tiny ${showingResult ? "primary" : ""}`}
                      onClick={() => setPreviewTarget("result")}
                      title="The rendered film: Vietnamese dub and subtitles"
                    >
                      Dubbed result
                    </button>
                  </div>
                )}
                <StageToolbar
                  time={videoTime}
                  duration={media?.metadata?.durationSecs ?? null}
                  isPlaying={isPlaying}
                  disabled={!previewPath}
                  onToggle={togglePlayback}
                />
              </Stage>

              <div className="workpane">
                <div className="workpane-header">
                  <span className="workpane-title">
                    {SECTION_META[section].workspaceLabel}
                  </span>
                  {error && (
                    <span
                      className="badge badge--err"
                      role="alert"
                      title={error}
                    >
                      Error
                    </span>
                  )}
                </div>
                <div className="workpane-body">
                  {error && (
                    <div className="banner banner--error msg-row" role="alert">
                      <span>{error}</span>
                      <button
                        className="btn ghost icon"
                        onClick={() => setError(null)}
                        title="Dismiss"
                        aria-label="Dismiss error"
                      >
                        <IconClose size={14} />
                      </button>
                    </div>
                  )}
                  {notice && (
                    <div className="banner banner--info msg-row" role="status">
                      <span>{notice}</span>
                      <button
                        className="btn ghost icon"
                        onClick={() => setNotice(null)}
                        title="Dismiss"
                        aria-label="Dismiss message"
                      >
                        <IconClose size={14} />
                      </button>
                    </div>
                  )}

                  {section === "media" && (
                    <>
                      <Panel title="Source video">
                        <SourcePanel
                          project={project}
                          metadata={media?.metadata ?? null}
                          loading={mediaLoading}
                          busy={busy}
                          onImport={handleImport}
                        />
                      </Panel>
                      <Panel title="Audio for transcription">
                        <AudioPanel
                          hasSource={!!project.sourceMediaPath}
                          ffmpegAvailable={ffmpeg?.available ?? false}
                          ffmpegError={ffmpeg?.error ?? null}
                          audioPath={media?.audioAbsolutePath ?? null}
                          audioSize={media?.audio?.outputSizeBytes ?? null}
                          durationSecs={media?.audio?.durationSecs ?? null}
                          createdAt={media?.audio?.createdAt ?? null}
                          activeJob={activeExtractionJob}
                          progress={
                            activeExtractionJob
                              ? jobProgress[activeExtractionJob.id] ?? 0
                              : 0
                          }
                          onExtract={handleExtract}
                          onCancel={handleCancel}
                          busy={busy}
                        />
                      </Panel>
                    </>
                  )}

                  {section === "transcription" && (
                    <Panel title="Speech recognition">
                      <TranscriptionPanel
                        hasAudio={!!media?.audioAbsolutePath}
                        sttEnv={sttEnv}
                        models={whisperModels}
                        transcript={media?.transcript ?? null}
                        options={sttOptions}
                        onOptionsChange={setSttOptions}
                        activeJob={activeTranscribeJob}
                        progress={
                          activeTranscribeJob
                            ? jobProgress[activeTranscribeJob.id] ?? 0
                            : 0
                        }
                        downloadJob={activeDownloadJob}
                        downloadProgress={
                          activeDownloadJob
                            ? jobProgress[activeDownloadJob.id] ?? 0
                            : 0
                        }
                        busy={busy}
                        onTranscribe={handleTranscribe}
                        onCancel={handleCancelTranscribe}
                        onDownloadModel={handleDownloadModel}
                        onCancelDownload={handleCancelDownload}
                      />
                    </Panel>
                  )}

                  {section === "translation" && (
                    <>
                      <Panel title="Translation">
                        <TranslationPanel
                          hasTranscript={!!media?.transcript}
                          transcriptSummary={media?.transcript ?? null}
                          translationSummary={media?.translation ?? null}
                          env={translationEnv}
                          models={translationModels}
                          recommendedPresets={translationRecommendedPresets}
                          options={translateOptions}
                          onOptionsChange={setTranslateOptions}
                          activeJob={activeTranslateJob}
                          progress={
                            activeTranslateJob
                              ? jobProgress[activeTranslateJob.id] ?? 0
                              : 0
                          }
                          downloadJob={activeTranslateDownloadJob}
                          downloadProgress={
                            activeTranslateDownloadJob
                              ? jobProgress[activeTranslateDownloadJob.id] ?? 0
                              : 0
                          }
                          busy={busy}
                          onTranslate={handleTranslate}
                          onCancel={handleCancelTranslate}
                          onCancelDownload={handleCancelTranslateDownload}
                          onRescanModels={() =>
                            void refreshTranslationModels()
                          }
                        />
                      </Panel>
                      {(translationDoc ||
                        (media?.translation &&
                          media.translation.translatedCount > 0)) && (
                        <Panel title="Translation editor">
                          <TranslationEditor
                            doc={translationDoc}
                            onUpdate={handleUpdateSegment}
                            disabled={!!activeTranslateJob}
                          />
                        </Panel>
                      )}
                    </>
                  )}

                  {section === "subtitles" && (
                    <Panel title="Subtitles">
                      <SubtitlePanel
                        summary={media?.subtitles ?? null}
                        doc={subtitleDoc}
                        loading={subtitleLoading}
                        hasTranscript={!!media?.transcript}
                        currentTime={videoTime}
                        onSeek={(t) => {
                          seekVideo(t);
                          const sid = findActiveSubtitleId(subtitleDoc, t);
                          if (sid != null) setSelectedSubtitleId(sid);
                        }}
                        onRebuild={handleRebuildSubtitles}
                        onPatch={handleSubtitlePatch}
                        onAdd={handleSubtitleAdd}
                        onDelete={handleSubtitleDelete}
                        onSplit={handleSubtitleSplit}
                        onMerge={handleSubtitleMerge}
                        onImport={handleSubtitleImport}
                        onExport={handleSubtitleExport}
                        hasVideo={!!project.sourceMediaPath}
                        voices={ttsVoices}
                        ttsManifest={ttsManifest}
                        ttsEngine={ttsEngine}
                        ttsVoiceId={ttsVoiceId}
                        ttsSettings={ttsSettings}
                        onTtsPreview={handleTtsPreview}
                        onTtsRegenerate={handleTtsRegenerate}
                        syncManifest={syncManifest}
                        syncSettings={syncSettings}
                        onSyncPreview={handleSyncPreview}
                        onSyncRegenerate={handleSyncRegenerate}
                      />
                    </Panel>
                  )}

                  {section === "voices" && (
                    <>
                      <Panel title="TTS / Dubbing">
                        <TtsPanel
                          env={ttsEnv}
                          voices={ttsVoices}
                          recommendedPresets={ttsRecommendedVoices}
                          targetLanguage={project?.targetLanguage ?? "vi"}
                          manifest={ttsManifest}
                          summary={media?.tts ?? null}
                          subtitleDoc={subtitleDoc}
                          engine={ttsEngine}
                          qualityMode={ttsQualityMode}
                          voiceId={ttsVoiceId}
                          settings={ttsSettings}
                          onEngineChange={setTtsEngine}
                          onQualityModeChange={setTtsQualityMode}
                          onVoiceChange={setTtsVoiceId}
                          onSettingsChange={setTtsSettings}
                          activeJob={activeTtsJob}
                          progress={
                            activeTtsJob
                              ? jobProgress[activeTtsJob.id] ?? 0
                              : 0
                          }
                          progressDetail={
                            activeTtsJob
                              ? ttsProgressDetailByJob[activeTtsJob.id] ?? null
                              : null
                          }
                          downloadJob={activeTtsDownloadJob}
                          downloadProgress={
                            activeTtsDownloadJob
                              ? jobProgress[activeTtsDownloadJob.id] ?? 0
                              : 0
                          }
                          busy={busy}
                          onGenerateMissing={handleTtsGenerateMissing}
                          onGenerateAll={handleTtsGenerateAll}
                          onCancel={handleTtsCancel}
                          onCancelDownload={handleCancelTtsDownload}
                          onRescanVoices={() =>
                            void Promise.all([
                              refreshTtsEnv(),
                              refreshTtsVoices(),
                            ])
                          }
                          onInstallF5={() =>
                            downloadTtsVoice("f5-vietnamese-vivoice")
                          }
                          onCreateVoiceProfile={createTtsVoiceProfile}
                          onAssignSpeakerVoice={(speaker, voiceId) =>
                            void assignSubtitleVoiceToSpeaker(
                              project.id,
                              speaker,
                              voiceId,
                            )
                          }
                          lastPreview={lastTtsPreview}
                        />
                      </Panel>
                      <Panel title="Voice sync">
                        <SyncPanel
                          env={syncEnv}
                          summary={media?.sync ?? null}
                          subtitleDoc={subtitleDoc}
                          ttsSummary={media?.tts ?? null}
                          settings={syncSettings}
                          onSettingsChange={setSyncSettings}
                          activeJob={activeSyncJob}
                          progress={
                            activeSyncJob
                              ? jobProgress[activeSyncJob.id] ?? 0
                              : 0
                          }
                          busy={busy}
                          onApplyMissing={handleSyncApplyMissing}
                          onApplyAll={handleSyncApplyAll}
                          onCancel={handleSyncCancel}
                          lastPreview={lastSyncPreview}
                        />
                      </Panel>
                    </>
                  )}

                  {section === "mix" && (
                    <Panel title="Audio mix">
                      <MixPanel
                        env={mixEnv}
                        summary={media?.mix ?? null}
                        syncSummary={media?.sync ?? null}
                        settings={mixSettings}
                        onSettingsChange={setMixSettings}
                        activeJob={activeMixJob}
                        progress={
                          activeMixJob
                            ? jobProgress[activeMixJob.id] ?? 0
                            : 0
                        }
                        busy={busy}
                        onApply={handleMixApply}
                        onCancel={handleMixCancel}
                        onPlay={handleMixPlay}
                        lastPreview={lastMixPreview}
                      />
                    </Panel>
                  )}

                  {section === "render" && (
                    <div id="panel-render">
                      <Panel title="Final render">
                        <RenderPanel
                          env={renderEnv}
                          summary={media?.render ?? null}
                          mixSummary={media?.mix ?? null}
                          subtitleSummary={media?.subtitles ?? null}
                          settings={renderSettings}
                          onSettingsChange={setRenderSettings}
                          activeJob={activeRenderJob}
                          progress={
                            activeRenderJob
                              ? jobProgress[activeRenderJob.id] ?? 0
                              : 0
                          }
                          busy={busy}
                          onApply={() => handleRenderApply(false)}
                          onRegenerate={() => handleRenderApply(true)}
                          onCancel={handleRenderCancel}
                          onPickOutputPath={handleRenderPickOutput}
                          onClearOutputPath={handleRenderClearOutput}
                        />
                      </Panel>
                      <Panel title="Publish">
                        <YouTubePanel
                          projectId={project.id}
                          projectName={project.name}
                          sourceLanguage={project.sourceLanguage}
                          targetLanguage={project.targetLanguage}
                          render={media?.render ?? null}
                          subtitles={media?.subtitles ?? null}
                        />
                      </Panel>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <Timeline
            durationSecs={media?.metadata?.durationSecs ?? null}
            currentTime={videoTime}
            onSeek={seekVideo}
            subtitleDoc={subtitleDoc}
            hasAudio={!!media?.audioAbsolutePath}
            hasVoice={(media?.tts?.generatedCount ?? 0) > 0}
            selectedSubtitleId={effectiveSubtitleId}
            onSelectSubtitle={handleSelectSubtitle}
          />
        </div>

        <aside className="inspector" aria-label="Inspector">
          <Inspector
            project={project}
            media={media ?? null}
            workflow={workflow}
            subtitleDoc={subtitleDoc}
            selectedSubtitleId={effectiveSubtitleId}
            currentTime={videoTime}
            processing={activeProcessing}
            jobProgress={jobProgress}
            onCancelJob={(jobId) => void cancelJob(jobId)}
            onJumpToSection={setSection}
            pipelineStep={pipelineStep}
          />
        </aside>
      </main>
    </div>
  );
}

// =========================================================================
// UI REDESIGN — SHELL SUB-COMPONENTS
// =========================================================================
//
// These are the pure presentation pieces of the new editor layout.
// They only render props — no store subscriptions, no IPC — so they
// can be swapped/restyled without touching the pipeline handlers
// above. Everything sub-panel-related (SourcePanel, TranscriptionPanel,
// SubtitlePanel, ...) stays exactly as it was.
// =========================================================================

type EditorSection =
  | "media"
  | "transcription"
  | "translation"
  | "subtitles"
  | "voices"
  | "mix"
  | "render";

const SECTION_META: Record<
  EditorSection,
  { label: string; workspaceLabel: string; icon: React.ReactNode; hint: string }
> = {
  media:        { label: "Media",         workspaceLabel: "Media & Audio",       icon: <IconMedia size={18} />,     hint: "Source video and audio" },
  transcription:{ label: "Transcribe",    workspaceLabel: "Speech Recognition",  icon: <IconWaveform size={18} />,  hint: "Whisper → transcript" },
  translation:  { label: "Translate",     workspaceLabel: "Translation",         icon: <IconSparkles size={18} />,  hint: "LLM translation + editor" },
  subtitles:    { label: "Subtitles",     workspaceLabel: "Subtitle Editor",     icon: <IconSubtitles size={18} />, hint: "Edit lines and timing" },
  voices:       { label: "Voices",        workspaceLabel: "Dubbing & Sync",      icon: <IconMic size={18} />,       hint: "TTS + voice sync" },
  mix:          { label: "Mix",           workspaceLabel: "Audio Mix",           icon: <IconLayers size={18} />,    hint: "Duck original, add voice" },
  render:       { label: "Render",        workspaceLabel: "Final Render",        icon: <IconExport size={18} />,    hint: "Export the finished movie" },
};

function EditorRail(props: {
  active: EditorSection;
  onChange: (s: EditorSection) => void;
}) {
  const order: EditorSection[] = [
    "media",
    "transcription",
    "translation",
    "subtitles",
    "voices",
    "mix",
    "render",
  ];
  return (
    <nav className="rail" aria-label="Editor sections">
      {order.map((s) => (
        <button
          key={s}
          className={`rail-item ${props.active === s ? "active" : ""}`}
          onClick={() => props.onChange(s)}
          title={SECTION_META[s].hint}
        >
          <span className="rail-icon">{SECTION_META[s].icon}</span>
          <span>{SECTION_META[s].label}</span>
        </button>
      ))}
      <div className="rail-spacer" />
      <div className="rail-divider" />
      <Link
        to="/settings"
        className="rail-item"
        title="Settings"
        style={{ textDecoration: "none" }}
      >
        <span className="rail-icon">
          <IconSettings size={18} />
        </span>
        <span>Settings</span>
      </Link>
    </nav>
  );
}

// -------------------------------------------------------------------------
// Contextual sidepane — what shows up next to the rail depends on which
// section the user picked. Media = asset browser; Subtitles = compact
// line list; every other section keeps a workflow-focused summary so
// the pane is never blank.
// -------------------------------------------------------------------------

type SectionBrowserProps = {
  section: EditorSection;
  project: Project;
  media: ProjectMediaState | null;
  subtitleDoc: SubtitleDoc | null;
  selectedSubtitleId: number | null;
  onImport: () => void;
  onImportCopy: () => void;
  onSelectSubtitle: (id: number) => void;
  workflow: WorkflowState;
};

function SectionBrowser(p: SectionBrowserProps) {
  const media = p.media;

  if (p.section === "media") {
    return (
      <>
        <div className="sidepane-group-title">Video</div>
        {p.project.sourceMediaPath ? (
          <div className="asset" title={p.project.sourceMediaPath}>
            <div className="asset-icon">
              <IconMedia size={16} />
            </div>
            <div className="asset-main">
              <div className="asset-name">
                {basename(p.project.sourceMediaPath)}
              </div>
              <div className="asset-meta">
                {media?.metadata
                  ? `${media.metadata.width ?? "?"}×${media.metadata.height ?? "?"} · ${formatDuration(media.metadata.durationSecs)}`
                  : "video"}
              </div>
            </div>
          </div>
        ) : (
          <div className="asset-empty">
            No video yet.
            <div className="actions" style={{ marginTop: 8 }}>
              <button className="btn small primary" onClick={p.onImport}>
                Import
              </button>
              <button className="btn small ghost" onClick={p.onImportCopy}>
                Import copy
              </button>
            </div>
          </div>
        )}

        <div className="sidepane-group-title">Audio</div>
        {media?.audioAbsolutePath ? (
          <div className="asset" title={media.audioAbsolutePath}>
            <div className="asset-icon">
              <IconWaveform size={16} />
            </div>
            <div className="asset-main">
              <div className="asset-name">{basename(media.audioAbsolutePath)}</div>
              <div className="asset-meta">
                {media.audio
                  ? `${humanBytes(media.audio.outputSizeBytes)} · ${formatDuration(media.audio.durationSecs)}`
                  : "extracted audio"}
              </div>
            </div>
          </div>
        ) : (
          <div className="asset-empty">
            Extracted audio will appear here after transcription.
          </div>
        )}

        <div className="sidepane-group-title">Subtitles</div>
        {media?.subtitles ? (
          <div className="asset">
            <div className="asset-icon">
              <IconSubtitles size={16} />
            </div>
            <div className="asset-main">
              <div className="asset-name">
                {media.subtitles.segmentCount ?? 0} lines
              </div>
              <div className="asset-meta">
                {media.subtitles.updatedAt
                  ? new Date(media.subtitles.updatedAt).toLocaleString()
                  : "up to date"}
              </div>
            </div>
          </div>
        ) : (
          <div className="asset-empty">No subtitle document yet.</div>
        )}
      </>
    );
  }

  if (p.section === "subtitles" && p.subtitleDoc) {
    return (
      <>
        <div className="sidepane-group-title">
          Lines · {p.subtitleDoc.segments.length}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {p.subtitleDoc.segments.slice(0, 60).map((seg) => (
            <button
              key={seg.id}
              className={`asset${
                p.selectedSubtitleId === seg.id ? " selected" : ""
              }`}
              onClick={() => p.onSelectSubtitle(seg.id)}
              style={{
                textAlign: "left",
                cursor: "pointer",
                background:
                  p.selectedSubtitleId === seg.id
                    ? "var(--bg-selected)"
                    : undefined,
                borderColor:
                  p.selectedSubtitleId === seg.id
                    ? "var(--accent)"
                    : undefined,
              }}
            >
              <div
                className="asset-icon"
                style={{ background: "transparent" }}
              >
                <span
                  className="mono xsmall"
                  style={{ color: "var(--fg-muted)" }}
                >
                  {String(seg.id).padStart(3, "0")}
                </span>
              </div>
              <div className="asset-main">
                <div className="asset-name">
                  {seg.translatedText || seg.sourceText || "…"}
                </div>
                <div className="asset-meta">
                  {formatTimecode(seg.start)} → {formatTimecode(seg.end)}
                </div>
              </div>
            </button>
          ))}
          {p.subtitleDoc.segments.length > 60 && (
            <div className="asset-empty">
              …and {p.subtitleDoc.segments.length - 60} more. Use the editor
              to filter.
            </div>
          )}
        </div>
      </>
    );
  }

  return (
    <>
      <div className="sidepane-group-title">Workflow</div>
      {p.workflow.steps.map((step) => (
        <div key={step.key} className={`workflow-step ${step.state}`}>
          <span className="wf-marker" aria-hidden="true">
            {step.state === "done" ? "✓" : step.state === "error" ? "!" : ""}
          </span>
          <span>{step.label}</span>
        </div>
      ))}
    </>
  );
}

// -------------------------------------------------------------------------
// Stage — dark video canvas.
// -------------------------------------------------------------------------

function Stage(props: {
  sourcePath: string | null;
  onImport: () => void;
  busy: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="stage on-canvas">
      <div className="stage-canvas">
        {props.sourcePath ? (
          props.children
        ) : (
          <div className="stage-empty">
            <div style={{ marginBottom: 12 }}>
              No video imported yet — drop a movie in to get started.
            </div>
            <button
              className="btn primary"
              onClick={props.onImport}
              disabled={props.busy}
            >
              <IconFolder size={14} />
              <span>Import video</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function StageToolbar(props: {
  time: number;
  duration: number | null;
  isPlaying: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="stage-toolbar">
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button
          className="icon-btn"
          onClick={props.onToggle}
          disabled={props.disabled}
          title={props.isPlaying ? "Pause" : "Play"}
          aria-label={props.isPlaying ? "Pause" : "Play"}
        >
          {props.isPlaying ? <IconPause size={16} /> : <IconPlay size={16} />}
        </button>
        <span className="tc">
          {formatTimecode(props.time)}
          <span className="tc-sep"> / </span>
          {props.duration != null ? formatTimecode(props.duration) : "--:--"}
        </span>
      </div>
      <div className="subtle xsmall">Preview</div>
    </div>
  );
}

// -------------------------------------------------------------------------
// Timeline — multi-track dark strip. This is a purely visual scrubber:
// clicks seek the video, and the subtitle track surfaces the doc as
// clickable clips. The clips are best-effort — they show the shape of
// the media without touching the underlying editing pipeline.
// -------------------------------------------------------------------------
const EMPTY_SUBTITLE_SEGMENTS: SubtitleSegment[] = [];

function Timeline(props: {
  durationSecs: number | null;
  currentTime: number;
  onSeek: (t: number) => void;
  subtitleDoc: SubtitleDoc | null;
  hasAudio: boolean;
  hasVoice: boolean;
  selectedSubtitleId: number | null;
  onSelectSubtitle: (id: number) => void;
}) {
  const [pxPerSec, setPxPerSec] = useState(24);
  const [viewport, setViewport] = useState({ left: 0, width: 1200 });
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const duration = props.durationSecs ?? 0;
  const totalW = Math.max(duration * pxPerSec, 320);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scroll = e.currentTarget.scrollLeft ?? 0;
    const x = e.clientX - rect.left + scroll;
    props.onSeek(Math.max(0, x / pxPerSec));
  };

  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;
    const updateWidth = () =>
      setViewport((current) => {
        const width = Math.max(1, element.clientWidth);
        return Math.abs(current.width - width) < 1
          ? current
          : { ...current, width };
      });
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const handleScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      const left = event.currentTarget.scrollLeft;
      setViewport((current) =>
        Math.abs(current.left - left) < Math.max(100, current.width / 4)
          ? current
          : { ...current, left },
      );
    },
    [],
  );

  // Keep the playhead in view as it advances.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !duration) return;
    const playX = props.currentTime * pxPerSec;
    if (playX < el.scrollLeft || playX > el.scrollLeft + el.clientWidth - 40) {
      el.scrollLeft = Math.max(0, playX - el.clientWidth / 2);
    }
  }, [props.currentTime, pxPerSec, duration]);

  const ticks = useMemo(() => {
    const out: { at: number; major: boolean; label: string }[] = [];
    if (!duration) return out;
    const step = pickTickStep(pxPerSec);
    const overscanSecs = viewport.width / pxPerSec;
    const visibleStart = Math.max(
      0,
      (viewport.left - viewport.width) / pxPerSec,
    );
    const visibleEnd = Math.min(
      duration,
      (viewport.left + viewport.width * 2) / pxPerSec,
    );
    const firstTick = Math.floor(visibleStart / step) * step;
    for (let t = firstTick; t <= visibleEnd + overscanSecs * 0.01; t += step) {
      const major = Math.round(t) % (step * 5) === 0;
      out.push({ at: t, major, label: formatTimecodeShort(t) });
    }
    return out;
  }, [pxPerSec, duration, viewport]);

  const segments = props.subtitleDoc?.segments ?? EMPTY_SUBTITLE_SEGMENTS;
  const visibleSegments = useMemo(() => {
    const visibleStart = Math.max(
      0,
      (viewport.left - viewport.width) / pxPerSec,
    );
    const visibleEnd =
      (viewport.left + viewport.width * 2) / Math.max(1, pxPerSec);
    return segments.filter(
      (segment) =>
        segment.end >= visibleStart && segment.start <= visibleEnd,
    );
  }, [pxPerSec, segments, viewport]);
  const voiceClips = useMemo(() => {
    if (!props.hasVoice || !duration) return null;
    return visibleSegments.map((segment) => (
      <div
        key={segment.id}
        className={`track-clip voice${
          props.selectedSubtitleId === segment.id ? " selected" : ""
        }`}
        style={{
          left: segment.start * pxPerSec,
          width: Math.max(2, (segment.end - segment.start) * pxPerSec),
        }}
        onClick={(event) => {
          event.stopPropagation();
          props.onSelectSubtitle(segment.id);
        }}
        title={segment.translatedText || segment.sourceText || ""}
      >
        {segment.translatedText || segment.sourceText || "voice"}
      </div>
    ));
  }, [
    duration,
    props.hasVoice,
    props.onSelectSubtitle,
    props.selectedSubtitleId,
    pxPerSec,
    visibleSegments,
  ]);
  const subtitleClips = useMemo(
    () =>
      visibleSegments.map((segment) => (
        <div
          key={segment.id}
          className={`track-clip subtitle${
            props.selectedSubtitleId === segment.id ? " selected" : ""
          }`}
          style={{
            left: segment.start * pxPerSec,
            width: Math.max(2, (segment.end - segment.start) * pxPerSec),
          }}
          onClick={(event) => {
            event.stopPropagation();
            props.onSelectSubtitle(segment.id);
          }}
          title={segment.translatedText || segment.sourceText || ""}
        >
          {segment.translatedText || segment.sourceText || "…"}
        </div>
      )),
    [
      props.onSelectSubtitle,
      props.selectedSubtitleId,
      pxPerSec,
      visibleSegments,
    ],
  );

  return (
    <div className="timeline on-canvas" aria-label="Timeline">
      <div className="timeline-header">
        <div>
          {duration
            ? `${formatTimecode(props.currentTime)} · ${formatTimecode(duration)}`
            : "No media loaded"}
        </div>
        <div className="zoom-controls">
          <button
            onClick={() => setPxPerSec((z) => Math.max(4, z / 1.4))}
            title="Zoom out"
            aria-label="Zoom out"
          >
            <IconZoomOut size={12} />
          </button>
          <button
            onClick={() => setPxPerSec((z) => Math.min(400, z * 1.4))}
            title="Zoom in"
            aria-label="Zoom in"
          >
            <IconZoomIn size={12} />
          </button>
        </div>
      </div>
      <div className="timeline-body">
        <div className="tracks-labels">
          <div className="track-label">
            <span className="tag-name">V1</span> Video
          </div>
          <div className="track-label">
            <span className="tag-name">A1</span> Audio
          </div>
          <div className="track-label">
            <span className="tag-name">A2</span> Voice
          </div>
          <div className="track-label">
            <span className="tag-name">S1</span> Subtitle
          </div>
        </div>
        <div
          className="tracks-canvas"
          ref={canvasRef}
          onClick={handleClick}
          onScroll={handleScroll}
        >
          <div style={{ width: totalW, position: "relative" }}>
            <div className="ruler" style={{ width: totalW }}>
              {ticks.map((t, i) => (
                <div
                  key={i}
                  className={`ruler-tick${t.major ? " major" : ""}`}
                  style={{ left: t.at * pxPerSec }}
                >
                  {t.major ? t.label : ""}
                </div>
              ))}
            </div>
            <div className="track-lanes">
              <div className="track-lane">
                {duration ? (
                  <div
                    className="track-clip video"
                    style={{ left: 0, width: duration * pxPerSec }}
                    title="Video"
                  >
                    Video track
                  </div>
                ) : null}
              </div>
              <div className="track-lane">
                {props.hasAudio && duration ? (
                  <div
                    className="track-clip audio"
                    style={{ left: 0, width: duration * pxPerSec }}
                    title="Original audio"
                  >
                    Original audio
                  </div>
                ) : null}
              </div>
              <div className="track-lane">
                {voiceClips}
              </div>
              <div className="track-lane">{subtitleClips}</div>
            </div>
            <div
              className="playhead"
              style={{ left: props.currentTime * pxPerSec }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function pickTickStep(pxPerSec: number): number {
  const target = 90;
  const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  for (const step of candidates) if (step * pxPerSec >= target) return step;
  return 1200;
}

function formatTimecodeShort(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    return `${h}:${String(m % 60).padStart(2, "0")}`;
  }
  return `${m}:${String(s).padStart(2, "0")}`;
}

// -------------------------------------------------------------------------
// Right inspector — contextual info + AI workflow + live processing.
// -------------------------------------------------------------------------

type WorkflowState = {
  steps: {
    key: string;
    label: string;
    state: "waiting" | "active" | "done" | "error";
  }[];
};

function computeWorkflow(input: {
  hasSource: boolean;
  hasAudio: boolean;
  hasTranscript: boolean;
  translationRatio: number;
  ttsRatio: number;
  syncRatio: number;
  mixReady: boolean;
  renderReady: boolean;
  active: {
    extract: boolean;
    transcribe: boolean;
    translate: boolean;
    tts: boolean;
    sync: boolean;
    mix: boolean;
    render: boolean;
  };
}): WorkflowState {
  const mk = (
    key: string,
    label: string,
    isDone: boolean,
    isActive: boolean,
  ) => ({
    key,
    label,
    state: (isActive
      ? "active"
      : isDone
        ? "done"
        : "waiting") as WorkflowState["steps"][number]["state"],
  });
  return {
    steps: [
      mk("import",     "Import video",      input.hasSource,           false),
      mk("extract",    "Extract audio",     input.hasAudio,            input.active.extract),
      mk("transcribe", "Transcribe",        input.hasTranscript,       input.active.transcribe),
      mk("translate",  "Translate",         input.translationRatio >= .999, input.active.translate),
      mk("tts",        "Generate voice",    input.ttsRatio >= .999,    input.active.tts),
      mk("sync",       "Sync voice",        input.syncRatio >= .999,   input.active.sync),
      mk("mix",        "Mix audio",         input.mixReady,            input.active.mix),
      mk("render",     "Render movie",      input.renderReady,         input.active.render),
    ],
  };
}

function Inspector(props: {
  project: Project;
  media: ProjectMediaState | null;
  workflow: WorkflowState;
  subtitleDoc: SubtitleDoc | null;
  selectedSubtitleId: number | null;
  currentTime: number;
  processing: { label: string; job: JobSnapshot }[];
  jobProgress: Record<string, number>;
  onCancelJob: (jobId: string) => void;
  onJumpToSection: (s: EditorSection) => void;
  pipelineStep: string | null;
}) {
  const seg = useMemo(
    () =>
      props.selectedSubtitleId != null
        ? (props.subtitleDoc?.segments.find(
            (s) => s.id === props.selectedSubtitleId,
          ) ?? null)
        : null,
    [props.subtitleDoc, props.selectedSubtitleId],
  );

  return (
    <>
      <div className="inspector-header">
        <span className="inspector-title">Inspector</span>
      </div>
      <div className="inspector-body">
        {seg ? (
          <div className="section">
            <div className="section-title">Selected subtitle</div>
            <div className="kv-row">
              <span>ID</span>
              <span className="mono">
                {String(seg.id).padStart(3, "0")}
              </span>
            </div>
            <div className="kv-row">
              <span>Start</span>
              <span className="mono">{formatTimecode(seg.start)}</span>
            </div>
            <div className="kv-row">
              <span>End</span>
              <span className="mono">{formatTimecode(seg.end)}</span>
            </div>
            <div className="kv-row">
              <span>Duration</span>
              <span className="mono">
                {(seg.end - seg.start).toFixed(2)}s
              </span>
            </div>
            <div className="field" style={{ marginTop: 8 }}>
              <span>Original</span>
              <div className="translation-row-source">
                {seg.sourceText || "—"}
              </div>
            </div>
            <div className="field">
              <span>Translation</span>
              <div className="translation-row-source">
                {seg.translatedText || "—"}
              </div>
            </div>
            <div className="field">
              <span>Dubbing</span>
              <div className="translation-row-source">
                {seg.dubbingText || seg.translatedText || "—"}
              </div>
            </div>
            <button
              className="btn small"
              onClick={() => props.onJumpToSection("subtitles")}
            >
              Open in editor
            </button>
          </div>
        ) : (
          <div className="section">
            <div className="section-title">Project</div>
            <div className="kv-row">
              <span>Name</span>
              <span>{props.project.name}</span>
            </div>
            <div className="kv-row">
              <span>Languages</span>
              <span className="mono">
                {props.project.sourceLanguage.toUpperCase()} →{" "}
                {props.project.targetLanguage.toUpperCase()}
              </span>
            </div>
            {props.media?.metadata ? (
              <>
                <div className="kv-row">
                  <span>Resolution</span>
                  <span className="mono">
                    {props.media.metadata.width}×
                    {props.media.metadata.height}
                  </span>
                </div>
                <div className="kv-row">
                  <span>Frame rate</span>
                  <span className="mono">
                    {formatFps(props.media.metadata.fps)}
                  </span>
                </div>
                <div className="kv-row">
                  <span>Duration</span>
                  <span className="mono">
                    {formatDuration(props.media.metadata.durationSecs)}
                  </span>
                </div>
                <div className="kv-row">
                  <span>Format</span>
                  <span className="mono">
                    {props.media.metadata.format ?? "—"}
                  </span>
                </div>
              </>
            ) : (
              <div className="muted small">
                Import a video to see its metadata here.
              </div>
            )}
            <div className="kv-row">
              <span>Playhead</span>
              <span className="mono">
                {formatTimecode(props.currentTime)}
              </span>
            </div>
          </div>
        )}

        <div className="section">
          <div className="section-title">AI workflow</div>
          {props.pipelineStep && (
            <div className="banner banner--info" style={{ marginBottom: 8 }}>
              Running the full pipeline — {props.pipelineStep.toLowerCase()}.
              You can leave this window open; each stage starts
              automatically.
            </div>
          )}
          {props.workflow.steps.map((step) => (
            <div key={step.key} className={`workflow-step ${step.state}`}>
              <span className="wf-marker" aria-hidden="true">
                {step.state === "done" ? "✓" : step.state === "error" ? "!" : ""}
              </span>
              <span>{step.label}</span>
            </div>
          ))}
        </div>

        {props.processing.length > 0 && (
          <div className="section">
            <div className="section-title">Processing</div>
            {props.processing.map(({ label, job }) => {
              const pct = Math.round(
                (props.jobProgress[job.id] ?? 0) * 100,
              );
              return (
                <div
                  key={job.id}
                  style={{ display: "flex", flexDirection: "column", gap: 4 }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span>{label}</span>
                    <button
                      className="btn tiny ghost danger"
                      onClick={() => props.onCancelJob(job.id)}
                      title="Cancel"
                    >
                      Cancel
                    </button>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="xsmall muted">{pct}%</div>
                </div>
              );
            })}
          </div>
        )}

        <div className="section">
          <div className="section-title">Storage</div>
          <div className="mono xsmall" style={{ wordBreak: "break-all" }}>
            {props.project.rootPath}
          </div>
        </div>
      </div>
    </>
  );
}

function basename(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function formatFps(fps: number | null | undefined): string {
  if (fps == null) return "—";
  return `${fps.toFixed(fps > 30 ? 0 : 2)} fps`;
}

// ---------------- Source panel ----------------

function SourcePanel(props: {
  project: Project;
  metadata: VideoMetadata | null;
  loading: boolean;
  busy: boolean;
  onImport: (copy: boolean) => void;
}) {
  const { project, metadata, loading, busy, onImport } = props;
  if (project.sourceMediaPath) {
    return (
      <div className="source-panel">
        <div className="kv-row"><span>Path</span><code className="mono small">{project.sourceMediaPath}</code></div>
        <div className="kv-row"><span>Import mode</span><span>{project.sourceImportMode ?? "unknown"}</span></div>
        {project.sourceSize != null && (
          <div className="kv-row"><span>Size</span><span>{humanBytes(project.sourceSize)}</span></div>
        )}
        {project.sourceHash && (
          <div className="kv-row"><span>Hash</span><code className="mono small">{project.sourceHash.slice(0, 24)}…</code></div>
        )}
        <hr />
        {loading && <div className="loading small">Reading metadata…</div>}
        {metadata && <MetadataGrid metadata={metadata} />}
        <div className="actions">
          <button className="btn" onClick={() => onImport(false)} disabled={busy}>
            Replace (reference)
          </button>
          <button className="btn ghost" onClick={() => onImport(true)} disabled={busy}>
            Replace (copy into project)
          </button>
        </div>
      </div>
    );
  }
  return (
    <div className="empty-state">
      <p>No video imported yet.</p>
      <p className="small">
        <strong>Reference</strong> keeps the movie where it is — recommended for
        multi-GB files. <strong>Copy</strong> duplicates it into the project
        folder so the project is self-contained.
      </p>
      <div className="actions">
        <button className="btn primary" onClick={() => onImport(false)} disabled={busy}>
          Import (reference)
        </button>
        <button className="btn" onClick={() => onImport(true)} disabled={busy}>
          Import (copy into project)
        </button>
      </div>
    </div>
  );
}

function MetadataGrid({ metadata }: { metadata: VideoMetadata }) {
  return (
    <div className="metadata-grid">
      <div><span>Duration</span><b>{formatDuration(metadata.durationSecs)}</b></div>
      <div>
        <span>Resolution</span>
        <b>{metadata.width ?? "?"} × {metadata.height ?? "?"}</b>
      </div>
      <div><span>FPS</span><b>{metadata.fps ?? "?"}</b></div>
      <div><span>Video</span><b>{metadata.videoCodec ?? "—"}</b></div>
      <div><span>Audio</span><b>{metadata.audioCodec ?? "—"}</b></div>
      <div>
        <span>Audio channels</span>
        <b>{metadata.audioChannels ?? "—"} @ {metadata.audioSampleRate ?? "—"} Hz</b>
      </div>
      <div><span>Container</span><b>{metadata.format ?? "?"}</b></div>
      <div><span>File size</span><b>{humanBytes(metadata.fileSize)}</b></div>
      <div><span>Audio streams</span><b>{metadata.audioStreamCount}</b></div>
      <div><span>Subtitle streams</span><b>{metadata.subtitleStreamCount}</b></div>
    </div>
  );
}

// ---------------- Video preview ----------------

/** Containers WebKit (Safari / Tauri on macOS) will *not* render.
 *  The transcribe/translate/render pipeline still works on these —
 *  it just means the in-app player above is blank. */
const UNSUPPORTED_PREVIEW_EXTENSIONS = new Set([
  "mkv",
  "avi",
  "flv",
  "wmv",
  "ts",
  "mts",
  "m2ts",
  "mpg",
  "mpeg",
  "vob",
]);

/** Video codecs WebKit typically can't decode even inside a
 *  supported container (`.mp4`). Detected from ffprobe. */
const UNSUPPORTED_PREVIEW_CODECS = new Set(["av1", "vp9"]);

function detectPreviewCompatibility(
  absolutePath: string,
  metadata: VideoMetadata | null,
): { supported: true } | { supported: false; reason: string; hint: string } {
  const dot = absolutePath.lastIndexOf(".");
  const ext =
    dot >= 0 ? absolutePath.slice(dot + 1).toLowerCase() : "";
  if (UNSUPPORTED_PREVIEW_EXTENSIONS.has(ext)) {
    return {
      supported: false,
      reason: `The .${ext} container is not supported by the built-in web preview.`,
      hint: `Convert to .mp4 with:  ffmpeg -i "input.${ext}" -c copy "output.mp4"  (or add -c:a aac if the audio codec is not MP4-compatible). The rest of the pipeline — transcribe, translate, TTS, sync, mix, render — works on the original file regardless.`,
    };
  }
  const codec = (metadata?.videoCodec ?? "").toLowerCase();
  if (codec && UNSUPPORTED_PREVIEW_CODECS.has(codec)) {
    return {
      supported: false,
      reason: `The ${codec.toUpperCase()} video codec is not supported by the built-in web preview.`,
      hint: `Re-encode to H.264:  ffmpeg -i "input" -c:v libx264 -crf 20 -c:a aac -b:a 192k "output.mp4"  and re-import. Pipeline stages still work on the original file.`,
    };
  }
  return { supported: true };
}

function VideoPreview(props: {
  absolutePath: string;
  metadata: VideoMetadata | null;
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
  onTimeUpdate: (t: number) => void;
  overlayText: string | null;
}) {
  const { absolutePath, metadata, videoRef, onTimeUpdate, overlayText } =
    props;
  const src = mediaUrl(absolutePath);
  const compat = detectPreviewCompatibility(absolutePath, metadata);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [runtimeDetail, setRuntimeDetail] = useState<string | null>(null);

  const diagnoseMediaTransport = async () => {
    if (!src) return;
    try {
      const response = await fetch(src, {
        cache: "no-store",
        headers: { Range: "bytes=0-1" },
      });
      const range = response.headers.get("content-range");
      const type = response.headers.get("content-type");
      setRuntimeDetail(
        response.ok
          ? `Local media server is reachable (${response.status}, ${type ?? "unknown type"}${range ? `, ${range}` : ""}). The remaining failure is in WebKit's decoder.`
          : `Local media server returned HTTP ${response.status}.`,
      );
    } catch (err) {
      setRuntimeDetail(
        `WebKit could not reach the local media server: ${err instanceof Error ? err.message : String(err)}.`,
      );
    }
  };

  // The media server hands out its URL asynchronously at startup, so on
  // the very first render there is nothing to point at yet.
  if (!src) {
    return (
      <div className="video-wrap">
        <div className="video-unsupported">
          <strong>Preview starting…</strong>
          <p className="small">Connecting to the local media server.</p>
        </div>
      </div>
    );
  }

  // If we already know the file can't render, don't even mount the
  // <video> — WebKit shows a stubborn play-button poster otherwise.
  if (!compat.supported) {
    return (
      <div className="video-wrap">
        <div className="video-unsupported">
          <strong>Preview unavailable</strong>
          <p>{compat.reason}</p>
          <p className="small">{compat.hint}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="video-wrap">
      <video
        ref={videoRef}
        className="video-preview"
        preload="metadata"
        src={src}
        onLoadedMetadata={() => {
          setRuntimeError(null);
          setRuntimeDetail(null);
        }}
        onTimeUpdate={(e) =>
          onTimeUpdate((e.target as HTMLVideoElement).currentTime)
        }
        onError={(e) => {
          const el = e.target as HTMLVideoElement;
          const code = el.error?.code;
          const codeText =
            code === 1
              ? "load aborted"
              : code === 2
                ? "network error"
                : code === 3
                  ? "decode error — codec not supported by webview"
                  : code === 4
                    ? "source not supported — container or codec not supported by webview"
                    : "unknown";
          setRuntimeError(codeText);
          void diagnoseMediaTransport();
        }}
      >
        Your webview does not support the video element.
      </video>
      {runtimeError && (
        <div className="video-unsupported inline">
          <strong>Preview error:</strong> {runtimeError}
          <div className="small">
            {metadata?.videoCodec
              ? `The file reports ${metadata.videoCodec}${
                  metadata.audioCodec ? ` + ${metadata.audioCodec}` : ""
                }. `
              : ""}
            H.264 video with AAC audio in .mp4 is the combination the
            webview handles best. Every pipeline stage — transcribe,
            translate, dub, render — works on the original either way.
          </div>
          {runtimeDetail && <div className="small">{runtimeDetail}</div>}
        </div>
      )}
      {overlayText && (
        <div className="stage-overlay">
          {overlayText.split("\n").map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------- Audio panel ----------------

function AudioPanel(props: {
  hasSource: boolean;
  ffmpegAvailable: boolean;
  ffmpegError: string | null;
  audioPath: string | null;
  audioSize: number | null;
  durationSecs: number | null;
  createdAt: string | null;
  activeJob: JobSnapshot | null;
  progress: number;
  busy: boolean;
  onExtract: () => void;
  onCancel: () => void;
}) {
  if (!props.hasSource) {
    return <div className="empty-state small">Import a video first.</div>;
  }
  if (!props.ffmpegAvailable) {
    return (
      <div className="banner banner--warn">
        FFmpeg is unavailable. {props.ffmpegError}
        <div className="small">
          Install FFmpeg (e.g. <code>brew install ffmpeg</code> on macOS) or
          set a custom path in Settings.
        </div>
      </div>
    );
  }
  if (props.activeJob) {
    const pct = Math.round(props.progress * 100);
    return (
      <div>
        <div className="progress-row">
          <div className="progress-label">Extracting audio…</div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="progress-value">{pct}%</div>
        </div>
        <div className="actions">
          <button className="btn danger" onClick={props.onCancel}>Cancel</button>
        </div>
      </div>
    );
  }
  if (props.audioPath) {
    return (
      <div className="audio-panel">
        <div className="kv-row"><span>Cached at</span><code className="mono small">{props.audioPath}</code></div>
        {props.audioSize != null && (
          <div className="kv-row"><span>Size</span><span>{humanBytes(props.audioSize)}</span></div>
        )}
        {props.durationSecs != null && (
          <div className="kv-row"><span>Duration</span><span>{formatDuration(props.durationSecs)}</span></div>
        )}
        {props.createdAt && (
          <div className="kv-row"><span>Extracted</span><span>{new Date(props.createdAt).toLocaleString()}</span></div>
        )}
        <div className="actions">
          <button className="btn" onClick={props.onExtract} disabled={props.busy}>
            Re-extract
          </button>
        </div>
      </div>
    );
  }
  return (
    <div>
      <p className="small">
        Extracts a 16 kHz mono PCM WAV, suitable for Whisper (Phase 3).
      </p>
      <div className="actions">
        <button className="btn primary" onClick={props.onExtract} disabled={props.busy}>
          Extract audio
        </button>
      </div>
    </div>
  );
}

// ---------------- Transcription panel ----------------

function TranscriptionPanel(props: {
  hasAudio: boolean;
  sttEnv: SttEnv | null;
  models: WhisperModelInfo[];
  transcript: TranscriptSummary | null;
  options: SttOptions;
  onOptionsChange: (opts: SttOptions) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  downloadJob: JobSnapshot | null;
  downloadProgress: number;
  busy: boolean;
  onTranscribe: () => void;
  onCancel: () => void;
  onDownloadModel: (name: string) => void;
  onCancelDownload: () => void;
}) {
  const {
    hasAudio, sttEnv, models, transcript, options, onOptionsChange,
    activeJob, progress, downloadJob, downloadProgress, busy,
    onTranscribe, onCancel, onDownloadModel, onCancelDownload,
  } = props;

  if (!hasAudio) {
    return (
      <div className="empty-state small">
        Extract audio first — Whisper needs the 16 kHz mono WAV.
      </div>
    );
  }

  const selectedModel = models.find((m) => m.name === options.model);
  const profile = currentSttProfile(options);
  const largeV3 = sttEnv?.largeV3 ?? null;
  const isLargeV3 =
    options.model === "large-v3" || options.model === "large";
  const hardwareBlocked = isLargeV3 && largeV3 != null && largeV3.canRun === false;
  const stage = activeJob?.status === "running" || activeJob?.status === "queued"
    ? guessStage(progress)
    : null;

  const applyProfile = (id: QualityProfile) => {
    onOptionsChange({
      ...options,
      ...QUALITY_PROFILE_PRESETS[id],
    });
  };

  const modelStatus = (name: string): string => {
    if (name === "large-v3" && largeV3 && largeV3.canRun === false) {
      return "Unavailable";
    }
    const info = models.find((m) => m.name === name);
    if (!info) return "Unknown";
    if (downloadJob && options.model === name) return "Loading";
    return info.installed ? "Installed" : "Not installed";
  };

  return (
    <div className="stt-panel">
      {sttEnv && !sttEnv.whisperInstalled && (
        <div className="banner banner--warn">
          faster-whisper is not installed in the worker environment.
          Add it to <code>python/pyproject.toml</code> optional deps or run
          <code>pip install faster-whisper</code> in the worker venv.
        </div>
      )}

      <div className="field">
        <span>STT model</span>
        <div className="stt-profiles" role="radiogroup" aria-label="STT quality">
          {STT_PROFILE_META.map((item) => {
            const preset = QUALITY_PROFILE_PRESETS[item.id];
            const status = modelStatus(preset.model);
            const selected = profile === item.id;
            return (
              <label
                key={item.id}
                className={`stt-profile${selected ? " stt-profile--active" : ""}`}
              >
                <input
                  type="radio"
                  name="stt-quality"
                  checked={selected}
                  disabled={!!activeJob || busy}
                  onChange={() => applyProfile(item.id)}
                />
                <span>
                  <b>{item.label}</b>
                  <span className="stt-profile-model">
                    {whisperDisplayName(preset.model)}
                  </span>
                  <span className="stt-profile-meta">
                    {item.blurb} · {status}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {isLargeV3 && largeV3?.warning && !hardwareBlocked && (
        <div className="banner banner--warn">
          {largeV3.warning}
        </div>
      )}

      {hardwareBlocked && (
        <div className="banner banner--warn">
          <div>
            {largeV3?.reason ||
              "Whisper large-v3 cannot run with the current hardware configuration."}
          </div>
          <div className="actions" style={{ marginTop: 8 }}>
            <button
              className="btn"
              type="button"
              onClick={() => applyProfile("balanced")}
            >
              Use Balanced ({whisperDisplayName(QUALITY_PROFILE_PRESETS.balanced.model)})
            </button>
          </div>
        </div>
      )}

      {selectedModel && !selectedModel.installed && !downloadJob && !hardwareBlocked && (
        <div className="banner">
          {whisperDisplayName(selectedModel.name)} is not installed.
          It can be downloaded during setup; transcription stays offline afterwards.
          <button
            className="btn ghost small"
            style={{ marginLeft: 8 }}
            onClick={() => onDownloadModel(selectedModel.name)}
            disabled={busy}
          >
            Download {selectedModel.name}
          </button>
        </div>
      )}

      <div className="stt-grid">
        <label className="field">
          <span>Language</span>
          <select
            value={options.language ?? ""}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({
                ...options,
                language: e.target.value === "" ? null : e.target.value,
              })
            }
          >
            {LANGUAGE_OPTIONS.map((l) => (
              <option key={l.code ?? "auto"} value={l.code ?? ""}>{l.label}</option>
            ))}
          </select>
        </label>
      </div>

      <details className="stt-advanced">
        <summary>Advanced</summary>
        <div className="stt-grid">
          <label className="field">
            <span>Exact model</span>
            <select
              value={options.model}
              disabled={!!activeJob || busy}
              onChange={(e) =>
                onOptionsChange({
                  ...options,
                  model: e.target.value,
                  qualityProfile: "custom",
                })
              }
            >
              {models.length === 0 && (
                <option value={options.model}>{options.model}</option>
              )}
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} — {m.paramsM}M params
                  {m.installed ? "" : "  (not installed)"}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Device</span>
            <select
              value={options.device ?? ""}
              disabled={!!activeJob || busy}
              onChange={(e) =>
                onOptionsChange({
                  ...options,
                  device: e.target.value === "" ? null : e.target.value,
                })
              }
            >
              <option value="">Auto ({sttEnv?.defaultDevice ?? "cpu"})</option>
              {(sttEnv?.devices ?? []).map((d) => (
                <option key={d.kind} value={d.kind} disabled={!d.supported}>
                  {d.label}{d.supported ? "" : " (unsupported)"}
                </option>
              ))}
            </select>
          </label>

          <label className="field field--check">
            <input
              type="checkbox"
              checked={options.wordTimestamps}
              disabled={!!activeJob || busy}
              onChange={(e) =>
                onOptionsChange({ ...options, wordTimestamps: e.target.checked })
              }
            />
            <span>Word-level timestamps</span>
          </label>

          <label className="field field--check">
            <input
              type="checkbox"
              checked={options.vadFilter}
              disabled={!!activeJob || busy}
              onChange={(e) =>
                onOptionsChange({ ...options, vadFilter: e.target.checked })
              }
            />
            <span>Voice activity detection</span>
          </label>
        </div>
      </details>

      {downloadJob ? (
        <div className="stt-progress">
          <div className="progress-row">
            <div className="progress-label">Downloading model…</div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.round(downloadProgress * 100)}%` }}
              />
            </div>
            <div className="progress-value">{Math.round(downloadProgress * 100)}%</div>
          </div>
          <div className="actions">
            <button className="btn danger" onClick={onCancelDownload}>Cancel download</button>
          </div>
        </div>
      ) : activeJob ? (
        <div className="stt-progress">
          <div className="progress-row">
            <div className="progress-label">
              {stage ?? "Transcribing…"}
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
            <div className="progress-value">{Math.round(progress * 100)}%</div>
          </div>
          <div className="actions">
            <button className="btn danger" onClick={onCancel}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="actions">
          <button
            className="btn primary"
            disabled={busy || !selectedModel || hardwareBlocked}
            onClick={onTranscribe}
            title={
              hardwareBlocked
                ? "Whisper large-v3 cannot run on this hardware. Switch to Balanced."
                : selectedModel && !selectedModel.installed
                ? `Model ${selectedModel.name} will be downloaded first, then transcription starts automatically.`
                : undefined
            }
          >
            {selectedModel && !selectedModel.installed
              ? `Download & transcribe (${selectedModel.name})`
              : transcript
                ? "Re-transcribe"
                : "Start transcription"}
          </button>
        </div>
      )}

      {transcript && !activeJob && (
        <div className="stt-summary">
          <div className="kv-row">
            <span>Segments</span>
            <b>{transcript.segmentCount} detected</b>
          </div>
          <div className="kv-row">
            <span>Detected language</span><span>{transcript.language}</span>
          </div>
          <div className="kv-row">
            <span>Model</span><span>{transcript.model} / {transcript.device} / {transcript.computeType}</span>
          </div>
          <div className="kv-row">
            <span>Created</span>
            <span>{new Date(transcript.createdAt).toLocaleString()}</span>
          </div>
          <div className="kv-row">
            <span>Path</span>
            <code className="mono small">{transcript.relativePath}</code>
          </div>
        </div>
      )}
    </div>
  );
}

function guessStage(fraction: number): string {
  if (fraction < 0.02) return "Loading model…";
  if (fraction < 0.99) return "Transcribing…";
  return "Finalizing…";
}

// ---------------- Translation panel ----------------

function TranslationPanel(props: {
  hasTranscript: boolean;
  transcriptSummary: TranscriptSummary | null;
  translationSummary: TranslationSummary | null;
  env: TranslationEnv | null;
  models: TranslationModelInfo[];
  recommendedPresets: TranslationRecommendedPreset[];
  options: TranslateOptions;
  onOptionsChange: (opts: TranslateOptions) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  downloadJob: JobSnapshot | null;
  downloadProgress: number;
  busy: boolean;
  onTranslate: () => void;
  onCancel: () => void;
  onCancelDownload: () => void;
  onRescanModels: () => void;
}) {
  const {
    hasTranscript,
    transcriptSummary,
    translationSummary,
    env,
    models,
    recommendedPresets,
    options,
    onOptionsChange,
    activeJob,
    progress,
    downloadJob,
    downloadProgress,
    busy,
    onTranslate,
    onCancel,
    onCancelDownload,
    onRescanModels,
  } = props;

  if (!hasTranscript) {
    return (
      <div className="empty-state small">
        Run speech recognition first — translation needs the transcript.
      </div>
    );
  }

  const modelInstalled =
    !!options.model && models.some((m) => m.name === options.model);
  const totalSegments =
    translationSummary?.segmentCount ?? transcriptSummary?.segmentCount ?? 0;
  const translatedSoFar = translationSummary?.translatedCount ?? 0;
  // Phase 12 UX — the app can auto-download a preset when the
  // selected model isn't present. Trigger the auto-download flow
  // whenever the chosen model isn't installed (not just when the
  // list is empty) so a stale `translateOptions.model` referencing
  // a deleted/renamed GGUF doesn't leave the button disabled.
  const defaultPreset =
    recommendedPresets.find((p) => p.isDefault) ?? recommendedPresets[0] ?? null;
  const canAutoDownload = !!defaultPreset;
  const willAutoDownload =
    !modelInstalled && canAutoDownload && !!env?.llamaInstalled;

  return (
    <div className="stt-panel">
      {env && !env.llamaInstalled && (
        <div className="banner banner--warn">
          llama-cpp-python is not installed in the worker environment.
          Add it to <code>python/pyproject.toml</code> optional deps or
          run <code>pip install llama-cpp-python</code> in the worker venv.
        </div>
      )}
      {env && env.llamaInstalled && models.length === 0 && !canAutoDownload && (
        <div className="banner banner--warn">
          No GGUF models found under
          <code>{env.translationRoot}</code>. Drop a GGUF file (Qwen2,
          Llama 3, Mistral, ...) in that folder and click{" "}
          <em>Rescan models</em>.
        </div>
      )}
      {willAutoDownload && !downloadJob && (
        <div className="banner">
          No translation model installed yet. Clicking <em>Translate</em>{" "}
          will download <b>{defaultPreset!.label}</b> to{" "}
          <code>{env!.translationRoot}</code> and then start
          translation. This is a one-time download.
        </div>
      )}

      <div className="stt-grid">
        <label className="field">
          <span>Source language</span>
          <select
            value={options.sourceLanguage}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({ ...options, sourceLanguage: e.target.value })
            }
          >
            {TRANSLATION_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Target language</span>
          <select
            value={options.targetLanguage}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({ ...options, targetLanguage: e.target.value })
            }
          >
            {TRANSLATION_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Model (GGUF)</span>
          <select
            value={options.model}
            disabled={!!activeJob || busy || models.length === 0}
            onChange={(e) =>
              onOptionsChange({ ...options, model: e.target.value })
            }
          >
            {models.length === 0 && (
              <option value="">— none installed —</option>
            )}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.isDefault ? " (default)" : ""}
                {" — "}
                {humanBytes(m.sizeBytes)}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Chunk size</span>
          <input
            type="number"
            min={5}
            max={200}
            value={options.chunkSize}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({
                ...options,
                chunkSize: Math.max(1, Number(e.target.value) || 30),
              })
            }
          />
        </label>

        <label className="field">
          <span>Context before</span>
          <input
            type="number"
            min={0}
            max={20}
            value={options.contextBefore}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({
                ...options,
                contextBefore: Math.max(0, Number(e.target.value) || 0),
              })
            }
          />
        </label>

        <label className="field">
          <span>Context after</span>
          <input
            type="number"
            min={0}
            max={20}
            value={options.contextAfter}
            disabled={!!activeJob || busy}
            onChange={(e) =>
              onOptionsChange({
                ...options,
                contextAfter: Math.max(0, Number(e.target.value) || 0),
              })
            }
          />
        </label>
      </div>

      {downloadJob ? (
        <div className="stt-progress">
          <div className="progress-row">
            <div className="progress-label">
              Downloading translation model
              {defaultPreset ? ` — ${defaultPreset.label}` : ""}…
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${Math.round(downloadProgress * 100)}%`,
                }}
              />
            </div>
            <div className="progress-value">
              {Math.round(downloadProgress * 100)}%
            </div>
          </div>
          <div className="actions">
            <button className="btn danger" onClick={onCancelDownload}>
              Cancel download
            </button>
          </div>
        </div>
      ) : activeJob ? (
        <div className="stt-progress">
          <div className="progress-row">
            <div className="progress-label">
              Translating…{" "}
              {totalSegments > 0
                ? `${translatedSoFar} / ${totalSegments} segments`
                : ""}
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
            <div className="progress-value">
              {Math.round(progress * 100)}%
            </div>
          </div>
          <div className="actions">
            <button className="btn danger" onClick={onCancel}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="actions">
          <button
            className="btn primary"
            // Enabled either when a model is installed OR we can
            // auto-download the recommended preset — the whole point
            // is to unblock users who have a fresh install.
            disabled={busy || (!modelInstalled && !willAutoDownload)}
            onClick={onTranslate}
            title={
              willAutoDownload && defaultPreset
                ? `Will download ${defaultPreset.label} (~${humanBytes(defaultPreset.approxSizeBytes)}) and then translate.`
                : undefined
            }
          >
            {willAutoDownload && defaultPreset
              ? `Download & translate (${defaultPreset.label})`
              : translationSummary &&
                  translationSummary.translatedCount >=
                    translationSummary.segmentCount
                ? "Re-translate"
                : translationSummary
                  ? `Resume translation (${translationSummary.translatedCount}/${translationSummary.segmentCount})`
                  : "Translate"}
          </button>
          <button className="btn ghost" onClick={onRescanModels}>
            Rescan models
          </button>
        </div>
      )}

      {translationSummary && (
        <div className="stt-summary">
          <div className="kv-row">
            <span>Progress</span>
            <b>
              {translationSummary.translatedCount} /{" "}
              {translationSummary.segmentCount} segments translated
              {translationSummary.editedCount > 0
                ? ` · ${translationSummary.editedCount} edited`
                : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Model</span>
            <span>{translationSummary.model}</span>
          </div>
          <div className="kv-row">
            <span>Prompt</span>
            <code className="mono small">
              {translationSummary.promptVersion}
            </code>
          </div>
          <div className="kv-row">
            <span>Updated</span>
            <span>
              {new Date(translationSummary.updatedAt).toLocaleString()}
            </span>
          </div>
          <div className="kv-row">
            <span>Path</span>
            <code className="mono small">
              {translationSummary.relativePath}
            </code>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------- Translation editor ----------------

// Phase 11 — virtualised list constants (mirror the subtitle list).
// A row is roughly meta (22 px) + source (~30 px) + textarea (~55 px)
// + gap (~10 px). 132 px is a safe upper bound that keeps everything
// visible without overlap; the textarea can still grow interactively.
const TRANSLATION_ROW_HEIGHT = 132;
const TRANSLATION_LIST_HEIGHT = 480;
const TRANSLATION_OVERSCAN = 4;

function TranslationEditor(props: {
  doc: TranslationDoc | null;
  onUpdate: (segmentId: number, value: string) => void;
  disabled: boolean;
}) {
  const { doc, onUpdate, disabled } = props;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);

  if (!doc) {
    return (
      <div className="empty-state small">
        Loading translation…
      </div>
    );
  }
  if (doc.segments.length === 0) {
    return (
      <div className="empty-state small">
        No translated segments yet.
      </div>
    );
  }

  const segments = doc.segments;
  const totalHeight = segments.length * TRANSLATION_ROW_HEIGHT;
  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / TRANSLATION_ROW_HEIGHT) - TRANSLATION_OVERSCAN,
  );
  const visibleCount =
    Math.ceil(TRANSLATION_LIST_HEIGHT / TRANSLATION_ROW_HEIGHT) +
    TRANSLATION_OVERSCAN * 2;
  const endIndex = Math.min(segments.length, startIndex + visibleCount);
  const visible = segments.slice(startIndex, endIndex);

  return (
    <div className="translation-editor">
      <div
        ref={scrollRef}
        className="translation-editor-list virtualised"
        style={{ height: TRANSLATION_LIST_HEIGHT }}
        onScroll={(e) =>
          setScrollTop((e.target as HTMLDivElement).scrollTop)
        }
      >
        <div style={{ height: totalHeight, position: "relative" }}>
          {visible.map((seg, i) => {
            const idx = startIndex + i;
            const top = idx * TRANSLATION_ROW_HEIGHT;
            return (
              <div
                key={seg.id}
                style={{
                  position: "absolute",
                  top,
                  left: 0,
                  right: 6,
                  height: TRANSLATION_ROW_HEIGHT,
                  paddingBottom: 8,
                }}
              >
                <TranslationRow
                  seg={seg}
                  disabled={disabled}
                  onSave={(value) => onUpdate(seg.id, value)}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function TranslationRow(props: {
  seg: {
    id: number;
    sourceText: string;
    translation: string;
    start: number;
    end: number;
    edited: boolean;
  };
  disabled: boolean;
  onSave: (value: string) => void;
}) {
  const { seg, disabled, onSave } = props;
  // Local draft so typing feels snappy — we push to the backend on blur.
  const [draft, setDraft] = useState(seg.translation);
  useEffect(() => {
    setDraft(seg.translation);
  }, [seg.translation]);

  const empty = !seg.translation.trim();
  return (
    <div className={`translation-row${empty ? " translation-row--empty" : ""}`}>
      <div className="translation-row-meta">
        <span className="translation-row-id">#{seg.id}</span>
        <span className="translation-row-time small">
          {formatDuration(seg.start)} → {formatDuration(seg.end)}
        </span>
        {seg.edited && <span className="tag tag--edited">edited</span>}
        {empty && <span className="tag tag--pending">pending</span>}
      </div>
      <div className="translation-row-source">{seg.sourceText}</div>
      <textarea
        className="translation-row-input"
        rows={2}
        value={draft}
        disabled={disabled}
        placeholder={disabled ? "Translating…" : "Not translated yet"}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          if (draft !== seg.translation) {
            onSave(draft);
          }
        }}
      />
    </div>
  );
}

// ---------------- Subtitle panel ----------------

const SUBTITLE_ROW_HEIGHT = 232; // rough px height per row for virtualization
const SUBTITLE_LIST_HEIGHT = 520;
const SUBTITLE_OVERSCAN = 4;

function SubtitlePanel(props: {
  summary: SubtitleSummary | null;
  doc: SubtitleDoc | null;
  loading: boolean;
  hasTranscript: boolean;
  hasVideo: boolean;
  currentTime: number;
  onSeek: (time: number) => void;
  onRebuild: () => void;
  onPatch: (id: number, patch: SubtitleSegmentPatch) => void;
  onAdd: (afterId: number | null, start: number, end: number) => void;
  onDelete: (id: number) => void;
  onSplit: (id: number, time: number) => void;
  onMerge: (id: number) => void;
  onImport: () => void;
  onExport: (format: SubtitleFormat, kind: ExportKind) => void;
  voices: VoiceInfo[];
  ttsManifest: TtsManifest | null;
  ttsEngine: string;
  ttsVoiceId: string;
  ttsSettings: TtsSettings;
  onTtsPreview: (segmentId: number) => void;
  onTtsRegenerate: (segmentId: number) => void;
  syncManifest: SyncManifest | null;
  syncSettings: SyncSettings;
  onSyncPreview: (segmentId: number) => void;
  onSyncRegenerate: (segmentId: number) => void;
}) {
  const {
    summary,
    doc,
    loading,
    hasTranscript,
    hasVideo,
    currentTime,
    onSeek,
    onRebuild,
    onPatch,
    onAdd,
    onDelete,
    onSplit,
    onMerge,
    onImport,
    onExport,
    voices,
    ttsManifest,
    ttsEngine,
    ttsVoiceId,
    ttsSettings,
    onTtsPreview,
    onTtsRegenerate,
    syncManifest,
    syncSettings,
    onSyncPreview,
    onSyncRegenerate,
  } = props;

  const [exportFormat, setExportFormat] = useState<SubtitleFormat>("srt");
  const [exportKind, setExportKind] = useState<ExportKind>("translated");
  const [filter, setFilter] = useState("");
  const activeId = useMemo(
    () => findActiveSubtitleId(doc, currentTime),
    [doc, currentTime],
  );
  const segments = doc?.segments ?? EMPTY_SUBTITLE_SEGMENTS;
  const filtered = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) return segments;
    return segments.filter((segment) =>
      `${segment.sourceText}\n${segment.translatedText}\n${segment.dubbingText ?? ""}`
        .toLowerCase()
        .includes(query),
    );
  }, [filter, segments]);

  if (!doc && !summary) {
    if (!hasTranscript) {
      return (
        <div className="empty-state small">
          Run speech recognition to produce a transcript, then click
          <em> Build subtitles</em> to derive the canonical subtitle model.
        </div>
      );
    }
    return (
      <div className="subtitles-empty">
        <p className="small">
          No subtitle document yet. Build it from the current transcript
          (and translation, if available), or import an existing SRT/ASS
          file.
        </p>
        <div className="actions">
          <button className="btn primary" onClick={onRebuild}>
            Build subtitles
          </button>
          <button className="btn" onClick={onImport}>
            Import SRT/ASS
          </button>
        </div>
      </div>
    );
  }

  if (loading && !doc) {
    return <div className="loading small">Loading subtitles…</div>;
  }

  const dirty = doc?.dirty ??
    summary?.dirty ?? { tts: false, sync: false, mix: false, render: false };
  const anyDirty = dirty.tts || dirty.sync || dirty.mix || dirty.render;

  return (
    <div className="subtitle-panel">
      <div className="subtitle-summary">
        <div className="kv-row">
          <span>Segments</span>
          <b>
            {segments.length}
            {summary?.translatedCount != null && segments.length > 0
              ? ` · ${summary.translatedCount} translated`
              : ""}
            {summary?.speakerCount ? ` · ${summary.speakerCount} speakers` : ""}
            {summary?.overlapCount
              ? ` · ${summary.overlapCount} overlap${
                  summary.overlapCount === 1 ? "" : "s"
                }`
              : ""}
          </b>
        </div>
        <div className="kv-row">
          <span>Origin</span>
          <span>{doc?.derivedFrom.origin ?? summary?.derivedFrom.origin}</span>
        </div>
        {summary?.relativePath && (
          <div className="kv-row">
            <span>Path</span>
            <code className="mono small">{summary.relativePath}</code>
          </div>
        )}
        {anyDirty && (
          <div className="banner banner--warn small">
            Downstream stages need rerun:
            {dirty.tts ? " TTS" : ""}
            {dirty.sync ? `${dirty.tts ? " · " : " "}Sync` : ""}
            {dirty.mix ? " · Mix" : ""}
            {dirty.render ? " · Render" : ""}
          </div>
        )}
      </div>

      <div className="subtitle-toolbar">
        <input
          type="search"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="subtitle-filter"
        />
        <button className="btn ghost small" onClick={onRebuild}>
          Rebuild from transcript
        </button>
        <button
          className="btn ghost small"
          onClick={() => {
            const last = segments[segments.length - 1];
            const start = last ? last.end : currentTime;
            onAdd(last?.id ?? null, start, start + 2);
          }}
        >
          Add row
        </button>
        <button className="btn ghost small" onClick={onImport}>
          Import
        </button>
        <div className="subtitle-export">
          <select
            className="small"
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as SubtitleFormat)}
            aria-label="Export format"
          >
            <option value="srt">SRT</option>
            <option value="ass">ASS</option>
          </select>
          <select
            className="small"
            value={exportKind}
            onChange={(e) => setExportKind(e.target.value as ExportKind)}
            aria-label="Export content"
          >
            <option value="translated">Translated only</option>
            <option value="source">Source only</option>
            <option value="bilingual">Bilingual</option>
          </select>
          <button
            className="btn small"
            disabled={segments.length === 0}
            onClick={() => onExport(exportFormat, exportKind)}
          >
            Export
          </button>
        </div>
      </div>

      <SubtitleList
        segments={filtered}
        activeId={activeId}
        currentTime={currentTime}
        hasVideo={hasVideo}
        onSeek={onSeek}
        onPatch={onPatch}
        onDelete={onDelete}
        onSplit={onSplit}
        onMerge={onMerge}
        voices={voices}
        ttsManifest={ttsManifest}
        ttsEngine={ttsEngine}
        ttsVoiceId={ttsVoiceId}
        ttsSettings={ttsSettings}
        onTtsPreview={onTtsPreview}
        onTtsRegenerate={onTtsRegenerate}
        syncManifest={syncManifest}
        syncSettings={syncSettings}
        onSyncPreview={onSyncPreview}
        onSyncRegenerate={onSyncRegenerate}
      />
    </div>
  );
}

function SubtitleList(props: {
  segments: SubtitleSegment[];
  activeId: number | null;
  currentTime: number;
  hasVideo: boolean;
  onSeek: (time: number) => void;
  onPatch: (id: number, patch: SubtitleSegmentPatch) => void;
  onDelete: (id: number) => void;
  onSplit: (id: number, time: number) => void;
  onMerge: (id: number) => void;
  voices: VoiceInfo[];
  ttsManifest: TtsManifest | null;
  ttsEngine: string;
  ttsVoiceId: string;
  ttsSettings: TtsSettings;
  onTtsPreview: (segmentId: number) => void;
  onTtsRegenerate: (segmentId: number) => void;
  syncManifest: SyncManifest | null;
  syncSettings: SyncSettings;
  onSyncPreview: (segmentId: number) => void;
  onSyncRegenerate: (segmentId: number) => void;
}) {
  const {
    segments,
    activeId,
    currentTime,
    hasVideo,
    onSeek,
    onPatch,
    onDelete,
    onSplit,
    onMerge,
    voices,
    ttsManifest,
    ttsEngine,
    ttsVoiceId,
    ttsSettings,
    onTtsPreview,
    onTtsRegenerate,
    syncManifest,
    syncSettings,
    onSyncPreview,
    onSyncRegenerate,
  } = props;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const ttsBySegment = useMemo(
    () =>
      new Map(
        (ttsManifest?.segments ?? []).map((entry) => [
          entry.segmentId,
          entry,
        ]),
      ),
    [ttsManifest],
  );
  const syncBySegment = useMemo(
    () =>
      new Map(
        (syncManifest?.segments ?? []).map((entry) => [
          entry.segmentId,
          entry,
        ]),
      ),
    [syncManifest],
  );
  const engineVoices = useMemo(
    () => voices.filter((voice) => voice.engine === ttsEngine),
    [ttsEngine, voices],
  );

  const totalHeight = segments.length * SUBTITLE_ROW_HEIGHT;
  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / SUBTITLE_ROW_HEIGHT) - SUBTITLE_OVERSCAN,
  );
  const visibleCount =
    Math.ceil(SUBTITLE_LIST_HEIGHT / SUBTITLE_ROW_HEIGHT) +
    SUBTITLE_OVERSCAN * 2;
  const endIndex = Math.min(segments.length, startIndex + visibleCount);
  const visible = segments.slice(startIndex, endIndex);

  if (segments.length === 0) {
    return <div className="empty-state small">No subtitle rows.</div>;
  }

  return (
    <div
      ref={scrollRef}
      className="subtitle-list"
      style={{ height: SUBTITLE_LIST_HEIGHT }}
      onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
    >
      <div style={{ height: totalHeight, position: "relative" }}>
        {visible.map((seg, i) => {
          const idx = startIndex + i;
          const top = idx * SUBTITLE_ROW_HEIGHT;
          return (
            <div
              key={seg.id}
              className="subtitle-row-wrap"
              style={{
                position: "absolute",
                top,
                left: 0,
                right: 0,
                height: SUBTITLE_ROW_HEIGHT,
              }}
            >
              <SubtitleRow
                seg={seg}
                active={seg.id === activeId}
                canSplitAt={
                  hasVideo &&
                  currentTime > seg.start + 0.05 &&
                  currentTime < seg.end - 0.05
                }
                currentTime={currentTime}
                hasVideo={hasVideo}
                onSeek={onSeek}
                onPatch={onPatch}
                onDelete={onDelete}
                onSplit={onSplit}
                onMerge={onMerge}
                voices={engineVoices}
                ttsStatus={computeTtsStatus(
                  seg,
                  ttsBySegment.get(seg.id) ?? null,
                  ttsEngine,
                  ttsVoiceId,
                  ttsSettings,
                )}
                onTtsPreview={onTtsPreview}
                onTtsRegenerate={onTtsRegenerate}
                syncStatus={computeSyncStatus(
                  seg,
                  syncBySegment.get(seg.id) ?? null,
                  syncManifest,
                  syncSettings,
                )}
                onSyncPreview={onSyncPreview}
                onSyncRegenerate={onSyncRegenerate}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SubtitleRow(props: {
  seg: SubtitleSegment;
  active: boolean;
  canSplitAt: boolean;
  currentTime: number;
  hasVideo: boolean;
  onSeek: (time: number) => void;
  onPatch: (id: number, patch: SubtitleSegmentPatch) => void;
  onDelete: (id: number) => void;
  onSplit: (id: number, time: number) => void;
  onMerge: (id: number) => void;
  voices: VoiceInfo[];
  ttsStatus: TtsRowStatus;
  onTtsPreview: (segmentId: number) => void;
  onTtsRegenerate: (segmentId: number) => void;
  syncStatus: SyncRowStatus;
  onSyncPreview: (segmentId: number) => void;
  onSyncRegenerate: (segmentId: number) => void;
}) {
  const {
    seg,
    active,
    canSplitAt,
    currentTime,
    hasVideo,
    onSeek,
    onPatch,
    onDelete,
    onSplit,
    onMerge,
    voices,
    ttsStatus,
    onTtsPreview,
    onTtsRegenerate,
    syncStatus,
    onSyncPreview,
    onSyncRegenerate,
  } = props;
  const [source, setSource] = useState(seg.sourceText);
  const [translated, setTranslated] = useState(seg.translatedText);
  const [dubbing, setDubbing] = useState(seg.dubbingText ?? "");
  const [speaker, setSpeaker] = useState(seg.speaker ?? "");
  const [voice, setVoice] = useState(seg.voiceId ?? "");
  const [startStr, setStartStr] = useState(formatTimecode(seg.start));
  const [endStr, setEndStr] = useState(formatTimecode(seg.end));
  useEffect(() => setSource(seg.sourceText), [seg.sourceText]);
  useEffect(() => setTranslated(seg.translatedText), [seg.translatedText]);
  useEffect(() => setDubbing(seg.dubbingText ?? ""), [seg.dubbingText]);
  useEffect(() => setSpeaker(seg.speaker ?? ""), [seg.speaker]);
  useEffect(() => setVoice(seg.voiceId ?? ""), [seg.voiceId]);
  useEffect(() => setStartStr(formatTimecode(seg.start)), [seg.start]);
  useEffect(() => setEndStr(formatTimecode(seg.end)), [seg.end]);

  const commitTiming = () => {
    const start = parseTimecode(startStr);
    const end = parseTimecode(endStr);
    const patch: SubtitleSegmentPatch = {};
    if (start != null && Math.abs(start - seg.start) > 1e-4) patch.start = start;
    if (end != null && Math.abs(end - seg.end) > 1e-4) patch.end = end;
    if (Object.keys(patch).length > 0) {
      onPatch(seg.id, patch);
    } else {
      setStartStr(formatTimecode(seg.start));
      setEndStr(formatTimecode(seg.end));
    }
  };

  return (
    <div className={`subtitle-row${active ? " subtitle-row--active" : ""}`}>
      <div className="subtitle-row-head">
        <button
          className="subtitle-row-id"
          onClick={() => onSeek(seg.start)}
          title="Seek video to this segment"
        >
          #{seg.id}
        </button>
        <input
          className="subtitle-timecode"
          value={startStr}
          onChange={(e) => setStartStr(e.target.value)}
          onBlur={commitTiming}
          aria-label="Start time"
        />
        <span className="subtitle-timecode-sep">→</span>
        <input
          className="subtitle-timecode"
          value={endStr}
          onChange={(e) => setEndStr(e.target.value)}
          onBlur={commitTiming}
          aria-label="End time"
        />
        <span className="subtitle-duration small">
          {(seg.end - seg.start).toFixed(2)}s
        </span>
        <input
          className="subtitle-speaker"
          placeholder="Speaker"
          value={speaker}
          onChange={(e) => setSpeaker(e.target.value)}
          onBlur={() => {
            if ((seg.speaker ?? "") !== speaker) {
              onPatch(seg.id, {
                speaker: speaker.trim() === "" ? null : speaker,
              });
            }
          }}
        />
        {voices.length > 0 ? (
          <select
            className="subtitle-voice"
            value={voice}
            onChange={(e) => {
              const next = e.target.value;
              setVoice(next);
              if ((seg.voiceId ?? "") !== next) {
                onPatch(seg.id, {
                  voiceId: next.trim() === "" ? null : next,
                });
              }
            }}
            aria-label="Voice"
          >
            <option value="">Default voice</option>
            {voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
                {v.gender && v.gender !== "unknown" ? ` · ${v.gender}` : ""}
              </option>
            ))}
          </select>
        ) : (
          <input
            className="subtitle-voice"
            placeholder="Voice"
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            onBlur={() => {
              if ((seg.voiceId ?? "") !== voice) {
                onPatch(seg.id, {
                  voiceId: voice.trim() === "" ? null : voice,
                });
              }
            }}
          />
        )}
      </div>
      <div className="subtitle-row-body">
        <textarea
          className="subtitle-text subtitle-text--source"
          rows={2}
          value={source}
          placeholder="Source text"
          onChange={(e) => setSource(e.target.value)}
          onBlur={() => {
            if (source !== seg.sourceText) {
              onPatch(seg.id, { sourceText: source });
            }
          }}
        />
        <textarea
          className="subtitle-text subtitle-text--translated"
          rows={2}
          value={translated}
          placeholder="Translated text"
          onChange={(e) => setTranslated(e.target.value)}
          onBlur={() => {
            if (translated !== seg.translatedText) {
              onPatch(seg.id, { translatedText: translated });
            }
          }}
        />
        <textarea
          className="subtitle-text subtitle-text--dubbing"
          rows={2}
          value={dubbing}
          placeholder="Dubbing text (spoken line)"
          onChange={(e) => setDubbing(e.target.value)}
          onBlur={() => {
            if (dubbing !== (seg.dubbingText ?? "")) {
              onPatch(seg.id, { dubbingText: dubbing });
            }
          }}
        />
      </div>
      <div className="subtitle-row-actions">
        <span
          className={`tts-badge tts-badge--${ttsStatus.state}`}
          title={ttsStatus.hint}
        >
          {ttsStatus.state === "generated"
            ? `✓ Generated · ${ttsStatus.durationSecs?.toFixed(2) ?? "?"}s`
            : ttsStatus.state === "stale"
              ? "⚠ Outdated"
              : "⚠ Missing"}
        </span>
        <span
          className={`sync-badge sync-badge--${syncStatus.state}`}
          title={syncStatus.hint}
        >
          {syncStatus.label}
        </span>
        <button
          className="btn ghost small"
          onClick={() => onTtsPreview(seg.id)}
          title="Preview voice for this line"
        >
          ▶ Preview voice
        </button>
        <button
          className="btn ghost small"
          onClick={() => onTtsRegenerate(seg.id)}
          title="Regenerate voice for this line"
        >
          Regenerate voice
        </button>
        <button
          className="btn ghost small"
          disabled={syncStatus.state === "missing" || syncStatus.state === "stale"}
          onClick={() => onSyncPreview(seg.id)}
          title="Preview time-aligned voice for this line"
        >
          ▶ Preview synced
        </button>
        <button
          className="btn ghost small"
          onClick={() => onSyncRegenerate(seg.id)}
          title="Rebuild synced WAV for this line"
        >
          Resync
        </button>
        <button
          className="btn ghost small"
          disabled={!canSplitAt}
          onClick={() => onSplit(seg.id, currentTime)}
          title={
            hasVideo
              ? "Split at the current video playhead"
              : "Video preview required to split"
          }
        >
          Split at playhead
        </button>
        <button
          className="btn ghost small"
          onClick={() => onMerge(seg.id)}
          title="Merge with the next segment"
        >
          Merge next
        </button>
        <button className="btn danger small" onClick={() => onDelete(seg.id)}>
          Delete
        </button>
      </div>
    </div>
  );
}

// ---------------- TTS helpers ----------------

type TtsRowStatus = {
  state: "generated" | "stale" | "missing";
  hint: string;
  durationSecs: number | null;
};

function computeTtsStatus(
  seg: SubtitleSegment,
  entry: TtsSegmentEntry | null,
  engine: string,
  defaultVoiceId: string,
  settings: TtsSettings,
): TtsRowStatus {
  const text = (
    seg.dubbingText ||
    seg.translatedText ||
    seg.sourceText ||
    ""
  ).trim();
  if (!text) {
    return {
      state: "missing",
      hint: "No text to synthesise yet.",
      durationSecs: null,
    };
  }
  const voice = seg.voiceId || defaultVoiceId;
  if (!voice) {
    return {
      state: "missing",
      hint: "No voice selected.",
      durationSecs: null,
    };
  }
  if (!entry) {
    return {
      state: "missing",
      hint: "Not generated yet.",
      durationSecs: null,
    };
  }
  const textOk = entry.text === text;
  const engineOk = entry.engine === engine;
  const voiceOk = entry.voiceId === voice;
  const speedOk = Math.abs(entry.speed - settings.speed) < 1e-4;
  const pitchOk = Math.abs(entry.pitch - settings.pitch) < 1e-4;
  const volumeOk = Math.abs(entry.volume - settings.volume) < 1e-4;
  const deviceOk = entry.device === settings.device;
  if (
    textOk &&
    engineOk &&
    voiceOk &&
    speedOk &&
    pitchOk &&
    volumeOk &&
    deviceOk
  ) {
    return {
      state: "generated",
      hint: `Cached ${entry.file}`,
      durationSecs: entry.durationSecs,
    };
  }
  return {
    state: "stale",
    hint: "Text / voice / settings changed since the last generation.",
    durationSecs: entry.durationSecs,
  };
}

// ---------------- Sync helpers ----------------

type SyncRowState = "fits" | "adjusted" | "too_long" | "empty" | "stale" | "missing";

type SyncRowStatus = {
  state: SyncRowState;
  label: string;
  hint: string;
};

/**
 * The row-level classification mirrors the Python planner but works on
 * whatever is already in `sync.json`. If the entry is present but its
 * `ttsCacheKey`, target duration, or settings drifted, we surface a
 * `stale` status so the user knows to resync.
 */
function computeSyncStatus(
  seg: SubtitleSegment,
  entry: SyncSegmentEntry | null,
  manifest: SyncManifest | null,
  settings: SyncSettings,
): SyncRowStatus {
  const targetDuration = Math.max(0, seg.end - seg.start);
  if (!entry) {
    return {
      state: "missing",
      label: "⚠ Not synced",
      hint: "Run voice sync to align this line to its subtitle window.",
    };
  }
  const targetMatches =
    Math.abs(entry.targetDurationSecs - targetDuration) < 1e-3;
  const settingsMatch =
    manifest &&
    Math.abs(manifest.settings.minSpeed - settings.minSpeed) < 1e-4 &&
    Math.abs(manifest.settings.maxSpeed - settings.maxSpeed) < 1e-4 &&
    manifest.settings.outputChannels === settings.outputChannels &&
    (manifest.settings.outputSampleRate ?? null) ===
      (settings.outputSampleRate ?? null);
  if (!targetMatches || !settingsMatch) {
    return {
      state: "stale",
      label: "⚠ Outdated",
      hint: "Subtitle timing or sync settings changed since this WAV was built.",
    };
  }
  switch (entry.status) {
    case "fits":
      return {
        state: "fits",
        label: `✓ Fits · ${entry.finalDurationSecs.toFixed(2)}s`,
        hint: `Padded with silence to fit ${targetDuration.toFixed(2)}s window.`,
      };
    case "adjusted":
      return {
        state: "adjusted",
        label: `⚠ Adjusted · ${entry.speedFactor.toFixed(2)}×`,
        hint: `Time-stretched by ${entry.speedFactor.toFixed(2)}× to fit ${targetDuration.toFixed(2)}s.`,
      };
    case "too_long":
      return {
        state: "too_long",
        label: "⚠ Too long",
        hint: `Voice (${entry.originalDurationSecs.toFixed(2)}s) still exceeds ${targetDuration.toFixed(2)}s at the max allowed speed. Shorten the translation or extend the timing.`,
      };
    case "empty":
      return {
        state: "empty",
        label: "· Empty",
        hint: "No speech — this window is pure silence.",
      };
    default:
      return {
        state: "missing",
        label: "⚠ Not synced",
        hint: "Unknown status.",
      };
  }
}

// ---------------- Sync panel ----------------

function SyncPanel(props: {
  env: SyncEnv | null;
  summary: SyncSummary | null;
  subtitleDoc: SubtitleDoc | null;
  ttsSummary: TtsSummary | null;
  settings: SyncSettings;
  onSettingsChange: (settings: Partial<SyncSettings>) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  busy: boolean;
  onApplyMissing: () => void;
  onApplyAll: () => void;
  onCancel: () => void;
  lastPreview: PreviewSyncResult | null;
}) {
  const {
    env,
    summary,
    subtitleDoc,
    ttsSummary,
    settings,
    onSettingsChange,
    activeJob,
    progress,
    busy,
    onApplyMissing,
    onApplyAll,
    onCancel,
    lastPreview,
  } = props;

  const hasSubtitles = (subtitleDoc?.segments.length ?? 0) > 0;
  const hasVoices = (ttsSummary?.generatedCount ?? 0) > 0;
  const ffmpegOk = env?.ffmpegAvailable ?? false;
  const canApply = hasSubtitles && hasVoices && ffmpegOk && !activeJob;

  return (
    <div className="sync-panel">
      <div className="sync-grid">
        <label>
          <span>Min speed ({settings.minSpeed.toFixed(2)}×)</span>
          <input
            type="range"
            min={0.5}
            max={1.0}
            step={0.01}
            value={settings.minSpeed}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ minSpeed: parseFloat(e.target.value) })
            }
          />
        </label>
        <label>
          <span>Max speed ({settings.maxSpeed.toFixed(2)}×)</span>
          <input
            type="range"
            min={1.0}
            max={2.0}
            step={0.01}
            value={settings.maxSpeed}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ maxSpeed: parseFloat(e.target.value) })
            }
          />
        </label>
        <label>
          <span>Channels</span>
          <select
            value={settings.outputChannels}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ outputChannels: parseInt(e.target.value, 10) })
            }
          >
            <option value={1}>Mono (1)</option>
            <option value={2}>Stereo (2)</option>
          </select>
        </label>
      </div>

      {env && (
        <div className="sync-env small">
          <div>
            <span>FFmpeg</span>{" "}
            {ffmpegOk ? (
              <code className="mono small">{env.ffmpegPath ?? "ffmpeg"}</code>
            ) : (
              <em>not available — install ffmpeg to enable sync</em>
            )}
          </div>
        </div>
      )}

      {summary && (
        <div className="sync-summary">
          <div className="kv-row">
            <span>Coverage</span>
            <b>
              {summary.syncedCount}/{summary.subtitleCount} synced
              {summary.staleCount > 0 ? ` · ${summary.staleCount} outdated` : ""}
              {summary.missingCount > 0
                ? ` · ${summary.missingCount} missing`
                : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Fit breakdown</span>
            <b>
              ✓ {summary.fitsCount} fits · ⚠ {summary.adjustedCount} adjusted ·
              ⚠ {summary.tooLongCount} too long
              {summary.emptyCount > 0 ? ` · · ${summary.emptyCount} empty` : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Manifest</span>
            <code className="mono small">{summary.relativePath}</code>
          </div>
        </div>
      )}

      <div className="sync-actions">
        <button
          className="btn"
          onClick={onApplyMissing}
          disabled={!canApply || busy}
        >
          Sync missing
        </button>
        <button
          className="btn primary"
          onClick={onApplyAll}
          disabled={!canApply || busy}
        >
          Sync all
        </button>
        {activeJob && (
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      {activeJob && (
        <div className="sync-progress">
          <ProgressRow label="Aligning" pct={progress} />
        </div>
      )}

      {!hasSubtitles && (
        <div className="empty-state small">
          Build subtitles before running voice sync.
        </div>
      )}
      {hasSubtitles && !hasVoices && (
        <div className="empty-state small">
          Generate TTS voices before running voice sync.
        </div>
      )}
      {hasSubtitles && hasVoices && !ffmpegOk && (
        <div className="empty-state small">
          FFmpeg is required for time-stretching and silence padding. Install
          FFmpeg and rescan.
        </div>
      )}

      {summary && summary.tooLongCount > 0 && (
        <div className="banner banner--warn small">
          {summary.tooLongCount}{" "}
          {summary.tooLongCount === 1 ? "line does" : "lines do"} not fit
          within the allowed speed range. Shorten the translation or extend
          the subtitle window, then resync.
        </div>
      )}

      {lastPreview && (
        <div className="sync-preview">
          <div className="small">
            Last preview: segment #{lastPreview.segmentId} · target{" "}
            {lastPreview.targetDurationSecs.toFixed(2)}s → final{" "}
            {lastPreview.finalDurationSecs.toFixed(2)}s (
            {lastPreview.speedFactor.toFixed(2)}×,{" "}
            {lastPreview.cacheHit ? "cached" : "fresh"})
          </div>
          <audio id="sync-preview-audio" controls preload="none" />
        </div>
      )}
    </div>
  );
}

// ---------------- Mix panel ----------------

function MixPanel(props: {
  env: MixEnv | null;
  summary: MixSummary | null;
  syncSummary: SyncSummary | null;
  settings: MixSettings;
  onSettingsChange: (settings: Partial<MixSettings>) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  busy: boolean;
  onApply: () => void;
  onCancel: () => void;
  onPlay: () => void;
  lastPreview: PreviewMixResult | null;
}) {
  const {
    env,
    summary,
    syncSummary,
    settings,
    onSettingsChange,
    activeJob,
    progress,
    busy,
    onApply,
    onCancel,
    onPlay,
    lastPreview,
  } = props;

  const ffmpegOk = env?.ffmpegAvailable ?? false;
  const hasSynced = (syncSummary?.syncedCount ?? 0) > 0;
  const canApply = hasSynced && ffmpegOk && !activeJob;

  return (
    <div className="mix-panel">
      <div className="mix-grid">
        <label>
          <span>Original volume ({Math.round(settings.originalVolume * 100)}%)</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={settings.originalVolume}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ originalVolume: parseFloat(e.target.value) })
            }
          />
        </label>
        <label>
          <span>Voice volume ({Math.round(settings.voiceVolume * 100)}%)</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={settings.voiceVolume}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ voiceVolume: parseFloat(e.target.value) })
            }
          />
        </label>
        <label className="mix-toggle">
          <input
            type="checkbox"
            checked={settings.duckingEnabled}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ duckingEnabled: e.target.checked })
            }
          />
          <span>Enable ducking</span>
        </label>
        <label>
          <span>
            Ducking depth ({settings.duckingDepthDb.toFixed(0)} dB)
          </span>
          <input
            type="range"
            min={0}
            max={30}
            step={1}
            value={settings.duckingDepthDb}
            disabled={!!activeJob || !settings.duckingEnabled}
            onChange={(e) =>
              onSettingsChange({ duckingDepthDb: parseFloat(e.target.value) })
            }
          />
        </label>
        <label>
          <span>Channels</span>
          <select
            value={settings.outputChannels}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({ outputChannels: parseInt(e.target.value, 10) })
            }
          >
            <option value={1}>Mono (1)</option>
            <option value={2}>Stereo (2)</option>
          </select>
        </label>
      </div>

      {env && (
        <div className="mix-env small">
          <div>
            <span>FFmpeg</span>{" "}
            {ffmpegOk ? (
              <code className="mono small">{env.ffmpegPath ?? "ffmpeg"}</code>
            ) : (
              <em>not available — install ffmpeg to enable mixing</em>
            )}
          </div>
        </div>
      )}

      {summary && (
        <div className="mix-summary">
          <div className="kv-row">
            <span>Status</span>
            <b>
              <MixStatusBadge status={summary.status} />
              {summary.needsGenerate && summary.status !== "missing"
                ? " · needs regeneration"
                : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Coverage</span>
            <b>
              {summary.voiceSegmentCount}/{summary.subtitleCount} voice segments
              {summary.durationSecs != null
                ? ` · ${formatDuration(summary.durationSecs)}`
                : ""}
              {summary.sizeBytes != null
                ? ` · ${humanBytes(summary.sizeBytes)}`
                : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Manifest</span>
            <code className="mono small">{summary.manifestRelativePath}</code>
          </div>
          {summary.relativePath && (
            <div className="kv-row">
              <span>Output</span>
              <code className="mono small">{summary.relativePath}</code>
            </div>
          )}
        </div>
      )}

      <div className="mix-actions">
        <button
          className="btn primary"
          onClick={onApply}
          disabled={!canApply || busy}
        >
          {summary?.status === "ready" && !summary.needsGenerate
            ? "Regenerate mix"
            : "Generate mix"}
        </button>
        {activeJob && (
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
        {lastPreview && (
          <button className="btn" onClick={onPlay}>
            ▶ Play mix
          </button>
        )}
      </div>

      {activeJob && (
        <div className="mix-progress">
          <ProgressRow label="Mixing" pct={progress} />
        </div>
      )}

      {!hasSynced && (
        <div className="empty-state small">
          Run voice sync so per-segment WAVs exist under{" "}
          <code>voices/synced/</code>, then generate the mix.
        </div>
      )}
      {hasSynced && !ffmpegOk && (
        <div className="empty-state small">
          FFmpeg is required for mixing. Install FFmpeg and rescan.
        </div>
      )}

      {summary?.warning && (
        <div className="banner banner--warn small">{summary.warning}</div>
      )}

      {lastPreview && (
        <div className="mix-preview">
          <div className="small">
            Last mix: {formatDuration(lastPreview.durationSecs)} ·{" "}
            {lastPreview.sampleRate} Hz ·{" "}
            {lastPreview.channels === 1 ? "mono" : "stereo"}{" "}
            ({lastPreview.cacheHit ? "cached" : "fresh"})
          </div>
          <audio id="mix-preview-audio" controls preload="none" />
        </div>
      )}
    </div>
  );
}

function MixStatusBadge({ status }: { status: MixSummary["status"] }) {
  if (status === "ready") return <span className="badge badge--ok">Ready</span>;
  if (status === "stale")
    return <span className="badge badge--warn">Stale</span>;
  return <span className="badge badge--muted">Missing</span>;
}

// ---------------- Render panel ----------------

function RenderPanel(props: {
  env: RenderEnv | null;
  summary: RenderSummary | null;
  mixSummary: MixSummary | null;
  subtitleSummary: SubtitleSummary | null;
  settings: RenderSettings;
  onSettingsChange: (settings: Partial<RenderSettings>) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  busy: boolean;
  onApply: () => void;
  onRegenerate: () => void;
  onCancel: () => void;
  onPickOutputPath: () => void;
  onClearOutputPath: () => void;
}) {
  const {
    env,
    summary,
    mixSummary,
    subtitleSummary,
    settings,
    onSettingsChange,
    activeJob,
    progress,
    busy,
    onApply,
    onRegenerate,
    onCancel,
    onPickOutputPath,
    onClearOutputPath,
  } = props;

  const ffmpegOk = env?.ffmpegAvailable ?? false;
  const hasMix = mixSummary?.status === "ready";
  const hasSubtitles = (subtitleSummary?.segmentCount ?? 0) > 0;
  const canApply = ffmpegOk && hasMix && hasSubtitles && !activeJob;
  // Burning needs libass, which this FFmpeg may not have been built with.
  const burnSupported = env?.subtitleBurnAvailable ?? false;
  const canBurn = hasSubtitles && burnSupported;

  const videoCodecs = env?.videoCodecs ?? ["copy", "libx264", "libx265"];
  const audioCodecs = env?.audioCodecs ?? ["aac", "libopus", "ac3", "mp3"];
  const outputFormats = env?.outputFormats ?? ["mp4", "mkv"];

  const currentVideoCodec =
    settings.videoCodec.kind === "copy"
      ? "copy"
      : settings.videoCodec.codec;

  const onVideoCodecChange = (value: string) => {
    if (value === "copy") {
      onSettingsChange({ videoCodec: { kind: "copy" } });
    } else {
      onSettingsChange({ videoCodec: { kind: "reencode", codec: value } });
    }
  };

  return (
    <div className="render-panel">
      <div className="render-grid">
        <label>
          <span>Subtitle mode</span>
          <select
            value={settings.subtitleMode}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({
                subtitleMode: e.target.value as SubtitleMode,
              })
            }
          >
            <option value="none">None (no subtitles)</option>
            <option value="external">
              External (.srt file beside the movie)
            </option>
            <option value="burned" disabled={!canBurn}>
              Burned into video (always visible)
              {burnSupported ? "" : " — needs FFmpeg with libass"}
            </option>
          </select>
        </label>

        <label>
          <span>Output format</span>
          <select
            value={settings.outputFormat}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({
                outputFormat: e.target.value as OutputFormat,
              })
            }
          >
            {outputFormats.map((f) => (
              <option key={f} value={f}>
                {f.toUpperCase()}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Video codec</span>
          <select
            value={currentVideoCodec}
            disabled={!!activeJob || settings.subtitleMode === "burned"}
            onChange={(e) => onVideoCodecChange(e.target.value)}
          >
            {videoCodecs.map((c) => (
              <option key={c} value={c}>
                {c === "copy" ? "Copy (no re-encode — fastest)" : c}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Audio codec</span>
          <select
            value={settings.audioCodec.codec}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({
                audioCodec: { ...settings.audioCodec, codec: e.target.value },
              })
            }
          >
            {audioCodecs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Audio bitrate</span>
          <input
            type="text"
            placeholder="192k"
            value={settings.audioCodec.bitrate ?? ""}
            disabled={!!activeJob}
            onChange={(e) =>
              onSettingsChange({
                audioCodec: {
                  ...settings.audioCodec,
                  bitrate: e.target.value.trim() || null,
                },
              })
            }
          />
        </label>
      </div>

      <div className="render-output">
        <div className="kv-row">
          <span>Output path</span>
          <code className="mono small">
            {settings.outputPath ?? summary?.defaultOutputAbsolute ?? ""}
          </code>
        </div>
        <div className="actions">
          <button
            className="btn"
            onClick={onPickOutputPath}
            disabled={!!activeJob}
          >
            Choose output path…
          </button>
          {settings.outputPath && (
            <button
              className="btn ghost"
              onClick={onClearOutputPath}
              disabled={!!activeJob}
            >
              Use default
            </button>
          )}
        </div>
      </div>

      {settings.subtitleMode === "burned" && (
        <div className="banner banner--muted small">
          Burning subtitles forces a video re-encode (libx264).
        </div>
      )}

      {ffmpegOk && !burnSupported && (
        <div className="banner banner--warn small">
          This FFmpeg was built without libass, so it cannot burn subtitles
          into the picture. External mode writes a <code>.srt</code> beside
          the movie, which only shows up if the player loads it. To burn
          them in, install a full FFmpeg build (macOS:{" "}
          <code className="mono">brew install ffmpeg</code>) and point
          Settings → FFmpeg path at it.
        </div>
      )}

      {settings.subtitleMode === "external" && (
        <div className="banner banner--muted small">
          The .srt is written next to the movie. Most players need it turned
          on manually, and uploads usually drop it — burn the subtitles in if
          the file has to carry them.
        </div>
      )}

      {env && (
        <div className="render-env small">
          <div>
            <span>FFmpeg</span>{" "}
            {ffmpegOk ? (
              <code className="mono small">{env.ffmpegPath ?? "ffmpeg"}</code>
            ) : (
              <em>not available — install ffmpeg to enable rendering</em>
            )}
          </div>
        </div>
      )}

      {summary && (
        <div className="render-summary">
          <div className="kv-row">
            <span>Status</span>
            <b>
              <RenderStatusBadge status={summary.status} />
              {summary.needsRender && summary.status !== "missing"
                ? " · needs regeneration"
                : ""}
            </b>
          </div>
          {summary.durationSecs != null && (
            <div className="kv-row">
              <span>Duration</span>
              <b>{formatDuration(summary.durationSecs)}</b>
            </div>
          )}
          {summary.sizeBytes != null && (
            <div className="kv-row">
              <span>Size</span>
              <b>{humanBytes(summary.sizeBytes)}</b>
            </div>
          )}
          {summary.absolutePath && (
            <div className="kv-row">
              <span>Movie</span>
              <code className="mono small">{summary.absolutePath}</code>
            </div>
          )}
          {summary.subtitleAbsolutePath && (
            <div className="kv-row">
              <span>Subtitle</span>
              <code className="mono small">
                {summary.subtitleAbsolutePath}
              </code>
            </div>
          )}
          <div className="kv-row">
            <span>Streams</span>
            <b>
              {summary.videoStreamCount}v · {summary.audioStreamCount}a ·{" "}
              {summary.subtitleStreamCount}s
            </b>
          </div>
          <div className="kv-row">
            <span>Manifest</span>
            <code className="mono small">{summary.manifestRelativePath}</code>
          </div>
        </div>
      )}

      <div className="render-actions">
        <button
          className="btn primary"
          onClick={onApply}
          disabled={!canApply || busy}
        >
          {summary?.status === "ready" && !summary.needsRender
            ? "Up to date"
            : "Render movie"}
        </button>
        {summary?.status === "ready" && !summary.needsRender && (
          <button
            className="btn"
            onClick={onRegenerate}
            disabled={!canApply || busy}
          >
            Regenerate
          </button>
        )}
        {activeJob && (
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      {activeJob && (
        <div className="render-progress">
          <ProgressRow label="Rendering" pct={progress} />
        </div>
      )}

      {!hasSubtitles && (
        <div className="empty-state small">
          Build subtitles first — Phase 9 needs Vietnamese subtitles to
          render.
        </div>
      )}
      {hasSubtitles && !hasMix && (
        <div className="empty-state small">
          Run audio mix so <code>audio/mixed_vi.wav</code> exists, then
          render.
        </div>
      )}
      {hasSubtitles && hasMix && !ffmpegOk && (
        <div className="empty-state small">
          FFmpeg is required for rendering. Install FFmpeg and rescan.
        </div>
      )}

      {summary?.warning && (
        <div className="banner banner--warn small">{summary.warning}</div>
      )}
    </div>
  );
}

function RenderStatusBadge({ status }: { status: RenderSummary["status"] }) {
  if (status === "ready") return <span className="badge badge--ok">Ready</span>;
  if (status === "stale")
    return <span className="badge badge--warn">Stale</span>;
  return <span className="badge badge--muted">Missing</span>;
}

// ---------------- TTS panel ----------------

function formatTtsProgressStage(stage: string): string {
  switch (stage) {
    case "preparing":
      return "Preparing";
    case "loading_model":
      return "Loading model";
    case "generating_voice":
      return "Generating voice";
    case "completed":
      return "Completed";
    default:
      return stage.replaceAll("_", " ");
  }
}

function TtsPanel(props: {
  env: TtsEnv | null;
  voices: VoiceInfo[];
  recommendedPresets: TtsRecommendedVoicePreset[];
  targetLanguage: string;
  manifest: TtsManifest | null;
  summary: TtsSummary | null;
  subtitleDoc: SubtitleDoc | null;
  engine: string;
  qualityMode: "fast" | "balanced" | "quality";
  voiceId: string;
  settings: TtsSettings;
  onEngineChange: (engine: string) => void;
  onQualityModeChange: (mode: "fast" | "balanced" | "quality") => void;
  onVoiceChange: (voiceId: string) => void;
  onSettingsChange: (settings: Partial<TtsSettings>) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  progressDetail: import("@/ipc/types").TtsProgressDetailEvent | null;
  downloadJob: JobSnapshot | null;
  downloadProgress: number;
  busy: boolean;
  onGenerateMissing: () => void;
  onGenerateAll: () => void;
  onCancel: () => void;
  onCancelDownload: () => void;
  onRescanVoices: () => void;
  onInstallF5: () => Promise<void>;
  onCreateVoiceProfile: (
    request: import("@/ipc/types").CreateTtsVoiceProfileRequest,
  ) => Promise<VoiceInfo>;
  onAssignSpeakerVoice: (speaker: string, voiceId: string | null) => void;
  lastPreview: PreviewResult | null;
}) {
  const {
    env,
    voices,
    recommendedPresets,
    targetLanguage,
    summary,
    subtitleDoc,
    engine,
    qualityMode,
    voiceId,
    settings,
    onEngineChange,
    onQualityModeChange,
    onVoiceChange,
    onSettingsChange,
    activeJob,
    progress,
    progressDetail,
    downloadJob,
    downloadProgress,
    busy,
    onGenerateMissing,
    onGenerateAll,
    onCancel,
    onCancelDownload,
    onRescanVoices,
    onInstallF5,
    onCreateVoiceProfile,
    onAssignSpeakerVoice,
    lastPreview,
  } = props;
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [profileId, setProfileId] = useState("");
  const [profileName, setProfileName] = useState("");
  const [profileGender, setProfileGender] = useState<
    "male" | "female" | "neutral" | "unknown"
  >("unknown");
  const [referenceAudioPath, setReferenceAudioPath] = useState("");
  const [referenceText, setReferenceText] = useState("");
  const [profileEmotion, setProfileEmotion] = useState("neutral");
  const [profileStyle, setProfileStyle] = useState("default");
  const [profileError, setProfileError] = useState<string | null>(null);

  const availableEngines = env?.engines ?? [];
  const currentEngineInfo = availableEngines.find((e) => e.id === engine);
  const supportedSettings = new Set(
    currentEngineInfo?.supportedSettings ?? ["speed"],
  );
  const enginesVoices = useMemo(
    () => voices.filter((v) => v.engine === engine),
    [voices, engine],
  );
  const selectedVoice = enginesVoices.find((voice) => voice.id === voiceId) ?? null;
  const speakerMappings = useMemo(() => {
    const mappings = new Map<
      string,
      { count: number; voiceId: string; mixed: boolean }
    >();
    for (const segment of subtitleDoc?.segments ?? []) {
      const speaker = segment.speaker?.trim();
      if (!speaker) continue;
      const voice = segment.voiceId ?? "";
      const current = mappings.get(speaker);
      if (!current) {
        mappings.set(speaker, { count: 1, voiceId: voice, mixed: false });
      } else {
        current.count += 1;
        if (current.voiceId !== voice) current.mixed = true;
      }
    }
    return [...mappings.entries()];
  }, [subtitleDoc]);
  const hasSubtitles = (subtitleDoc?.segments.length ?? 0) > 0;
  // Phase 12 — pick the preset the app will auto-download when the
  // user has nothing installed. Prefer one that speaks the
  // project's target language, then fall back to the default.
  const enginePresets = recommendedPresets.filter(
    (preset) => preset.engine === engine,
  );
  const targetPreset =
    enginePresets.find((p) =>
      p.targetLanguages.includes((targetLanguage || "").toLowerCase()),
    ) ??
    enginePresets.find((p) => p.isDefault) ??
    enginePresets[0] ??
    null;
  const canAutoDownloadVoice =
    engine === "piper" &&
    !!targetPreset &&
    !!currentEngineInfo?.available &&
    hasSubtitles;
  const willAutoDownloadVoice = !voiceId && canAutoDownloadVoice;
  const f5Ready =
    engine !== "f5-vietnamese" ||
    (!!env?.f5RuntimeInstalled && !!env.f5Model.installed);
  const canGenerate =
    (!!voiceId || willAutoDownloadVoice) &&
    !!currentEngineInfo?.available &&
    f5Ready &&
    hasSubtitles &&
    !activeJob;

  const handleCreateProfile = async () => {
    setProfileError(null);
    try {
      await onCreateVoiceProfile({
        id: profileId,
        name: profileName,
        gender: profileGender,
        referenceAudioPath,
        referenceText,
        emotion: profileEmotion,
        style: profileStyle,
      });
      setProfileId("");
      setProfileName("");
      setReferenceAudioPath("");
      setReferenceText("");
    } catch (error) {
      setProfileError(formatError(error));
    }
  };

  return (
    <div className="tts-panel">
      <div className="tts-grid">
        <label>
          <span>Quality mode</span>
          <select
            value={qualityMode}
            onChange={(event) =>
              onQualityModeChange(
                event.target.value as "fast" | "balanced" | "quality",
              )
            }
            disabled={!!activeJob}
          >
            <option value="fast">FAST · Piper</option>
            <option value="balanced">BALANCED · Piper</option>
            <option value="quality">QUALITY · F5-TTS Vietnamese</option>
          </select>
        </label>
        <label>
          <span>Engine</span>
          <select
            value={engine}
            onChange={(e) => onEngineChange(e.target.value)}
            disabled={!!activeJob}
          >
            {availableEngines.length === 0 && (
              <option value={engine}>{engine}</option>
            )}
            {availableEngines.map((e) => (
              <option key={e.id} value={e.id} disabled={!e.available}>
                {e.name}
                {e.available ? "" : " (not installed)"}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>
            Voice
            <button
              className="btn ghost small tts-rescan"
              type="button"
              onClick={onRescanVoices}
              title="Rescan installed voices"
            >
              Rescan
            </button>
          </span>
          <select
            value={voiceId}
            onChange={(e) => onVoiceChange(e.target.value)}
            disabled={!!activeJob || enginesVoices.length === 0}
          >
            {enginesVoices.length === 0 && (
              <option value="">No voices installed</option>
            )}
            {enginesVoices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name} · {v.language}
                {v.gender !== "unknown" ? ` · ${v.gender}` : ""}
              </option>
            ))}
          </select>
        </label>
        {engine === "f5-vietnamese" && (
          <label>
            <span>Compute device</span>
            <select
              value={settings.device}
              onChange={(event) =>
                onSettingsChange({
                  device: event.target.value as TtsSettings["device"],
                })
              }
              disabled={!!activeJob}
            >
              <option value="auto">Auto ({env?.f5Hardware.backend ?? "detect"})</option>
              <option value="cuda">CUDA GPU</option>
              <option value="mps">Apple MPS</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
        )}
        <label>
          <span>Speed ({settings.speed.toFixed(2)}×)</span>
          <input
            type="range"
            min={engine === "f5-vietnamese" ? 0.9 : 0.5}
            max={engine === "f5-vietnamese" ? 1.12 : 2.0}
            step={0.05}
            value={settings.speed}
            disabled={!!activeJob || !supportedSettings.has("speed")}
            onChange={(e) =>
              onSettingsChange({ speed: parseFloat(e.target.value) })
            }
          />
        </label>
        {supportedSettings.has("pitch") && (
          <label>
            <span>Pitch ({settings.pitch.toFixed(0)} st)</span>
            <input
              type="range"
              min={-12}
              max={12}
              step={1}
              value={settings.pitch}
              disabled={!!activeJob}
              onChange={(e) =>
                onSettingsChange({ pitch: parseFloat(e.target.value) })
              }
            />
          </label>
        )}
        {supportedSettings.has("volume") && (
          <label>
            <span>Volume ({Math.round(settings.volume * 100)}%)</span>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.05}
              value={settings.volume}
              disabled={!!activeJob}
              onChange={(e) =>
                onSettingsChange({ volume: parseFloat(e.target.value) })
              }
            />
          </label>
        )}
      </div>

      {env && (
        <div className="tts-env small">
          <div>
            <span>Voices dir</span>{" "}
            <code className="mono small">{env.ttsRoot}</code>
          </div>
          <div>
            <span>Piper installed</span>{" "}
            {env.piperInstalled ? "✓" : "no (install piper-tts in the worker env)"}
          </div>
          <div>
            <span>F5 runtime</span>{" "}
            {env.f5RuntimeInstalled
              ? "✓"
              : "not installed (run scripts/setup-f5.ps1)"}
          </div>
          <div>
            <span>F5 model</span>{" "}
            {env.f5Model.installed
              ? `✓ ${humanBytes(env.f5Model.approxSizeBytes)}`
              : "not installed"}
          </div>
        </div>
      )}

      {engine === "f5-vietnamese" && env && (
        <>
          <div className="banner banner--warn small">
            F5-TTS ViVoice is licensed {env.f5Model.license} for
            non-commercial use. It is not bundled with the app and QUALITY
            never falls back to Piper.
          </div>
          {env.f5Hardware.warning && (
            <div className="banner banner--warn small">
              {env.f5Hardware.warning}
              {env.f5Hardware.gpuName
                ? ` GPU: ${env.f5Hardware.gpuName}${
                    env.f5Hardware.vramGb
                      ? ` (${env.f5Hardware.vramGb} GB VRAM)`
                      : ""
                  }.`
                : ""}
            </div>
          )}
          {!env.f5Model.installed && (
            <div className="tts-actions">
              <button
                className="btn"
                type="button"
                disabled={!!downloadJob}
                onClick={() => {
                  setProfileError(null);
                  void onInstallF5().catch((error) =>
                    setProfileError(formatError(error)),
                  );
                }}
              >
                Install F5 model ({humanBytes(env.f5Model.approxSizeBytes)})
              </button>
              <span className="small muted">
                Explicit one-time download; inference is offline afterwards.
              </span>
            </div>
          )}
          {env.f5Model.installed && (
            <div className="tts-advanced">
              <button
                className="btn ghost small"
                type="button"
                onClick={() => setAdvancedOpen((value) => !value)}
              >
                {advancedOpen ? "Hide advanced" : "Advanced · reference voice"}
              </button>
              {advancedOpen && (
                <div className="tts-grid">
                  <label>
                    <span>Profile ID</span>
                    <input
                      value={profileId}
                      placeholder="character-01"
                      onChange={(event) => setProfileId(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>Display name</span>
                    <input
                      value={profileName}
                      placeholder="Vietnamese Male 01"
                      onChange={(event) => setProfileName(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>Gender</span>
                    <select
                      value={profileGender}
                      onChange={(event) =>
                        setProfileGender(
                          event.target.value as typeof profileGender,
                        )
                      }
                    >
                      <option value="unknown">Unspecified</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="neutral">Neutral</option>
                    </select>
                  </label>
                  <label>
                    <span>Emotion metadata</span>
                    <select
                      value={profileEmotion}
                      onChange={(event) =>
                        setProfileEmotion(event.target.value)
                      }
                    >
                      {[
                        "neutral",
                        "happy",
                        "sad",
                        "angry",
                        "afraid",
                        "surprised",
                        "serious",
                        "excited",
                        "calm",
                        "whisper",
                      ].map((emotion) => (
                        <option key={emotion} value={emotion}>
                          {emotion}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Style metadata</span>
                    <input
                      value={profileStyle}
                      placeholder="default"
                      onChange={(event) => setProfileStyle(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>Reference WAV</span>
                    <div className="path-row">
                      <input value={referenceAudioPath} readOnly />
                      <button
                        className="btn ghost small"
                        type="button"
                        onClick={() => {
                          void pickTtsReferenceAudio().then((path) => {
                            if (path) setReferenceAudioPath(path);
                          });
                        }}
                      >
                        Browse
                      </button>
                    </div>
                  </label>
                  <label className="tts-reference-text">
                    <span>Exact reference transcript</span>
                    <textarea
                      value={referenceText}
                      placeholder="Transcript matching the reference WAV exactly…"
                      onChange={(event) => setReferenceText(event.target.value)}
                    />
                  </label>
                  <div className="tts-actions">
                    <button
                      className="btn"
                      type="button"
                      disabled={
                        !profileId.trim() ||
                        !profileName.trim() ||
                        !referenceAudioPath ||
                        !referenceText.trim()
                      }
                      onClick={() => void handleCreateProfile()}
                    >
                      Create local voice profile
                    </button>
                  </div>
                  <div className="small muted tts-reference-text">
                    ViVoice does not expose reliable emotion or pitch controls;
                    these fields are stored as metadata for future local
                    providers and are not simulated with extreme speed changes.
                  </div>
                </div>
              )}
            </div>
          )}
          {profileError && (
            <div className="banner banner--warn small">{profileError}</div>
          )}
        </>
      )}

      {speakerMappings.length > 0 && (
        <div className="tts-character-map">
          <div className="small muted">
            Character mapping · assignments apply to every subtitle with the
            same speaker.
          </div>
          {speakerMappings.map(([speaker, mapping]) => (
            <label className="kv-row" key={speaker}>
              <span>
                {speaker} <small>({mapping.count} lines)</small>
              </span>
              <select
                value={mapping.mixed ? "__mixed__" : mapping.voiceId}
                disabled={!!activeJob}
                onChange={(event) =>
                  onAssignSpeakerVoice(
                    speaker,
                    event.target.value ? event.target.value : null,
                  )
                }
              >
                {mapping.mixed && (
                  <option value="__mixed__" disabled>
                    Mixed voices
                  </option>
                )}
                <option value="">Project default voice</option>
                {enginesVoices.map((voice) => (
                  <option key={voice.id} value={voice.id}>
                    {voice.name}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}

      {selectedVoice?.engine === "f5-vietnamese" && (
        <div className="tts-env small">
          <div>
            <span>Reference audio</span>{" "}
            <code className="mono small">
              {selectedVoice.referenceAudioPath}
            </code>
          </div>
          <div>
            <span>Model</span>{" "}
            {selectedVoice.modelName} · {selectedVoice.license}
          </div>
          <div>
            <span>Metadata</span>{" "}
            {selectedVoice.emotion ?? "neutral"} ·{" "}
            {selectedVoice.style ?? "default"}
          </div>
        </div>
      )}

      {summary && (
        <div className="tts-summary">
          <div className="kv-row">
            <span>Coverage</span>
            <b>
              {summary.generatedCount}/{summary.subtitleCount} generated
              {summary.staleCount > 0 ? ` · ${summary.staleCount} outdated` : ""}
              {summary.missingCount > 0
                ? ` · ${summary.missingCount} missing`
                : ""}
            </b>
          </div>
          <div className="kv-row">
            <span>Manifest</span>
            <code className="mono small">{summary.relativePath}</code>
          </div>
        </div>
      )}

      {downloadJob ? (
        <div className="tts-progress">
          <ProgressRow
            label={`Downloading voice${
              targetPreset ? ` — ${targetPreset.label}` : ""
            }`}
            pct={downloadProgress}
          />
          <div className="tts-actions">
            <button className="btn danger" onClick={onCancelDownload}>
              Cancel download
            </button>
          </div>
        </div>
      ) : (
        <div className="tts-actions">
          <button
            className="btn"
            onClick={onGenerateMissing}
            disabled={!canGenerate || busy}
            title={
              willAutoDownloadVoice && targetPreset
                ? `Will download ${targetPreset.label} (~${humanBytes(targetPreset.approxSizeBytes)}) and then generate.`
                : undefined
            }
          >
            {willAutoDownloadVoice && targetPreset
              ? "Download voice & generate missing"
              : "Generate missing"}
          </button>
          <button
            className="btn primary"
            onClick={onGenerateAll}
            disabled={!canGenerate || busy}
            title={
              willAutoDownloadVoice && targetPreset
                ? `Will download ${targetPreset.label} (~${humanBytes(targetPreset.approxSizeBytes)}) and then generate.`
                : undefined
            }
          >
            {willAutoDownloadVoice && targetPreset
              ? `Download voice & generate all (${targetPreset.label})`
              : "Generate all"}
          </button>
          {activeJob && (
            <button className="btn ghost" onClick={onCancel}>
              Cancel
            </button>
          )}
        </div>
      )}

      {activeJob && !downloadJob && (
        <div className="tts-progress">
          <ProgressRow
            label={
              progressDetail
                ? `${formatTtsProgressStage(progressDetail.stage)}${
                    progressDetail.totalSegments > 0
                      ? ` · ${progressDetail.completedSegments}/${progressDetail.totalSegments}`
                      : ""
                  }`
                : "Generating voice"
            }
            pct={progress}
          />
        </div>
      )}

      {!hasSubtitles && (
        <div className="empty-state small">
          Build subtitles before generating voice.
        </div>
      )}
      {hasSubtitles && !voiceId && willAutoDownloadVoice && !downloadJob && (
        <div className="banner">
          No voice installed yet. Clicking <em>Generate</em> will
          download <b>{targetPreset!.label}</b> to{" "}
          <code>{env?.ttsRoot ?? "<models>/tts"}</code> and then start
          synthesis. This is a one-time download.
        </div>
      )}
      {hasSubtitles && !voiceId && !canAutoDownloadVoice && (
        <div className="empty-state small">
          Install at least one voice model under{" "}
          <code className="mono small">
            {env?.ttsRoot ?? "<models>/tts"}
          </code>
          , then click <em>Rescan</em>.
        </div>
      )}

      {lastPreview && (
        <div className="tts-preview">
          <div className="small">
            Last preview: segment #{lastPreview.segmentId} ·{" "}
            {lastPreview.durationSecs.toFixed(2)}s
            {lastPreview.cacheHit ? " (cached)" : " (fresh)"}
          </div>
          <audio id="tts-preview-audio" controls preload="none" />
        </div>
      )}
    </div>
  );
}

function findSubtitleAtTime(
  doc: SubtitleDoc | null,
  time: number,
): SubtitleSegment | null {
  const segments = doc?.segments;
  if (!segments?.length) return null;

  // Subtitle documents are kept sorted by start time by the Rust
  // service. Find the last cue whose start is at/before the playhead
  // instead of scanning every preceding cue on each timeupdate.
  let low = 0;
  let high = segments.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (segments[middle].start <= time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return null;
  const candidate = segments[low - 1];
  return time < candidate.end ? candidate : null;
}

function findActiveSubtitleId(
  doc: SubtitleDoc | null,
  time: number,
): number | null {
  return findSubtitleAtTime(doc, time)?.id ?? null;
}

/** `HH:MM:SS.ms` — human-readable timecode (SRT-like separator). */
function formatTimecode(t: number): string {
  if (!isFinite(t) || t < 0) return "00:00:00.000";
  const total_ms = Math.round(t * 1000);
  const ms = total_ms % 1000;
  const total_s = Math.floor(total_ms / 1000);
  const s = total_s % 60;
  const total_m = Math.floor(total_s / 60);
  const m = total_m % 60;
  const h = Math.floor(total_m / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

/** Accepts `HH:MM:SS.ms`, `MM:SS.ms`, `SS.ms`. Returns seconds or null. */
function parseTimecode(input: string): number | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const m = trimmed.match(/^(?:(\d+):)?(?:(\d+):)?(\d+(?:[.,]\d+)?)$/);
  if (!m) return null;
  const h = m[1] ? Number(m[1]) : 0;
  const mm = m[2] ? Number(m[2]) : 0;
  const ss = Number(m[3].replace(",", "."));
  if (![h, mm, ss].every(Number.isFinite)) return null;
  return h * 3600 + mm * 60 + ss;
}

// ---------------- Shared ----------------

function Panel(props: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel">
      <h3>{props.title}</h3>
      <div className="panel-body">{props.children}</div>
    </div>
  );
}

function ProgressRow(props: { label: string; pct: number }) {
  const pct = Math.max(0, Math.min(1, props.pct));
  return (
    <div className="progress-row">
      <div className="progress-label">{props.label}</div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct * 100}%` }} />
      </div>
      <div className="progress-value">{Math.round(pct * 100)}%</div>
    </div>
  );
}

function formatError(err: unknown): string {
  const e = err as AppError;
  if (e && typeof e === "object" && "code" in e) {
    return e.hint ? `${e.message}\n\nHint: ${e.hint}` : e.message;
  }
  return err instanceof Error ? err.message : JSON.stringify(err);
}

function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "0:00";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}
