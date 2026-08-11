import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  assetUrl,
  pickMediaFile,
  pickRenderOutputPath,
  pickSubtitleFile,
  pickSubtitleSavePath,
} from "@/ipc/bridge";
import { useAppStore } from "@/state/store";
import {
  defaultSttOptions,
  defaultTranslateOptions,
  isAppError,
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
  { code: null, label: "Auto detect (recommended)" },
  ...TRANSLATION_LANGUAGES.map((l) => ({ code: l.code, label: l.label })),
];

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

// Phase 12 UX — resolve when the given job reaches a terminal
// state (`completed` | `failed` | `cancelled`). Uses the zustand
// subscription so we don't spin on `setInterval`. A 30-minute cap
// keeps us from leaking a listener if the worker dies silently
// mid-download.
function waitForJobTerminal(
  jobId: string,
  timeoutMs = 30 * 60_000,
): Promise<JobSnapshot> {
  return new Promise((resolve, reject) => {
    const initial = useAppStore.getState().jobsById[jobId];
    if (initial && isTerminalStatus(initial.status)) {
      resolve(initial);
      return;
    }
    const timer = setTimeout(() => {
      unsub();
      reject(new Error("Timed out waiting for job to finish."));
    }, timeoutMs);
    const unsub = useAppStore.subscribe((state) => {
      const snap = state.jobsById[jobId];
      if (snap && isTerminalStatus(snap.status)) {
        clearTimeout(timer);
        unsub();
        resolve(snap);
      }
    });
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
  const ttsVoiceId = useAppStore((s) => s.ttsVoiceId);
  const ttsSettings = useAppStore((s) => s.ttsSettings);
  const lastTtsPreview = useAppStore((s) => s.lastTtsPreview);
  const refreshTtsEnv = useAppStore((s) => s.refreshTtsEnv);
  const refreshTtsVoices = useAppStore((s) => s.refreshTtsVoices);
  const loadTtsManifest = useAppStore((s) => s.loadTtsManifest);
  const refreshTtsSummary = useAppStore((s) => s.refreshTtsSummary);
  const setTtsEngine = useAppStore((s) => s.setTtsEngine);
  const setTtsVoiceId = useAppStore((s) => s.setTtsVoiceId);
  const setTtsSettings = useAppStore((s) => s.setTtsSettings);
  const previewTts = useAppStore((s) => s.previewTts);
  const startGenerateTts = useAppStore((s) => s.startGenerateTts);

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
  const [videoTime, setVideoTime] = useState(0);
  const seekVideo = useCallback((time: number) => {
    const el = videoRef.current;
    if (el) {
      el.currentTime = Math.max(0, time);
    }
  }, []);

  const [error, setError] = useState<string | null>(null);
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
    setError(null);
    openProject(id).catch((e) => setError(formatError(e)));
  }, [id, openProject]);

  useEffect(() => {
    // Best-effort: worker may not be up on first mount.
    void refreshSttEnv();
    void refreshWhisperModels();
    void refreshTranslationEnv();
    void refreshTranslationModels();
    void refreshTranslationRecommendedPresets();
    void refreshTtsEnv();
    void refreshTtsVoices();
    void refreshSyncEnv();
    void refreshMixEnv();
    void refreshRenderEnv();
  }, [
    refreshSttEnv,
    refreshWhisperModels,
    refreshTranslationEnv,
    refreshTranslationModels,
    refreshTranslationRecommendedPresets,
    refreshTtsEnv,
    refreshTtsVoices,
    refreshSyncEnv,
    refreshMixEnv,
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
    void refreshTtsSummary(id);
  }, [id, ttsEngine, ttsVoiceId, ttsSettings, refreshTtsSummary]);

  // Same for sync settings — coverage counts depend on min/max speed.
  useEffect(() => {
    if (!id) return;
    void refreshSyncSummary(id);
  }, [id, syncSettings, refreshSyncSummary]);

  // And for mix settings — status (Ready/Stale/Missing) depends on
  // volume + ducking, since those feed into the cache key.
  useEffect(() => {
    if (!id) return;
    void refreshMixSummary(id);
  }, [id, mixSettings, refreshMixSummary]);

  // Render settings feed into the cache key too — subtitle mode,
  // output format and codecs all move the Ready/Stale/Missing needle.
  useEffect(() => {
    if (!id) return;
    void refreshRenderSummary(id);
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

  if (error) {
    return (
      <div className="error-panel">
        <h2>Cannot open project</h2>
        <pre>{error}</pre>
        <Link to="/">← Back to dashboard</Link>
      </div>
    );
  }

  if (!project) {
    return <div className="loading">Loading project…</div>;
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

  const handleTranscribe = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      // Phase 12 UX — if the chosen Whisper model isn't downloaded
      // yet, transparently pull it before starting transcription so
      // the user doesn't have to bounce between two buttons. The
      // download itself surfaces via `activeDownloadJob` and drives
      // the existing progress UI in the STT panel; once it finishes
      // we refresh the registry (so `installed` flips to true) and
      // fall through to the real transcribe call.
      const chosen = whisperModels.find((m) => m.name === sttOptions.model);
      if (chosen && !chosen.installed) {
        const startedAt = Date.now();
        try {
          await downloadWhisperModel(chosen.name);
        } catch (dlErr) {
          // Phase 12 UX — Offline Mode blocks the ONE HTTP entry
          // point in the app. Rather than showing the raw error and
          // sending the user off to Settings, offer to flip Offline
          // Mode off in-place and retry. This is a one-click
          // recovery for what is otherwise a dead end.
          if (isAppError(dlErr) && dlErr.code === "MODEL_NETWORK_DISABLED") {
            const proceed = window.confirm(
              `To download the "${chosen.name}" model, the app needs one-time network access.\n\n` +
                `Offline Mode is currently ON, which is blocking the download.\n\n` +
                `Turn Offline Mode OFF and download now?\n` +
                `(You can turn it back on in Settings once the download finishes — the model works fully offline afterwards.)`,
            );
            if (!proceed) return;
            await updateSettings({ offlineMode: false });
            await downloadWhisperModel(chosen.name);
          } else {
            throw dlErr;
          }
        }
        const dlJobId = findLatestDownloadJobId(startedAt);
        if (dlJobId) {
          const result = await waitForJobTerminal(dlJobId);
          if (result.status === "cancelled") {
            return;
          }
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
      }
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

  const handleTranslate = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
      // Phase 12 UX — mirror of `handleTranscribe`. If the user has
      // no GGUF installed at all (or the model they picked has
      // disappeared from disk), transparently pull the recommended
      // preset from HuggingFace before starting translation. This
      // is the whole point of the "user just clicks Translate"
      // requirement: no manual model-picking dance, no digging into
      // Settings, no separate download button to hunt for.
      const chosenModelPresent =
        !!translateOptions.model &&
        translationModels.some((m) => m.name === translateOptions.model);
      if (!chosenModelPresent) {
        // Pick the default from the curated preset list. `is_default`
        // in the Python registry marks exactly one entry as the
        // preferred first-time download (a ~2 GB Qwen 2.5 3B for
        // balanced quality/size on any modern Mac).
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
            // Bubble a stable sentinel so the caller can bail
            // silently without turning cancellation into a red error
            // banner.
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
            // Same Offline-Mode one-click recovery as
            // `handleTranscribe`. `MODEL_NETWORK_DISABLED` is the
            // single structured code the download command emits
            // when Offline Mode is on.
            if (
              isAppError(dlErr) &&
              dlErr.code === "MODEL_NETWORK_DISABLED"
            ) {
              const proceed = window.confirm(
                `To download the translation model "${preset.label}", the app needs one-time network access.\n\n` +
                  `Offline Mode is currently ON, which is blocking the download.\n\n` +
                  `Turn Offline Mode OFF and download now?\n` +
                  `(You can turn it back on in Settings once the download finishes — the model works fully offline afterwards.)`,
              );
              if (!proceed) return;
              await updateSettings({ offlineMode: false });
              await runDownload();
            } else {
              throw dlErr;
            }
          }
        } catch (e) {
          if ((e as { code?: string })?.code === "USER_CANCELLED") return;
          throw e;
        }

        // Refresh the model list, then wire the just-installed GGUF
        // into `translateOptions.model` so `translate.translate`
        // resolves it on the worker side. Falls back to the file
        // name in the preset if the registry rescan hasn't caught
        // up yet.
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
        await startTranslate(project.id, nextOpts);
        return;
      }
      await startTranslate(project.id, translateOptions);
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
      if (audio instanceof HTMLAudioElement) {
        audio.src = assetUrl(result.absolutePath);
        audio.currentTime = 0;
        void audio.play();
      }
    } catch (e) {
      setError(formatError(e));
    }
  };

  const handleTtsGenerateMissing = async () => {
    if (!project) return;
    setError(null);
    setBusy(true);
    try {
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
      await startGenerateTts(project.id, { kind: "all" });
    } catch (e) {
      setError(formatError(e));
    } finally {
      setBusy(false);
    }
  };

  const handleTtsRegenerate = async (segmentId: number) => {
    if (!project) return;
    setError(null);
    try {
      await startGenerateTts(project.id, {
        kind: "selected",
        ids: [segmentId],
      });
    } catch (e) {
      setError(formatError(e));
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
      if (audio instanceof HTMLAudioElement) {
        audio.src = assetUrl(result.absolutePath);
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
    if (audio instanceof HTMLAudioElement && lastMixPreview) {
      audio.src = assetUrl(lastMixPreview.absolutePath);
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

  const handleRenderPickOutput = async () => {
    const ext: OutputFormat = renderSettings.outputFormat;
    const defaultName = `movie_vi.${ext}`;
    const path = await pickRenderOutputPath(defaultName, ext);
    if (path) {
      setRenderSettings({ outputPath: path });
    }
  };

  const handleRenderClearOutput = () => {
    setRenderSettings({ outputPath: null });
  };

  return (
    <section className="project-view">
      <header className="project-header">
        <div>
          <Link to="/" className="back-link">← Dashboard</Link>
          <h1>{project.name}</h1>
        </div>
        <div className="meta">
          <span>{project.sourceLanguage} → {project.targetLanguage}</span>
          <span className={`status status--${project.status}`}>
            {project.status}
          </span>
        </div>
      </header>

      {error && (
        <div className="banner banner--error" role="alert">
          {error}
        </div>
      )}

      <div className="project-body">
        <Panel title="Source video">
          <SourcePanel
            project={project}
            metadata={media?.metadata ?? null}
            loading={mediaLoading}
            busy={busy}
            onImport={handleImport}
          />
        </Panel>

        {project.sourceMediaPath && (
          <Panel title="Preview">
            <VideoPreview
              absolutePath={project.sourceMediaPath}
              metadata={media?.metadata ?? null}
              videoRef={videoRef}
              onTimeUpdate={setVideoTime}
              overlayText={findCurrentSubtitleText(subtitleDoc, videoTime)}
            />
          </Panel>
        )}

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
            onRescanModels={() => void refreshTranslationModels()}
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

        <Panel title="Subtitles">
          <SubtitlePanel
            summary={media?.subtitles ?? null}
            doc={subtitleDoc}
            loading={subtitleLoading}
            hasTranscript={!!media?.transcript}
            currentTime={videoTime}
            onSeek={seekVideo}
            onRebuild={handleRebuildSubtitles}
            onPatch={handleSubtitlePatch}
            onAdd={handleSubtitleAdd}
            onDelete={handleSubtitleDelete}
            onSplit={handleSubtitleSplit}
            onMerge={handleSubtitleMerge}
            onImport={handleSubtitleImport}
            onExport={handleSubtitleExport}
            hasVideo={!!project.sourceMediaPath}
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

        <Panel title="TTS / Dubbing">
          <TtsPanel
            env={ttsEnv}
            voices={ttsVoices}
            manifest={ttsManifest}
            summary={media?.tts ?? null}
            subtitleDoc={subtitleDoc}
            engine={ttsEngine}
            voiceId={ttsVoiceId}
            settings={ttsSettings}
            onEngineChange={setTtsEngine}
            onVoiceChange={setTtsVoiceId}
            onSettingsChange={setTtsSettings}
            activeJob={activeTtsJob}
            progress={
              activeTtsJob ? jobProgress[activeTtsJob.id] ?? 0 : 0
            }
            busy={busy}
            onGenerateMissing={handleTtsGenerateMissing}
            onGenerateAll={handleTtsGenerateAll}
            onCancel={handleTtsCancel}
            onRescanVoices={() => void refreshTtsVoices()}
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
            progress={activeSyncJob ? jobProgress[activeSyncJob.id] ?? 0 : 0}
            busy={busy}
            onApplyMissing={handleSyncApplyMissing}
            onApplyAll={handleSyncApplyAll}
            onCancel={handleSyncCancel}
            lastPreview={lastSyncPreview}
          />
        </Panel>

        <Panel title="Audio mix">
          <MixPanel
            env={mixEnv}
            summary={media?.mix ?? null}
            syncSummary={media?.sync ?? null}
            settings={mixSettings}
            onSettingsChange={setMixSettings}
            activeJob={activeMixJob}
            progress={activeMixJob ? jobProgress[activeMixJob.id] ?? 0 : 0}
            busy={busy}
            onApply={handleMixApply}
            onCancel={handleMixCancel}
            onPlay={handleMixPlay}
            lastPreview={lastMixPreview}
          />
        </Panel>

        <Panel title="Final render">
          <RenderPanel
            env={renderEnv}
            summary={media?.render ?? null}
            mixSummary={media?.mix ?? null}
            subtitleSummary={media?.subtitles ?? null}
            settings={renderSettings}
            onSettingsChange={setRenderSettings}
            activeJob={activeRenderJob}
            progress={activeRenderJob ? jobProgress[activeRenderJob.id] ?? 0 : 0}
            busy={busy}
            onApply={() => handleRenderApply(false)}
            onRegenerate={() => handleRenderApply(true)}
            onCancel={handleRenderCancel}
            onPickOutputPath={handleRenderPickOutput}
            onClearOutputPath={handleRenderClearOutput}
          />
        </Panel>

        <Panel title="Pipeline">
          <ProgressRow
            label="Audio extraction"
            pct={
              activeExtractionJob
                ? jobProgress[activeExtractionJob.id] ?? 0
                : media?.audio
                  ? 1
                  : project.progress.audio ?? 0
            }
          />
          <ProgressRow
            label="Transcription"
            pct={
              activeTranscribeJob
                ? jobProgress[activeTranscribeJob.id] ?? 0
                : media?.transcript
                  ? 1
                  : project.progress.transcription ?? 0
            }
          />
          <ProgressRow
            label="Translation"
            pct={
              activeTranslateJob
                ? jobProgress[activeTranslateJob.id] ?? 0
                : media?.translation && media?.transcript
                  ? media.translation.translatedCount /
                    Math.max(1, media.translation.segmentCount)
                  : project.progress.translation ?? 0
            }
          />
          <ProgressRow
            label="TTS"
            pct={
              activeTtsJob
                ? jobProgress[activeTtsJob.id] ?? 0
                : media?.tts && media.tts.subtitleCount > 0
                  ? media.tts.generatedCount /
                    Math.max(1, media.tts.subtitleCount)
                  : project.progress.tts ?? 0
            }
          />
          <ProgressRow
            label="Voice sync"
            pct={
              activeSyncJob
                ? jobProgress[activeSyncJob.id] ?? 0
                : media?.sync && media.sync.subtitleCount > 0
                  ? media.sync.syncedCount /
                    Math.max(1, media.sync.subtitleCount)
                  : project.progress.sync ?? 0
            }
          />
          <ProgressRow
            label="Audio mix"
            pct={
              activeMixJob
                ? jobProgress[activeMixJob.id] ?? 0
                : media?.mix && media.mix.status === "ready"
                  ? 1
                  : project.progress.mix ?? 0
            }
          />
          <ProgressRow
            label="Rendering"
            pct={
              activeRenderJob
                ? jobProgress[activeRenderJob.id] ?? 0
                : media?.render && media.render.status === "ready"
                  ? 1
                  : project.progress.render ?? 0
            }
          />
        </Panel>

        <Panel title="Storage">
          <div className="mono small">{project.rootPath}</div>
        </Panel>
      </div>
    </section>
  );
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
  const src = assetUrl(absolutePath);
  const compat = detectPreviewCompatibility(absolutePath, metadata);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

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
        controls
        preload="metadata"
        src={src}
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
        }}
      >
        Your webview does not support the video element.
      </video>
      {runtimeError && (
        <div className="video-unsupported inline">
          <strong>Preview error:</strong> {runtimeError}
          <div className="small">
            Convert the file to .mp4 with H.264 + AAC to preview it in
            the app. Transcribe/translate/render still work on the
            original file.
          </div>
        </div>
      )}
      {overlayText && (
        <div className="video-subtitle-overlay">
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
  const stage = activeJob?.status === "running" || activeJob?.status === "queued"
    ? guessStage(progress)
    : null;

  return (
    <div className="stt-panel">
      {sttEnv && !sttEnv.whisperInstalled && (
        <div className="banner banner--warn">
          faster-whisper is not installed in the worker environment.
          Add it to <code>python/pyproject.toml</code> optional deps or run
          <code>pip install faster-whisper</code> in the worker venv.
        </div>
      )}

      <div className="stt-grid">
        <label className="field">
          <span>Model</span>
          <select
            value={options.model}
            disabled={!!activeJob || busy}
            onChange={(e) => onOptionsChange({ ...options, model: e.target.value })}
          >
            {models.length === 0 && <option value={options.model}>{options.model}</option>}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name} — {m.paramsM}M params
                {m.installed ? "" : "  (not installed)"}
              </option>
            ))}
          </select>
          {selectedModel && !selectedModel.installed && !downloadJob && (
            <button
              className="btn ghost small"
              onClick={() => onDownloadModel(selectedModel.name)}
              disabled={busy}
            >
              Download {selectedModel.name}
            </button>
          )}
        </label>

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
      </div>

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
            disabled={busy || !selectedModel}
            onClick={onTranscribe}
            title={
              selectedModel && !selectedModel.installed
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
  // Phase 12 UX — the app can auto-download a preset when nothing
  // is installed. Only fall back to the manual-install banner when
  // no presets are available (e.g. offline manifest fetch failed).
  const defaultPreset =
    recommendedPresets.find((p) => p.isDefault) ?? recommendedPresets[0] ?? null;
  const canAutoDownload = !!defaultPreset;
  const willAutoDownload =
    models.length === 0 && canAutoDownload && !!env?.llamaInstalled;

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

const SUBTITLE_ROW_HEIGHT = 176; // rough px height per row for virtualization
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
  const segments = doc?.segments ?? [];
  const filtered = filter
    ? segments.filter((s) =>
        `${s.sourceText}\n${s.translatedText}`
          .toLowerCase()
          .includes(filter.toLowerCase()),
      )
    : segments;

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
                ttsStatus={computeTtsStatus(
                  seg,
                  ttsManifest,
                  ttsEngine,
                  ttsVoiceId,
                  ttsSettings,
                )}
                onTtsPreview={onTtsPreview}
                onTtsRegenerate={onTtsRegenerate}
                syncStatus={computeSyncStatus(
                  seg,
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
    ttsStatus,
    onTtsPreview,
    onTtsRegenerate,
    syncStatus,
    onSyncPreview,
    onSyncRegenerate,
  } = props;
  const [source, setSource] = useState(seg.sourceText);
  const [translated, setTranslated] = useState(seg.translatedText);
  const [speaker, setSpeaker] = useState(seg.speaker ?? "");
  const [voice, setVoice] = useState(seg.voiceId ?? "");
  const [startStr, setStartStr] = useState(formatTimecode(seg.start));
  const [endStr, setEndStr] = useState(formatTimecode(seg.end));
  useEffect(() => setSource(seg.sourceText), [seg.sourceText]);
  useEffect(() => setTranslated(seg.translatedText), [seg.translatedText]);
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
  manifest: TtsManifest | null,
  engine: string,
  defaultVoiceId: string,
  settings: TtsSettings,
): TtsRowStatus {
  const text = (seg.translatedText || seg.sourceText || "").trim();
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
  const entry = manifest?.segments.find((s) => s.segmentId === seg.id);
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
  if (textOk && engineOk && voiceOk && speedOk && pitchOk && volumeOk) {
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
  manifest: SyncManifest | null,
  settings: SyncSettings,
): SyncRowStatus {
  const targetDuration = Math.max(0, seg.end - seg.start);
  const entry = manifest?.segments.find((s) => s.segmentId === seg.id);
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
  const canBurn = hasSubtitles;

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
            <option value="external">External (movie_vi.srt sidecar)</option>
            <option value="burned" disabled={!canBurn}>
              Burned into video
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

function TtsPanel(props: {
  env: TtsEnv | null;
  voices: VoiceInfo[];
  manifest: TtsManifest | null;
  summary: TtsSummary | null;
  subtitleDoc: SubtitleDoc | null;
  engine: string;
  voiceId: string;
  settings: TtsSettings;
  onEngineChange: (engine: string) => void;
  onVoiceChange: (voiceId: string) => void;
  onSettingsChange: (settings: Partial<TtsSettings>) => void;
  activeJob: JobSnapshot | null;
  progress: number;
  busy: boolean;
  onGenerateMissing: () => void;
  onGenerateAll: () => void;
  onCancel: () => void;
  onRescanVoices: () => void;
  lastPreview: PreviewResult | null;
}) {
  const {
    env,
    voices,
    summary,
    subtitleDoc,
    engine,
    voiceId,
    settings,
    onEngineChange,
    onVoiceChange,
    onSettingsChange,
    activeJob,
    progress,
    busy,
    onGenerateMissing,
    onGenerateAll,
    onCancel,
    onRescanVoices,
    lastPreview,
  } = props;

  const availableEngines = env?.engines ?? [];
  const currentEngineInfo = availableEngines.find((e) => e.id === engine);
  const supportedSettings = new Set(
    currentEngineInfo?.supportedSettings ?? ["speed"],
  );
  const enginesVoices = useMemo(
    () => voices.filter((v) => v.engine === engine),
    [voices, engine],
  );
  const hasSubtitles = (subtitleDoc?.segments.length ?? 0) > 0;
  const canGenerate =
    !!voiceId && !!currentEngineInfo?.available && hasSubtitles && !activeJob;

  return (
    <div className="tts-panel">
      <div className="tts-grid">
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
        <label>
          <span>Speed ({settings.speed.toFixed(2)}×)</span>
          <input
            type="range"
            min={0.5}
            max={2.0}
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

      <div className="tts-actions">
        <button
          className="btn"
          onClick={onGenerateMissing}
          disabled={!canGenerate || busy}
        >
          Generate missing
        </button>
        <button
          className="btn primary"
          onClick={onGenerateAll}
          disabled={!canGenerate || busy}
        >
          Generate all
        </button>
        {activeJob && (
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      {activeJob && (
        <div className="tts-progress">
          <ProgressRow label="Synthesising" pct={progress} />
        </div>
      )}

      {!hasSubtitles && (
        <div className="empty-state small">
          Build subtitles before generating voice.
        </div>
      )}
      {hasSubtitles && !voiceId && (
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

function findCurrentSubtitleText(
  doc: SubtitleDoc | null,
  time: number,
): string | null {
  if (!doc) return null;
  const seg = doc.segments.find((s) => time >= s.start && time < s.end);
  if (!seg) return null;
  return seg.translatedText || seg.sourceText || null;
}

function findActiveSubtitleId(
  doc: SubtitleDoc | null,
  time: number,
): number | null {
  if (!doc) return null;
  const seg = doc.segments.find((s) => time >= s.start && time < s.end);
  return seg?.id ?? null;
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

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${units[i]}`;
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
