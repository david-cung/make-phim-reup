// Typed contracts mirrored from Rust. Keep in sync whenever the Rust
// side changes. camelCase on the wire; snake_case is invisible.

export type Iso639 = string;

export interface AppInfo {
  appName: string;
  appVersion: string;
  dataDir: string;
  configDir: string;
  logDir: string;
  projectsDir: string;
  modelsDir: string;
  os: string;
  arch: string;
}

export interface AppSettings {
  offlineMode: boolean;
  sourceLanguage: Iso639;
  targetLanguage: Iso639;
  maxConcurrentJobs: number;
  logLevel: "trace" | "debug" | "info" | "warn" | "error";
  whisperModel: string | null;
  translationModel: string | null;
  ttsVoice: string | null;
  ffmpegPath: string | null;
  ffprobePath: string | null;
  /** Phase 10 — user-supplied override for the models directory. */
  modelsDirOverride: string | null;
  /** Phase 10 — set once the user has visited (or skipped) the
   *  first-run model setup. Controls the Dashboard banner. */
  firstRunCompleted: boolean;
  /** Phase 11 — how many seconds after a stage's last job settles
   *  before the worker unloads that stage's model. `null` or `0`
   *  disables auto-unload (models stay resident until the user
   *  closes the app or clicks "Unload all"). */
  autoUnloadAfterSecs: number | null;
  /** Phase 11 — CPU threads hint for inference engines. `null` lets
   *  the engine pick (typically number of physical cores). */
  cpuThreads: number | null;
  /** Phase 11 — allow GPU / Metal back-ends when the underlying
   *  engine supports them. Machines without GPU support silently
   *  fall back to CPU regardless of this flag. */
  gpuAcceleration: boolean;
}

// `undefined` = "don't change" ; `null` = "clear" ; string = "set".
export type AppSettingsPatch = {
  [K in keyof AppSettings]?: AppSettings[K] | null;
};

export type ProjectStatus =
  | "created"
  | "ready"
  | "processing"
  | "error"
  | "archived";

export type SourceImportMode = "reference" | "copy";

export interface ProjectSummary {
  id: string;
  name: string;
  sourceLanguage: Iso639;
  targetLanguage: Iso639;
  status: ProjectStatus;
  createdAt: string;
  updatedAt: string;
  lastOpenedAt: string | null;
}

export interface Project extends ProjectSummary {
  rootPath: string;
  sourceMediaPath: string | null;
  progress: Record<string, number>;
  sourceHash: string | null;
  sourceSize: number | null;
  sourceModifiedAt: string | null;
  sourceImportMode: SourceImportMode | null;
  /** Phase 10 — per-project model selection. Missing = fall back
   *  to the global default in `AppSettings`. */
  whisperModel: string | null;
  translationModel: string | null;
  ttsEngine: string | null;
  ttsVoiceId: string | null;
}

/** Phase 10 — patch shape for `update_project_models`. `undefined`
 *  = leave alone, `null` = clear, string = set. */
export interface ProjectModelPatch {
  whisperModel?: string | null;
  translationModel?: string | null;
  ttsEngine?: string | null;
  ttsVoiceId?: string | null;
}

export interface CreateProjectInput {
  name: string;
  sourceLanguage: Iso639;
  targetLanguage: Iso639;
}

export type WorkerState = "starting" | "running" | "stopped" | "crashed";

export interface WorkerStatus {
  state: WorkerState;
  pid: number | null;
  uptimeMs: number;
  lastError: string | null;
}

export interface PingResponse {
  pong: true;
  pid: number;
  uptimeMs: number;
}

export interface EnvInfo {
  python: string;
  platform: string;
  ffmpegAvailable: boolean;
  ffmpegVersion: string | null;
  cpuCount: number;
}

export interface AppError {
  code: string;
  stage: string | null;
  message: string;
  recoverable: boolean;
  hint: string | null;
}

export function isAppError(value: unknown): value is AppError {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    "message" in value &&
    "recoverable" in value
  );
}

// ---------- Phase 2 additions ----------

export interface FfmpegAvailability {
  available: boolean;
  ffmpegPath: string | null;
  ffprobePath: string | null;
  version: string | null;
  error: string | null;
}

export interface StreamSummary {
  index: number;
  kind: string;
  codec: string | null;
  language: string | null;
  channels: number | null;
  sampleRate: number | null;
}

export interface VideoMetadata {
  durationSecs: number;
  width: number | null;
  height: number | null;
  fps: number | null;
  videoCodec: string | null;
  audioCodec: string | null;
  audioChannels: number | null;
  audioSampleRate: number | null;
  format: string | null;
  fileSize: number;
  bitRate: number | null;
  audioStreamCount: number;
  subtitleStreamCount: number;
  videoStreamCount: number;
  streams: StreamSummary[];
}

export interface SourceFingerprint {
  hash: string;
  sizeBytes: number;
  modifiedAt: string;
}

export interface AudioExtractParams {
  sampleRate: number;
  channels: number;
  codec: string;
}

export interface AudioCacheEntry {
  source: SourceFingerprint;
  sourcePath: string;
  params: AudioExtractParams;
  outputRelative: string;
  outputSizeBytes: number;
  durationSecs: number;
  createdAt: string;
}

export interface ImportMediaInput {
  projectId: string;
  sourcePath: string;
  copyIntoProject?: boolean;
}

export interface ImportMediaResult {
  project: Project;
  fingerprint: SourceFingerprint;
  mode: SourceImportMode;
  sourceMediaPath: string;
}

export type JobStage =
  | "extract_audio"
  | "transcribe"
  | "translate"
  | "tts"
  | "sync"
  | "mix"
  | "render";

export type JobStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobSnapshot {
  id: string;
  projectId: string;
  stage: JobStage;
  status: JobStatus;
  progress: number;
  errorCode: string | null;
  errorMessage: string | null;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
}

export interface JobProgressEvent {
  id: string;
  projectId: string;
  stage: JobStage;
  progress: number;
}

export interface ProjectMediaState {
  metadata: VideoMetadata | null;
  audio: AudioCacheEntry | null;
  audioAbsolutePath: string | null;
  transcript: TranscriptSummary | null;
  translation: TranslationSummary | null;
  subtitles: SubtitleSummary | null;
  tts: TtsSummary | null;
  sync: SyncSummary | null;
  mix: MixSummary | null;
  render: RenderSummary | null;
  activeJobs: JobSnapshot[];
}

export type ExtractionStart =
  | {
      kind: "cacheHit";
      entry: AudioCacheEntry;
      absoluteOutput: string;
    }
  | ({ kind: "started" } & JobSnapshot);

// ---------- Phase 3 additions ----------

export interface SttOptions {
  model: string;
  language: string | null;
  device: string | null;
  computeType: string | null;
  beamSize: number;
  wordTimestamps: boolean;
  vadFilter: boolean;
  initialPrompt: string | null;
  temperature: number;
  qualityProfile?: "fast" | "balanced" | "quality" | string | null;
  resegment?: boolean;
}

export type QualityProfile = "fast" | "balanced" | "quality";

export const QUALITY_PROFILE_PRESETS: Record<
  QualityProfile,
  Pick<SttOptions, "model" | "beamSize" | "wordTimestamps" | "vadFilter" | "qualityProfile" | "resegment">
> = {
  fast: {
    model: "small",
    beamSize: 1,
    wordTimestamps: true,
    vadFilter: true,
    qualityProfile: "fast",
    resegment: true,
  },
  balanced: {
    model: "medium",
    beamSize: 5,
    wordTimestamps: true,
    vadFilter: true,
    qualityProfile: "balanced",
    resegment: true,
  },
  quality: {
    model: "large-v3",
    beamSize: 8,
    wordTimestamps: true,
    vadFilter: true,
    qualityProfile: "quality",
    resegment: true,
  },
};

export interface SttHardwareInfo {
  ramTotalGb?: number | null;
  ramAvailableGb?: number | null;
  os?: string;
  arch?: string;
}

export interface LargeV3Capability {
  canRun: boolean;
  model: string;
  device: string;
  computeType: string;
  fallbackModel: string;
  reason: string | null;
  warning: string | null;
  ramTotalGb?: number | null;
  ramAvailableGb?: number | null;
  vramGb?: number | null;
}

export interface SttDeviceInfo {
  kind: "cpu" | "cuda" | "metal" | string;
  label: string;
  supported: boolean;
  count: number;
  detail: string | null;
}

export interface SttEnv {
  devices: SttDeviceInfo[];
  defaultDevice: string;
  whisperInstalled: boolean;
  modelsRoot: string;
  hardware?: SttHardwareInfo | null;
  largeV3?: LargeV3Capability | null;
  profiles?: Record<string, unknown> | null;
}

export interface WhisperModelInfo {
  name: string;
  repo: string;
  paramsM: number;
  installed: boolean;
  sizeBytes: number | null;
  path: string | null;
}

export interface WhisperWord {
  word: string;
  start: number;
  end: number;
  probability?: number | null;
}

export interface TranscribeSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  speakerId?: string | null;
  speakerConfidence?: number | null;
  avgLogprob?: number | null;
  noSpeechProb?: number | null;
  words?: WhisperWord[] | null;
}

export interface TranscriptSummary {
  language: string;
  model: string;
  device: string;
  computeType: string;
  wordTimestamps: boolean;
  segmentCount: number;
  durationSecs: number;
  cacheKey: string;
  createdAt: string;
  audioHash: string;
  relativePath: string;
}

export type TranscribeStart =
  | {
      kind: "cacheHit";
      transcript: TranscriptSummary;
      absolutePath: string;
    }
  | ({ kind: "started" } & JobSnapshot);

// Convenience factory: what the UI ships when the user clicks Start.
export function defaultSttOptions(
  overrides: Partial<SttOptions> = {},
): SttOptions {
  return {
    model: "medium",
    language: null,
    device: null,
    computeType: null,
    beamSize: 5,
    wordTimestamps: true,
    vadFilter: true,
    initialPrompt: null,
    temperature: 0,
    qualityProfile: "balanced",
    resegment: true,
    ...overrides,
  };
}

// ---------- Phase 4 additions (Local LLM translation) ----------

export interface TranslateOptions {
  /** GGUF filename under <models>/translation/. */
  model: string;
  sourceLanguage: string;
  targetLanguage: string;
  promptVersion: string;
  chunkSize: number;
  contextBefore: number;
  contextAfter: number;
  retryContextBefore: number;
  retryContextAfter: number;
  maxTranslationRetries: number;
  lowConfidenceThreshold: number;
  temperature: number;
  topP: number;
  maxTokens: number;
}

export interface TranslationEnv {
  llamaInstalled: boolean;
  modelsRoot: string;
  translationRoot: string;
  defaultModel: string | null;
  promptVersions: string[];
}

export interface TranslationModelInfo {
  name: string;
  path: string;
  sizeBytes: number;
  isDefault: boolean;
}

// Phase 12 — one row in the app's curated auto-download list.
// See `translation/registry.py::_RECOMMENDED_MODELS` for the source
// of truth; the wire shape is fixed by
// `translation::models::RecommendedPreset` in the Rust host.
export interface TranslationRecommendedPreset {
  preset: string;
  repo: string;
  filename: string;
  approxSizeBytes: number;
  label: string;
  isDefault: boolean;
}

export interface TranslatedSegment {
  id: number;
  sourceText: string;
  translation: string;
  dubbing?: string;
  start: number;
  end: number;
  edited: boolean;
  translationMetadata?: Record<string, unknown>;
}

export type GenderPresentation = "male" | "female" | "unknown" | string;
export type CharacterAgeGroup =
  | "child"
  | "younger"
  | "adult"
  | "older"
  | "unknown"
  | string;

export interface CharacterProfile {
  id: string;
  displayName: string;
  speakerIds: string[];
  notes: string;
  genderPresentation: GenderPresentation;
  ageGroup: CharacterAgeGroup;
  defaultSelfReference: string;
  defaultNeutralAddress: string;
  userDefined: boolean;
}

export interface RelationshipRule {
  fromCharacterId: string;
  toCharacterId: string;
  relationshipType: string;
  selfReference: string;
  addressTerm: string;
  confidence: number;
  source: string;
  userDefined: boolean;
}

export interface SegmentPronounFlag {
  segmentId: number;
  flags: string[];
  speakerCharacterId?: string | null;
  addresseeCharacterIds: string[];
  ruleKey?: string | null;
  updatedAt: string;
}

export interface PronounContextDoc {
  version: number;
  characters: CharacterProfile[];
  relationships: RelationshipRule[];
  reviewFlags: SegmentPronounFlag[];
  updatedAt: string;
}

export interface TranslationSummary {
  sourceLanguage: string;
  targetLanguage: string;
  model: string;
  promptVersion: string;
  segmentCount: number;
  translatedCount: number;
  editedCount: number;
  cacheKey: string;
  transcriptCacheKey: string;
  createdAt: string;
  updatedAt: string;
  relativePath: string;
}

export interface TranslationDoc {
  version: number;
  sourceLanguage: string;
  targetLanguage: string;
  segments: TranslatedSegment[];
  model: string;
  promptVersion: string;
  cacheKey: string;
  transcriptCacheKey: string;
  audioHash: string;
  createdAt: string;
  updatedAt: string;
  provider: string;
  options: Record<string, unknown>;
}

export type TranslateStart =
  | {
      kind: "cacheHit";
      summary: TranslationSummary;
      absolutePath: string;
    }
  | ({ kind: "started" } & JobSnapshot);

export interface TranslationChunkCompletedEvent {
  jobId: string;
  projectId: string;
  translatedCount: number;
  segmentCount: number;
}

/** Common language codes used in the UI dropdowns.
 *  Ordered by likely usage for a Vietnamese audience: target
 *  language first, then East Asian source languages, then the
 *  rest alphabetised. Native-script hints in parentheses make
 *  the labels self-evident regardless of the reader's locale. */
export const TRANSLATION_LANGUAGES: { code: string; label: string }[] = [
  { code: "vi", label: "Vietnamese (Tiếng Việt)" },
  { code: "en", label: "English" },
  { code: "zh", label: "Chinese — Mandarin (中文)" },
  { code: "yue", label: "Chinese — Cantonese (粵語)" },
  { code: "ja", label: "Japanese (日本語)" },
  { code: "ko", label: "Korean (한국어)" },
  { code: "th", label: "Thai (ไทย)" },
  { code: "id", label: "Indonesian (Bahasa Indonesia)" },
  { code: "ms", label: "Malay (Bahasa Melayu)" },
  { code: "de", label: "German (Deutsch)" },
  { code: "es", label: "Spanish (Español)" },
  { code: "fr", label: "French (Français)" },
  { code: "it", label: "Italian (Italiano)" },
  { code: "pt", label: "Portuguese (Português)" },
  { code: "ru", label: "Russian (Русский)" },
];

export function defaultTranslateOptions(
  overrides: Partial<TranslateOptions> = {},
): TranslateOptions {
  return {
    model: "",
    sourceLanguage: "en",
    targetLanguage: "vi",
    promptVersion: "translation_prompt_v4",
    chunkSize: 15,
    contextBefore: 5,
    contextAfter: 5,
    retryContextBefore: 12,
    retryContextAfter: 12,
    maxTranslationRetries: 2,
    lowConfidenceThreshold: 0.8,
    temperature: 0.2,
    topP: 0.95,
    maxTokens: 2048,
    ...overrides,
  };
}

// ---------- Phase 5 additions (Subtitle editor) ----------

/**
 * The canonical subtitle model — this is what downstream stages
 * (Phase 6 TTS, Phase 8 mix, Phase 9 render) consume. It's derived
 * from the transcript + translation and then persisted as the
 * user edits it.
 */
export interface SubtitleWord {
  text: string;
  start: number;
  end: number;
}

export interface SubtitleSegment {
  id: number;
  start: number;
  end: number;
  sourceText: string;
  translatedText: string;
  dubbingText?: string;
  words?: SubtitleWord[] | null;
  speaker?: string | null;
  speakerConfidence?: number | null;
  voiceId?: string | null;
}

export interface DirtyFlags {
  tts: boolean;
  /** Phase 7 — timing-adjusted per-segment WAVs need regeneration. */
  sync: boolean;
  mix: boolean;
  render: boolean;
}

export interface DerivedFrom {
  transcriptCacheKey?: string | null;
  translationCacheKey?: string | null;
  origin: string;
}

export interface SubtitleDoc {
  version: number;
  sourceLanguage: string;
  targetLanguage: string;
  segments: SubtitleSegment[];
  derivedFrom: DerivedFrom;
  dirty: DirtyFlags;
  nextId: number;
  createdAt: string;
  updatedAt: string;
}

export interface SubtitleSummary {
  sourceLanguage: string;
  targetLanguage: string;
  segmentCount: number;
  translatedCount: number;
  speakerCount: number;
  overlapCount: number;
  dirty: DirtyFlags;
  derivedFrom: DerivedFrom;
  createdAt: string;
  updatedAt: string;
  relativePath: string;
}

/**
 * Wire form of the segment patch. Fields not present are left
 * untouched. `speaker` / `voiceId` use `null` to clear, `""` also
 * clears, otherwise the provided string overwrites.
 */
export interface SubtitleSegmentPatch {
  start?: number;
  end?: number;
  sourceText?: string;
  translatedText?: string;
  dubbingText?: string;
  speaker?: string | null;
  voiceId?: string | null;
}

export type SubtitleFormat = "srt" | "ass";
export type ExportKind = "translated" | "source" | "bilingual";

export interface ExportSubtitlesResult {
  path: string;
  format: SubtitleFormat;
  segmentCount: number;
  bytesWritten: number;
}

export interface ImportSubtitlesResult {
  doc: SubtitleDoc;
  format: SubtitleFormat;
  sourcePath: string;
  segmentCount: number;
}

// ---------- Phase 6 additions (Local TTS / AI dubbing) ----------

export interface TtsSettings {
  speed: number;
  pitch: number;
  volume: number;
  device: "auto" | "cpu" | "cuda" | "mps";
}

export function defaultTtsSettings(
  overrides: Partial<TtsSettings> = {},
): TtsSettings {
  return {
    speed: 1.0,
    pitch: 0.0,
    volume: 1.0,
    device: "auto",
    ...overrides,
  };
}

export interface TtsEngineInfo {
  id: string;
  name: string;
  available: boolean;
  supportedSettings: string[];
  license?: string | null;
  commercialUse?: boolean | null;
}

export interface F5TtsModelInfo {
  id: string;
  engine: string;
  name: string;
  installed: boolean;
  status: "ready" | "installed" | "not_installed" | "loading" | "error";
  path: string;
  source: string;
  version: string;
  license: string;
  commercialUse: boolean;
  approxSizeBytes: number;
}

export interface F5TtsHardwareInfo {
  backend: string;
  cudaAvailable: boolean;
  mpsAvailable: boolean;
  gpuName?: string | null;
  vramGb?: number | null;
  ramTotalGb?: number | null;
  ramAvailableGb?: number | null;
  os: string;
  recommended: boolean;
  warning?: string | null;
  runtimeError?: string | null;
}

export interface TtsEnv {
  engines: TtsEngineInfo[];
  modelsRoot: string;
  ttsRoot: string;
  piperInstalled: boolean;
  f5RuntimeInstalled: boolean;
  f5Model: F5TtsModelInfo;
  f5Hardware: F5TtsHardwareInfo;
  defaultEngine: string;
}

export interface VoiceInfo {
  id: string;
  name: string;
  language: string;
  gender: string;
  engine: string;
  modelPath: string;
  configPath?: string | null;
  sampleRate: number;
  installed: boolean;
  quality?: string | null;
  supportedSettings: string[];
  referenceAudioPath?: string | null;
  referenceText?: string | null;
  modelName?: string | null;
  modelVersion?: string | null;
  modelSource?: string | null;
  license?: string | null;
  commercialUse?: boolean | null;
  cacheIdentity?: string | null;
  emotion?: string | null;
  style?: string | null;
}

// Phase 12 — one entry from the app's curated Piper voice
// auto-download list. Source of truth is
// ``tts/registry.py::_RECOMMENDED_VOICES``; Rust mirrors this shape
// via ``tts::models::RecommendedVoicePreset``. ``targetLanguages``
// lets the frontend pick a voice matching the project's target
// translation language when auto-downloading.
export interface TtsRecommendedVoicePreset {
  preset: string;
  engine: string;
  voiceId: string;
  language: string;
  targetLanguages: string[];
  quality: string;
  approxSizeBytes: number;
  label: string;
  isDefault: boolean;
  license?: string | null;
  commercialUse?: boolean | null;
}

export interface CreateTtsVoiceProfileRequest {
  id: string;
  name: string;
  gender: "male" | "female" | "neutral" | "unknown";
  referenceAudioPath: string;
  referenceText: string;
  emotion?: string | null;
  style?: string | null;
}

export interface TtsSegmentEntry {
  segmentId: number;
  engine: string;
  voiceId: string;
  modelName: string;
  cacheKey: string;
  textHash: string;
  /**
   * Trimmed literal text fed to the engine. Duplicated from
   * `subtitles.json` at generation time so the UI can do cheap
   * string-equality staleness checks without a crypto hash.
   */
  text: string;
  speed: number;
  pitch: number;
  volume: number;
  device: "auto" | "cpu" | "cuda" | "mps";
  file: string;
  durationSecs: number;
  sampleRate: number;
  channels: number;
  sizeBytes: number;
  generatedAt: string;
}

export interface VoiceProfile {
  characterId: string;
  speakerId?: string | null;
  voiceId: string;
  style: string;
  speed: number;
  confidence: number;
}

export interface TtsManifest {
  version: number;
  engine: string;
  defaultVoiceId: string;
  voiceProfiles: VoiceProfile[];
  segments: TtsSegmentEntry[];
  createdAt: string;
  updatedAt: string;
}

export interface TtsSummary {
  engine: string;
  defaultVoiceId: string;
  subtitleCount: number;
  generatedCount: number;
  missingCount: number;
  staleCount: number;
  updatedAt: string;
  relativePath: string;
}

export type GenerateMode =
  | { kind: "missing" }
  | { kind: "all" }
  | { kind: "selected"; ids: number[] };

export interface GenerateRequest {
  engine: string;
  defaultVoiceId: string;
  settings: TtsSettings;
  mode: GenerateMode;
}

export type TtsGenerateStart =
  | { kind: "upToDate"; summary: TtsSummary }
  | ({ kind: "started" } & JobSnapshot);

export interface PreviewResult {
  segmentId: number;
  engine: string;
  voiceId: string;
  absolutePath: string;
  relativePath: string;
  durationSecs: number;
  cacheHit: boolean;
}

export interface TtsSegmentCompletedEvent {
  jobId: string;
  projectId: string;
  segmentId: number;
  file: string;
  generatedCount: number;
  subtitleCount: number;
}

export interface TtsProgressDetailEvent {
  jobId: string;
  projectId: string;
  stage: "preparing" | "loading_model" | "generating_voice" | "completed" | string;
  completedSegments: number;
  totalSegments: number;
  currentSegmentId?: number | null;
}

// ---------- Phase 7 additions (Voice synchronisation) ----------

export interface SyncSettings {
  /** Slowest atempo factor we allow when a voice is *shorter* than
   * its window — kept at 1.0 by default (no artificial slowdown). */
  minSpeed: number;
  /** Fastest atempo factor we allow when a voice overflows its
   * window. Anything requiring more than this is flagged
   * `tooLong`. */
  maxSpeed: number;
  /** `null` = keep the source TTS WAV's sample rate. */
  outputSampleRate: number | null;
  outputChannels: number;
}

export function defaultSyncSettings(
  overrides: Partial<SyncSettings> = {},
): SyncSettings {
  return {
    minSpeed: 0.9,
    maxSpeed: 1.12,
    outputSampleRate: null,
    outputChannels: 1,
    ...overrides,
  };
}

export interface SyncEnv {
  ffmpegAvailable: boolean;
  ffmpegPath: string | null;
  defaultMinSpeed: number;
  defaultMaxSpeed: number;
}

export type SyncStatus = "empty" | "fits" | "adjusted" | "too_long";

export interface SyncSegmentEntry {
  segmentId: number;
  status: SyncStatus;
  targetStart: number;
  targetEnd: number;
  targetDurationSecs: number;
  originalDurationSecs: number;
  finalDurationSecs: number;
  speedFactor: number;
  cacheKey: string;
  ttsCacheKey: string;
  /** Path relative to the project root, e.g. `voices/synced/000012.wav`. */
  file: string;
  sampleRate: number;
  channels: number;
  sizeBytes: number;
  generatedAt: string;
}

export interface SyncManifest {
  version: number;
  settings: SyncSettings;
  segments: SyncSegmentEntry[];
  createdAt: string;
  updatedAt: string;
}

export interface SyncSummary {
  settings: SyncSettings;
  subtitleCount: number;
  syncedCount: number;
  missingCount: number;
  staleCount: number;
  tooLongCount: number;
  adjustedCount: number;
  fitsCount: number;
  emptyCount: number;
  updatedAt: string;
  relativePath: string;
}

export type SyncMode =
  | { kind: "missing" }
  | { kind: "all" }
  | { kind: "selected"; ids: number[] };

export interface SyncRequest {
  settings: SyncSettings;
  mode: SyncMode;
}

export type SyncGenerateStart =
  | { kind: "upToDate"; summary: SyncSummary }
  | ({ kind: "started" } & JobSnapshot);

export interface PreviewSyncResult {
  segmentId: number;
  status: SyncStatus;
  targetDurationSecs: number;
  originalDurationSecs: number;
  finalDurationSecs: number;
  speedFactor: number;
  absolutePath: string;
  relativePath: string;
  cacheHit: boolean;
}

export interface SyncSegmentCompletedEvent {
  jobId: string;
  projectId: string;
  segmentId: number;
  file: string;
  status: SyncStatus;
  syncedCount: number;
  subtitleCount: number;
}

// ---------- Phase 8 additions (Audio mixing) ----------

export interface MixSettings {
  /** Linear gain for the original movie soundtrack (0–2). */
  originalVolume: number;
  /** Linear gain for the Vietnamese voice-over (0–2). */
  voiceVolume: number;
  /** Enable side-chain compression of the original by the voice. */
  duckingEnabled: boolean;
  /** How much lower to push the original when voice speaks (dB, 0–30). */
  duckingDepthDb: number;
  /** Compressor threshold in dB (–60..0). */
  duckingThresholdDb: number;
  /** Attack time in ms (1–500). */
  duckingAttackMs: number;
  /** Release time in ms (10–5000). */
  duckingReleaseMs: number;
  /** `null` = keep source video's rate. */
  outputSampleRate: number | null;
  /** 1 = mono, 2 = stereo. */
  outputChannels: number;
}

export function defaultMixSettings(
  overrides: Partial<MixSettings> = {},
): MixSettings {
  return {
    originalVolume: 0.22,
    voiceVolume: 1.05,
    duckingEnabled: true,
    duckingDepthDb: 22.0,
    duckingThresholdDb: -22.0,
    duckingAttackMs: 12.0,
    duckingReleaseMs: 250.0,
    outputSampleRate: null,
    outputChannels: 2,
    ...overrides,
  };
}

export interface MixEnv {
  ffmpegAvailable: boolean;
  ffmpegPath: string | null;
  defaultSettings: MixSettings;
}

export type MixStatus = "missing" | "stale" | "ready";

export interface MixEntry {
  cacheKey: string;
  sourceFingerprint: SourceFingerprint;
  /** Path relative to the project root, e.g. `audio/mixed_vi.wav`. */
  file: string;
  durationSecs: number;
  sampleRate: number;
  channels: number;
  sizeBytes: number;
  voiceSegmentCount: number;
  subtitleCount: number;
  settings: MixSettings;
  generatedAt: string;
}

export interface MixManifest {
  version: number;
  settings: MixSettings;
  current: MixEntry | null;
  createdAt: string;
  updatedAt: string;
}

export interface MixSummary {
  status: MixStatus;
  settings: MixSettings;
  durationSecs: number | null;
  absolutePath: string | null;
  relativePath: string | null;
  voiceSegmentCount: number;
  subtitleCount: number;
  sizeBytes: number | null;
  generatedAt: string | null;
  needsGenerate: boolean;
  warning: string | null;
  manifestRelativePath: string;
}

export type MixMode = { kind: "all" };

export interface MixRequest {
  settings: MixSettings;
  mode: MixMode;
}

export type MixGenerateStart =
  | { kind: "upToDate"; summary: MixSummary }
  | ({ kind: "started" } & JobSnapshot);

export interface PreviewMixResult {
  absolutePath: string;
  relativePath: string;
  durationSecs: number;
  sampleRate: number;
  channels: number;
  cacheHit: boolean;
}

// ---------- Phase 9 additions (Final video rendering) ----------

export type SubtitleMode = "none" | "external" | "burned";
export type OutputFormat = "mp4" | "mkv";
export type RenderStatus = "missing" | "stale" | "ready";

export type VideoCodec =
  | { kind: "copy" }
  | { kind: "reencode"; codec: string };

export interface AudioCodec {
  codec: string;
  bitrate: string | null;
}

export interface RenderSettings {
  outputFormat: OutputFormat;
  videoCodec: VideoCodec;
  audioCodec: AudioCodec;
  subtitleMode: SubtitleMode;
  /** Absolute path where the final movie should be written. `null` =
   *  default under `<project>/output/movie_vi.<ext>`. */
  outputPath: string | null;
}

export function defaultRenderSettings(
  overrides: Partial<RenderSettings> = {},
): RenderSettings {
  return {
    outputFormat: "mp4",
    videoCodec: { kind: "copy" },
    audioCodec: { codec: "aac", bitrate: "192k" },
    subtitleMode: "burned",
    outputPath: null,
    ...overrides,
  };
}

export interface RenderEnv {
  ffmpegAvailable: boolean;
  ffmpegPath: string | null;
  defaultSettings: RenderSettings;
  videoCodecs: string[];
  audioCodecs: string[];
  outputFormats: string[];
  /** Whether this FFmpeg can burn subtitles — it needs libass, which many
   *  builds omit. False means `subtitleMode: "burned"` cannot be used. */
  subtitleBurnAvailable: boolean;
}

export interface RenderEntry {
  cacheKey: string;
  sourceFingerprint: SourceFingerprint;
  mixCacheKey: string;
  fileAbsolute: string;
  fileRelative: string | null;
  subtitleFileAbsolute: string | null;
  durationSecs: number;
  sizeBytes: number;
  videoStreamCount: number;
  audioStreamCount: number;
  subtitleStreamCount: number;
  subtitleMode: SubtitleMode;
  settings: RenderSettings;
  generatedAt: string;
}

export interface RenderManifest {
  version: number;
  settings: RenderSettings;
  current: RenderEntry | null;
  createdAt: string;
  updatedAt: string;
}

export interface RenderSummary {
  status: RenderStatus;
  settings: RenderSettings;
  durationSecs: number | null;
  absolutePath: string | null;
  relativePath: string | null;
  subtitleAbsolutePath: string | null;
  sizeBytes: number | null;
  generatedAt: string | null;
  videoStreamCount: number;
  audioStreamCount: number;
  subtitleStreamCount: number;
  subtitleMode: SubtitleMode;
  needsRender: boolean;
  defaultOutputAbsolute: string;
  warning: string | null;
  manifestRelativePath: string;
}

export interface RenderRequest {
  settings: RenderSettings;
  /** Force a fresh render even when the cache says the output is up
   *  to date. Used by the "Regenerate" button. */
  force: boolean;
}

export type RenderGenerateStart =
  | { kind: "upToDate"; summary: RenderSummary }
  | ({ kind: "started" } & JobSnapshot);

// ---------- Phase 10 additions (Local Model Manager) ----------

export type ModelKind = "whisper" | "translation" | "tts" | "voice";
export type ModelStatus = "available" | "missing" | "invalid";
export type ImportStrategy = "link" | "copy";

/** Every locally installed (or known-but-missing) AI model, in one
 *  flat list. The registry never stores model binaries — just
 *  metadata + paths — so this is safe to fetch repeatedly. */
export interface LocalModel {
  id: string;
  name: string;
  kind: ModelKind;
  engine?: string | null;
  language?: string | null;
  path?: string | null;
  sizeBytes?: number | null;
  version?: string | null;
  status: ModelStatus;
  hint?: string | null;
}

export interface ModelDirectoryInfo {
  path: string;
  isDefault: boolean;
  defaultPath: string;
  whisperSubdir: string;
  translationSubdir: string;
  ttsSubdir: string;
  exists: boolean;
}

export interface ImportModelSpec {
  kind: ModelKind;
  sourcePath: string;
  name?: string | null;
  engine?: string | null;
  strategy?: ImportStrategy;
}

// ---------- Phase 11 additions (Runtime stats + perf settings) ----------

/** Lightweight runtime snapshot exposed by `get_runtime_stats`. Safe
 *  to poll every couple of seconds — every field is O(1) on the
 *  Rust side and there's no worker RPC involved. */
export interface RuntimeStats {
  activeJobs: number;
  activeProjects: number;
  hostRssBytes: number | null;
  workerRssBytes: number | null;
  workerUptimeSecs: number | null;
}

// ---------- Phase 12 additions (Storage / crash recovery) ----------

/** Named directory the frontend may ask Rust to reveal or size.
 *  Restricted to app-owned roots — the frontend can never pass an
 *  arbitrary filesystem path. */
export type AppPathKind =
  | "data"
  | "config"
  | "log"
  | "cache"
  | "projects"
  | "models";

/** Snapshot of on-disk sizes returned by `get_storage_stats`.
 *  Computed with a bounded recursive walk so a pathological
 *  directory can't hang the Settings › Storage panel. */
export interface StorageStats {
  dataDir: string;
  dataBytes: number;
  cacheDir: string;
  cacheBytes: number;
  logDir: string;
  logBytes: number;
  projectsDir: string;
  projectsBytes: number;
  modelsDir: string;
  modelsBytes: number;
}

// ---------- Phase 13 (optional YouTube integration) ----------

export interface YouTubeAccount {
  id: string;
  channelId: string | null;
  channelTitle: string | null;
  thumbnailUrl: string | null;
  connectedAt: string;
}

export type YouTubeAccountStatus =
  | "connected"
  | "disconnected"
  | "expired"
  | "authenticationRequired";

export interface YouTubeConnectionState {
  configured: boolean;
  status: YouTubeAccountStatus;
  account: YouTubeAccount | null;
  offline: boolean;
}

export type YouTubePrivacyStatus = "private" | "unlisted" | "public";

export interface YouTubeVideoMetadata {
  title: string;
  description: string;
  tags: string[];
  privacyStatus: YouTubePrivacyStatus;
  categoryId: string;
  defaultLanguage?: string | null;
}

export interface YouTubePublishOptions {
  playlistId?: string | null;
  thumbnailPath?: string | null;
  publishTranslatedSubtitles: boolean;
  publishOriginalSubtitles: boolean;
}

export interface YouTubePlaylist {
  id: string;
  name: string;
}

export type YouTubeAssetStepState = "pending" | "completed" | "failed";

export interface YouTubeAssetStep {
  kind:
    | "playlist"
    | "thumbnail"
    | "translatedSubtitles"
    | "originalSubtitles"
    | "status"
    | "history";
  state: YouTubeAssetStepState;
  errorCode: string | null;
  errorMessage: string | null;
}

export interface YouTubePublishingHistoryEntry {
  videoId: string;
  title: string;
  privacyStatus: YouTubePrivacyStatus;
  uploadedAt: string;
  channelId: string;
  url: string;
}

export interface YouTubeThumbnailResult {
  path: string;
  timeSeconds: number;
}

export type YouTubeUploadState =
  | "idle"
  | "waiting"
  | "connecting"
  | "preparing"
  | "uploading"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface YouTubeUploadSnapshot {
  id: string;
  projectId: string;
  createdAt: string;
  state: YouTubeUploadState;
  filePath: string;
  bytesUploaded: number;
  totalBytes: number;
  progress: number;
  videoId: string | null;
  url: string | null;
  errorCode: string | null;
  errorMessage: string | null;
  canRetry: boolean;
  title: string;
  privacyStatus: YouTubePrivacyStatus;
  assetSteps: YouTubeAssetStep[];
}

export interface YouTubeUploadProgressEvent {
  upload: YouTubeUploadSnapshot;
}
