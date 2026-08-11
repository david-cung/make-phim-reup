# Architecture — Local Movie Translator

Version: 0.5 (Phase 5 — Subtitle System & Editor)
Status: **Design frozen for Phases 1–5**; later phases will extend, not rewrite.

---

## 1. Product summary

Local Movie Translator is an **offline-first desktop application** that turns a
foreign-language movie into a movie with Vietnamese subtitles and Vietnamese
voice-over, while preserving the original video stream and, as much as
possible, the original music/SFX.

The pipeline is:

```
Movie
  → speech-to-text (Whisper)
  → context-aware LLM translation
  → editable subtitle track (SRT / ASS)
  → local Vietnamese TTS
  → per-segment audio timing/stretching
  → mix with original audio
  → mux into final video
```

Every step is **file-based, resumable, cancellable, cacheable** and never
requires network access after models are installed.

---

## 2. High-level components

```
┌────────────────────────────────────────────────────────────────┐
│                       Local Movie Translator                    │
│                                                                 │
│  React + TypeScript (Vite)   ── UI only, no heavy compute       │
│               │                                                 │
│               │  Tauri IPC (typed commands + events)            │
│               ▼                                                 │
│  Rust (Tauri 2)              ── shell, orchestration, safety    │
│               │                                                 │
│               │  JSON-RPC 2.0 over stdio (localhost only)       │
│               ▼                                                 │
│  Python worker               ── AI + media processing           │
│               │                                                 │
│               │  subprocess / library calls                     │
│               ▼                                                 │
│  Whisper · llama.cpp · TTS · FFmpeg   ── heavy lifting          │
│                                                                 │
│  SQLite (project/job/subtitle state)                            │
│  Filesystem (source video, caches, models, exports)             │
└────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer   | Owns                                                                                                               | Never does                                                       |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| React   | rendering, form state, virtualized subtitle list, project navigation                                               | AI inference, file IO, spawning processes                        |
| Rust    | app lifecycle, filesystem, SQLite, job queue, IPC contracts, path validation, process management (Python, FFmpeg) | Whisper/LLM/TTS inference, subtitle logic                        |
| Python  | Whisper, LLM translation, TTS, subtitle math, audio helpers, media probing                                        | UI, window management, being a public HTTP server                |
| FFmpeg  | codecs, extraction, muxing, time-stretching, subtitle burn-in                                                     | anything else                                                    |
| SQLite  | small, structured, indexed metadata (projects, jobs, subtitle rows, settings)                                     | store binary media, model files, or logs                         |

---

## 3. Provider abstractions

All AI capabilities live behind Python interfaces so we can swap engines
without touching the pipeline or the UI.

```python
class SpeechToTextProvider(Protocol):
    def transcribe(self, audio_path: Path, opts: STTOptions) -> Transcript: ...

class TranslationProvider(Protocol):
    def translate(self, segments: list[Segment], ctx: TranslationContext) -> list[Segment]: ...

class TTSProvider(Protocol):
    def list_voices(self) -> list[Voice]: ...
    def synthesize(self, text: str, voice_id: str, opts: TTSOptions) -> Path: ...
```

Initial implementations, chosen but not baked in:

* `SpeechToTextProvider` → `FasterWhisperProvider` (`faster-whisper` / CTranslate2)
* `TranslationProvider`  → `LlamaCppProvider` (llama.cpp with a Vietnamese-capable GGUF)
* `TTSProvider`          → `PiperProvider` (Piper Vietnamese voices) — replaceable

`Ollama` is only allowed as a developer convenience and must never become a
production runtime dependency.

---

## 4. Data model

### 4.1 On disk (per project)

```
<projects_root>/<project-uuid>/
├── project.json          # canonical project metadata (mirror of SQLite row)
├── source/               # original media — NEVER modified
│   └── movie.mkv         # actually a hard-link or symlink where possible
├── audio/
│   ├── original.wav      # extracted mono/stereo pcm for STT
│   └── original.opus     # compressed copy for playback
├── transcription/
│   └── transcription.json
├── translation/
│   └── translation.json
├── subtitles/
│   ├── subtitles.json
│   ├── vi.srt
│   └── vi.ass
├── voices/
│   ├── voices.json      # per-segment TTS manifest (Phase 6)
│   ├── 000001.wav       # WAV per subtitle segment
│   └── 000002.wav
├── cache/
│   └── fingerprints.json # dependency-aware invalidation
├── output/
│   └── movie.vi.mkv
└── logs/
    └── project.jsonl
```

The **source movie is never mutated**. Any transformation writes into
`audio/`, `voices/`, `output/`, etc.

### 4.2 SQLite (`<app_data>/db.sqlite3`)

Only small, indexed metadata. Media, models and logs live on the filesystem.

```
projects(
  id TEXT PRIMARY KEY,           -- uuid v4
  name TEXT NOT NULL,
  source_language TEXT NOT NULL, -- ISO-639-1
  target_language TEXT NOT NULL,
  root_path TEXT NOT NULL,       -- absolute, canonical, inside allowed root
  source_media_path TEXT,        -- null until video is imported (Phase 2)
  status TEXT NOT NULL,          -- created | ready | processing | error | archived
  progress_json TEXT NOT NULL,   -- {"transcription":1.0, "translation":0.5,...}
  created_at TEXT NOT NULL,      -- RFC3339
  updated_at TEXT NOT NULL,
  last_opened_at TEXT
)

projects(                          -- (Phase 2 additions)
  ...,
  source_hash        TEXT,         -- sha256:<hex> partial-content fingerprint
  source_size        INTEGER,      -- bytes at import time
  source_modified_at TEXT,         -- source file mtime at import time
  source_import_mode TEXT          -- 'reference' | 'copy'
)

jobs(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,           -- extract_audio | transcribe | translate | tts | mix | render
  status TEXT NOT NULL,          -- queued | running | paused | completed | failed | cancelled
  progress REAL NOT NULL DEFAULT 0,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT
)

subtitle_segments(
  id INTEGER PRIMARY KEY,        -- stable segment id from Whisper
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  source_text TEXT NOT NULL,
  translated_text TEXT,
  speaker_id TEXT,
  voice_id TEXT,
  status TEXT NOT NULL           -- pending | translated | voice_generated | edited
)

speakers(id TEXT PRIMARY KEY, project_id TEXT, display_name TEXT, default_voice_id TEXT)
voices(id TEXT PRIMARY KEY, provider TEXT, locale TEXT, gender TEXT, path TEXT)
models(id TEXT PRIMARY KEY, kind TEXT, name TEXT, path TEXT, size_bytes INTEGER, installed_at TEXT)
settings(key TEXT PRIMARY KEY, value_json TEXT NOT NULL)
schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)
```

Indexes on `(project_id, id)` for `subtitle_segments`, `(project_id, status)`
for `jobs`, and `updated_at DESC` for the dashboard.

Phase 1 creates every table but only reads/writes `projects`, `settings` and
`schema_migrations`. Phase 2 adds a migration for the `source_*` columns on
`projects` (recording how the source movie was imported and its content
fingerprint) and starts using the `jobs` table for `extract_audio` runs.
Phase 3 reuses the `jobs` table for `transcribe` runs and persists
transcripts to `<project>/transcription/transcription.json` (schema in
§4.4); no new SQLite columns are required.

### 4.3 Per-project audio cache

The extractor writes a small manifest at
`<project>/cache/audio_cache.json`:

```jsonc
{
  "version": 1,
  "originalWav": {
    "source": {
      "hash": "sha256:...",
      "sizeBytes": 12345678,
      "modifiedAt": "2026-08-11T..."
    },
    "sourcePath": "/absolute/path/to/movie.mkv",
    "params": { "sampleRate": 16000, "channels": 1, "codec": "pcm_s16le" },
    "outputRelative": "audio/original.wav",
    "outputSizeBytes": 3840000,
    "durationSecs": 120.0,
    "createdAt": "2026-08-11T..."
  }
}
```

A cache **hit** requires all three: matching `(hash, sizeBytes)`, matching
`params`, and the referenced output file still on disk with the recorded
size. Anything else invalidates the cache and re-extracts.

### 4.4 Transcript file (Phase 3)

Path: `<project>/transcription/transcription.json`. This file is BOTH the
persisted deliverable AND the cache manifest — the extra metadata makes
cache invalidation obvious.

```jsonc
{
  "version": 1,
  "language": "en",                    // detected or user-forced
  "segments": [
    {
      "id": 0,
      "start": 12.31,                  // seconds
      "end": 14.82,
      "text": "Where are you going?",
      "avgLogprob": -0.12,             // optional
      "noSpeechProb": 0.01,            // optional
      "words": [                        // present iff options.wordTimestamps
        {"word":"Where","start":12.31,"end":12.72,"probability":0.98},
        ...
      ]
    }
  ],
  "model": "small",
  "device": "cpu",
  "computeType": "int8",
  "wordTimestamps": false,
  "audio": { "path": "audio/original.wav", "hash": "sha256:..." },
  "durationSecs": 3595.7,
  "cacheKey": "sha256:<hex>",          // audio hash + all cache-affecting options
  "createdAt": "2026-08-11T...",
  "provider": "faster-whisper",
  "options": { ... echo of the wire options ... }
}
```

The **cache key** is `sha256("v1" ‖ audioHash ‖ model ‖ language ‖ device ‖
computeType ‖ beamSize ‖ wordTimestamps ‖ vadFilter ‖ temperature ‖
initialPrompt)`. A cache hit requires `cacheKey` AND `audio.hash` to
match — the double check makes tampering obvious without a full
re-compute.

Segments produced by Whisper are the **canonical subtitle timings** for
the rest of the pipeline (Phase 4+). We never invent our own timings.

### 4.5 Translation file (Phase 4)

Path: `<project>/translation/translation.json`. This file is BOTH the
persisted deliverable AND the cache manifest. It is written
**incrementally** — every completed chunk lands atomically on disk so
a crash mid-translation leaves the completed portion intact and
"resume" is free.

```jsonc
{
  "version": 1,
  "sourceLanguage": "en",
  "targetLanguage": "vi",
  "segments": [
    {
      "id": 0,
      "sourceText": "Where are you going?",
      "translation": "Anh đang đi đâu vậy?",
      "start": 12.31,
      "end": 14.82,
      "edited": false                     // true iff a human edited it after the LLM ran
    }
  ],
  "model": "qwen2-7b-instruct-q4_k_m.gguf",
  "promptVersion": "translation_prompt_v1",
  "cacheKey": "sha256:<hex>",
  "transcriptCacheKey": "sha256:<hex>",   // must match the transcript's cache_key
  "audioHash": "sha256:<hex>",
  "createdAt": "2026-08-11T...",
  "updatedAt": "2026-08-11T...",
  "provider": "llama.cpp",
  "options": { ... echo of the wire options ... }
}
```

The **cache key** is
`sha256("translation_v1" ‖ transcriptCacheKey ‖ audioHash ‖
sourceLanguage ‖ targetLanguage ‖ model ‖ promptVersion ‖ chunkSize ‖
contextBefore ‖ contextAfter ‖ temperature ‖ topP ‖ maxTokens)`.

Cache-lookup semantics:

* **Complete hit** — cacheKey + transcriptCacheKey match AND every
  segment has a non-empty `translation` → return synchronously; no
  job spawned.
* **Resumable** — cacheKey + transcriptCacheKey match but at least
  one segment is empty → the service seeds the run with the existing
  segments and only asks the LLM to fill in the missing ids.
* **Miss** — either key doesn't match → seed a fresh doc, preserving
  any segments the user manually edited (`edited: true`) so a change
  of model/options never destroys human work.

Timings are duplicated from the transcript into every segment so the
file remains self-describing on disk (useful for debugging), but the
transcript is still the authoritative source. If the two disagree,
the transcript wins.

### 4.6 Subtitle file (Phase 5)

Path: `<project>/subtitles/subtitles.json`. This is the **canonical
subtitle model** consumed by every downstream stage from Phase 6
onwards. It is derived from the transcript (timing + source) and,
if present, the translation, then persisted as the user edits it.

```jsonc
{
  "version": 1,
  "sourceLanguage": "en",
  "targetLanguage": "vi",
  "segments": [
    {
      "id": 0,
      "start": 12.31,
      "end": 14.82,
      "sourceText": "Where are you going?",
      "translatedText": "Anh đang đi đâu vậy?",
      "speaker": "Alice",           // optional
      "voiceId": "piper-vi-1"       // optional
    }
  ],
  "derivedFrom": {
    "transcriptCacheKey": "sha256:…",  // may be null after SRT/ASS import
    "translationCacheKey": "sha256:…", // may be null if no translation
    "origin": "transcript+translation" // or "srt-import" / "ass-import" / "manual"
  },
  "dirty": { "tts": false, "mix": false, "render": false },
  "nextId": 128,                    // monotonic id allocator for splits/adds
  "createdAt": "2026-08-11T…",
  "updatedAt": "2026-08-11T…"
}
```

Invariants:

* **Stable ids.** `id` is preserved across edits. Split preserves the
  left half's id and allocates a fresh id (`nextId++`) for the right;
  merge keeps the leftmost id; delete never renumbers survivors.
  Downstream stages can safely key TTS/mix results by id.
* **Timing from Whisper.** On the first derivation, timing comes from
  the transcript segments (which come from Whisper — see §4.4).
  Manual edits are allowed but never accepted with `end <= start`;
  overlaps are permitted (dialogue can overlap in real movies) and
  surfaced as a warning count in the summary.
* **Preserve human edits on rebuild.** Clicking "Rebuild from
  transcript" merges: for any id that survives in the new transcript,
  the previous `translatedText` (if non-empty and different from what
  the fresh derivation produced), `speaker`, and `voiceId` are kept.
* **Dirty tracking.** Edits to `translatedText`, `start`, `end`,
  `speaker`, or `voiceId`, plus add / delete / split / merge, set
  `dirty.tts = dirty.mix = dirty.render = true`. Editing `sourceText`
  alone does NOT set dirty flags — nothing downstream reads it.
  Phase 6+ services call `clear_subtitle_dirty` after a successful
  pass.
* **Import.** SRT / ASS imports run through
  `subtitles::{srt, ass}::parse`. When a transcript exists we merge
  the imported translations onto matching-timeframe transcript rows
  so `sourceText` stays authoritative; when no transcript exists the
  imported text lands in both `sourceText` and `translatedText` so
  editing still works.
* **Export.** SRT / ASS export always reads `subtitles.json` — the
  canonical model — and never re-derives from transcript+translation
  at export time. Users pick `translated` / `source` / `bilingual` as
  the text field. Writes are atomic (tmp + rename).
* **No Python worker.** The whole subtitle pipeline is pure Rust —
  no AI, no external tools, no IPC to the worker.

---

## 5. IPC contracts

### 5.1 React ↔ Rust (Tauri commands)

Every command returns `Result<T, AppError>` and every payload is fully typed.
Phase 1 commands (unchanged):

```
get_app_info()                                              -> AppInfo
get_settings()                                              -> AppSettings
update_settings(patch)                                      -> AppSettings

list_projects()                                             -> Vec<ProjectSummary>
create_project({ name, source_language, target_language })  -> ProjectSummary
open_project(id)                                            -> Project
delete_project(id)                                          -> ()

worker_status()                                             -> WorkerStatus
worker_ping()                                               -> PingResponse
worker_env_info()                                           -> EnvInfo
```

Phase 2 additions:

```
get_ffmpeg_availability()                                   -> FfmpegAvailability
refresh_ffmpeg()                                            -> FfmpegAvailability
probe_media(path)                                           -> VideoMetadata
import_media({ projectId, sourcePath, copyIntoProject })    -> ImportMediaResult
get_project_media(projectId)                                -> ProjectMediaState
extract_audio(projectId)                                    -> ExtractionStart
cancel_job(jobId)                                           -> ()
list_active_jobs(projectId)                                 -> Vec<JobSnapshot>
```

Phase 3 additions:

```
get_stt_env()                                               -> SttEnv
list_whisper_models()                                       -> Vec<WhisperModelInfo>
download_whisper_model(name)                                -> JobSnapshot
transcribe(projectId, options?)                             -> TranscribeStart
get_project_transcript(projectId)                           -> Option<TranscriptSummary>
```

`ProjectMediaState` is extended to include the current `transcript`
summary (segment count, language, model, cacheKey, created_at) so the
Project screen has one source of truth for the whole media panel.
Transcription reuses `cancel_job` from Phase 2.

Phase 4 additions:

```
get_translation_env()                                       -> TranslationEnv
list_translation_models()                                   -> Vec<TranslationModelInfo>
translate(projectId, options)                               -> TranslateStart
get_project_translation(projectId)                          -> Option<TranslationSummary>
get_project_translation_doc(projectId)                      -> Option<TranslationDoc>
update_translation_segment(projectId, segmentId, translation) -> TranslationSummary
```

`ProjectMediaState` additionally carries a `translation` summary
(segment count, translated/edited counts, model, prompt version,
cacheKey, updated_at) so the project panel shows resume state at a
glance. Translation reuses `cancel_job`. The editor writes each
manual edit back via `update_translation_segment`, which flips the
segment's `edited` flag so subsequent LLM runs won't clobber it.

Phase 5 additions (subtitle system + editor):

```
get_project_subtitles(projectId)                            -> Option<SubtitleSummary>
get_project_subtitles_doc(projectId)                        -> Option<SubtitleDoc>
rebuild_project_subtitles(projectId)                        -> SubtitleDoc
update_subtitle_segment(projectId, segmentId, patch)        -> SubtitleDoc
add_subtitle_segment(projectId, afterId?, start, end)       -> SubtitleDoc
delete_subtitle_segment(projectId, segmentId)               -> SubtitleDoc
split_subtitle_segment(projectId, segmentId, splitTime)     -> SubtitleDoc
merge_subtitle_segment(projectId, segmentId)                -> SubtitleDoc
clear_subtitle_dirty(projectId)                             -> SubtitleDoc
import_subtitles(projectId, path)                           -> ImportSubtitlesResult
export_subtitles(projectId, path, format, kind?)            -> ExportSubtitlesResult
```

`ProjectMediaState` gains a `subtitles` summary (segment count,
translated count, speaker count, overlap count, dirty flags,
origin, updated_at) so the project panel shows subtitle status at
a glance. Every mutating command returns the *full* updated
`SubtitleDoc` — the frontend replaces its in-memory copy wholesale
so the UI never drifts from disk. All Phase-5 IPC is synchronous
(no jobs, no worker) because subtitle math is CPU-cheap.

The native file picker is exposed through `@tauri-apps/plugin-dialog`;
media source paths are converted for the `<video>` element with the
built-in `asset:` protocol (see §7). Nothing binary ever crosses IPC.

Events (Rust → React) via Tauri `emit`:

```
worker://status                 { state: "starting"|"running"|"stopped"|"crashed", pid?, uptimeMs }
job://progress                  { id, projectId, stage, progress }             # 0..1, throttled to ~10 Hz
job://update                    { id, projectId, stage, status, error?, ... }  # queued/running/terminal
translation://chunk_completed   { jobId, projectId, translatedCount, segmentCount }  # after each chunk lands on disk
project://updated               { id, patch }        # reserved for Phase 5+
```

**Nothing binary is ever sent through IPC.** Media and models are referenced
by absolute path and streamed by the process that owns them.

### 5.2 Rust ↔ Python worker (JSON-RPC 2.0 over stdio)

Line-delimited JSON. The worker reads `stdin` line-by-line and writes single
JSON responses to `stdout`. Structured logs go to `stderr`.

```jsonc
// request
{"jsonrpc":"2.0","id":"req-1","method":"ping","params":{}}
// response
{"jsonrpc":"2.0","id":"req-1","result":{"pong":true,"pid":48210,"uptime_ms":1234}}
// error
{"jsonrpc":"2.0","id":"req-1","error":{"code":"E_METHOD","message":"unknown method","data":{...}}}
```

Rust owns the child process, timeouts, cancellation and a request/response
map. The worker never binds to a socket in Phase 1. If HTTP is ever needed
later, it MUST bind to `127.0.0.1` only and require a one-shot secret handed
over stdin.

Phase 1 methods:

```
initialize({ app_version, log_level, data_root, models_root })
                                                    -> { ok: true, worker_version, python_version }
ping()                                              -> { pong: true, pid, uptime_ms }
env_info()                                          -> { python, platform, ffmpeg_available, ffmpeg_version, cpu_count }
shutdown()                                          -> { ok: true }         # worker exits cleanly
```

Phase 3 adds two flavours of RPC on top of the Phase-1 request/response
loop:

* **Async methods** — the worker spawns a daemon thread per request,
  returns nothing immediately, then sends the final response frame
  when the handler finishes. `stt.transcribe` and `stt.download_model`
  are the current members.
* **Notifications** — frames without an `id`, sent from either side.
  The worker emits progress via `stt.progress` / `stt.download_progress`
  notifications (each carrying the originating `requestId`). Rust sends
  `jsonrpc://cancel` notifications to signal cooperative cancellation.

Rust supervisor extensions to support the above:

* `request_no_timeout_with_id(id, method, params)` — like `request`
  but with a caller-chosen id and no deadline, so cancellation is the
  only exit condition for very long-running calls.
* `notify(method, params)` — write a notification frame (no id).
* `cancel_request(id)` — shortcut that sends `jsonrpc://cancel`.
* `subscribe(method, cb)` / `unsubscribe(id)` — per-method
  notification listeners the STT service uses to forward progress
  events to the `job://progress` Tauri channel.

Phase 3 methods:

```
stt.env()                                           -> { devices, defaultDevice, whisperInstalled, modelsRoot }
stt.list_models()                                   -> { models: [{ name, repo, paramsM, installed, sizeBytes?, path? }] }
stt.remove_model({ name })                          -> { ok, name }
stt.download_model({ name })     (async)            -> { ok, name, path, sizeBytes, alreadyInstalled }
                                                      # emits stt.download_progress notifications
stt.transcribe({ audioPath, audioHash, options })   -> { language, segments, cacheKey, model, device,
    (async)                                             computeType, wordTimestamps, options }
                                                      # emits stt.progress notifications
jsonrpc://cancel({ requestId })   (notification)    -> { cancelled, requestId }
```

Phase 4 methods:

```
translate.env()                                     -> { llamaInstalled, modelsRoot, translationRoot,
                                                         defaultModel, promptVersions }
translate.list_models()                             -> { models: [{ name, path, sizeBytes, isDefault }] }
translate.list_prompt_versions()                    -> { versions: [ "translation_prompt_v1", ... ] }
translate.translate({ transcriptCacheKey, audioHash, segments, existingTranslations?, options })
    (async)                                         -> { ok, cacheKey, totalSegments, translatedSegments,
                                                         chunks, model, sourceLanguage, targetLanguage,
                                                         promptVersion }
                                                      # emits translate.progress notifications
                                                      # emits translate.chunk_completed notifications
                                                      #   (id + translation per completed chunk segment)
```

---

## 6. Job / cache model

* **Job queue** lives in SQLite; Rust is the scheduler and only runs
  `max_concurrent_jobs` (default 1) at a time to protect memory.
* **Cache fingerprints**: each stage computes an input fingerprint (hash of
  media path + mtime + relevant settings + upstream fingerprint). Result
  files are indexed by fingerprint in `cache/fingerprints.json`. Editing one
  subtitle invalidates only its TTS output, never the whole render.
* **Model lifecycle**: only one large model resident at a time. The worker
  exposes `load_model`, `unload_model`. Rust decides when to unload based on
  configurable retention. Defaults: unload after 60 s idle.

---

## 7. Security

* No frontend value is ever concatenated into a shell command. All process
  invocations use `Command::new(bin).args(&[...])` with explicit arguments.
* Every user-supplied path is canonicalized and must be a descendant of one
  of the whitelisted roots (app data, app config, imported media roots
  chosen via native pickers).
* Path validation lives in `paths::validate_within(root, candidate)`. Any
  IPC command that accepts a path calls it.
* Python worker is spawned with `stdin`/`stdout` pipes only, no TCP.
* Auto-updates and telemetry are disabled at build time.
* Tauri's `asset:` protocol is enabled so the `<video>` preview can load
  local files directly (no IPC round-trip, no in-memory buffering). Phase 2
  scope is `["**"]` — the file picker is the actual security boundary, and
  Phase 10 will narrow the scope to the user's imported/media roots.

---

## 8. Offline mode

* Setting `offline_mode` defaults to `true`.
* When on, the Rust layer refuses to spawn any process that has known
  network side-effects (model downloads, update checks) and disables the
  UI affordances that trigger them.
* The application must not require internet after models are installed.
  Network errors during optional operations are surfaced as recoverable
  structured errors.

---

## 9. Logging & error handling

* Rust logs via `tracing` → JSON lines to `<app_cache>/logs/rust.log` with
  daily rotation. Log level configurable via settings.
* Python worker logs JSON lines to stderr; Rust captures them, tags with
  the job id and appends to `<app_cache>/logs/worker.log`.
* Errors returned to the UI use a stable shape:

  ```json
  {
    "code": "MODEL_NOT_FOUND",
    "stage": "translation",
    "message": "Translation model is not installed.",
    "recoverable": true,
    "hint": "Install a translation model in Settings → Models."
  }
  ```

* Raw stack traces are only surfaced in dev builds.

---

## 10. Performance rules

* Never load full video/audio into memory. Always stream.
* Never send binary media through IPC. Send paths + progress events.
* React must virtualize any subtitle list > ~200 rows.
* Only one AI model resident at a time (see §6).
* FFmpeg calls prefer `-c:v copy` when only audio/subtitle changed.

---

## 11. Cross-platform

* Primary target: macOS Apple Silicon (development machine).
* Second target: Windows x64.
* Never assume CUDA. Detect Metal (macOS) / DirectML / CPU dynamically from
  the worker and expose the answer via `env_info`.
* All paths use `std::path::PathBuf`; no forward-slash assumptions.

---

## 12. Video engine (Phase 2)

* FFmpeg CLI is the only media engine. Rust owns the process; nothing in
  the frontend spawns ffmpeg directly.
* Detection order: explicit `settings.ffmpeg_path` → bundled sidecar in
  the app resource dir (reserved for Phase 10 packaging) → first
  `ffmpeg`/`ffprobe` found on `PATH`.
* Supported containers: `mp4`, `m4v`, `mkv`, `mov`, `avi`, `webm`.
  Unsupported extensions are rejected before any FFmpeg process runs.
* Source video import defaults to **reference mode** (path + fingerprint
  are stored; the original file is never moved or copied). The user can
  explicitly opt into **copy mode** for portable projects.
* Content fingerprint = `sha256(u64 size ‖ first 64 KiB ‖ last 64 KiB)`.
  Fast even on 20 GB files; combined with `size_bytes` this is the cache
  key for extracted audio.
* Extraction command is fully argv-based (never a shell string):

  ```
  ffmpeg -y -nostdin -hide_banner -loglevel error \
         -i <src> -vn -sn -dn \
         -map 0:a:0 -ac 1 -ar 16000 -acodec pcm_s16le \
         -progress pipe:1 <dest>
  ```

* Progress is parsed from `-progress pipe:1` (real ffmpeg output,
  never faked). Events are throttled to ~10 Hz; DB progress rows are
  persisted at 5 % intervals to keep write amplification low.
* Cancellation: `SIGTERM` on Unix so ffmpeg finalises cleanly, followed
  by a 2 s grace period, then a hard kill. Partial output is deleted.
  Windows uses `TerminateProcess` and always deletes the partial file.
* Disk-space check runs before extraction (`fs4::available_space`) with a
  100 MiB slack in addition to the estimated PCM WAV size.
* Cache lookup on every extract call: matching fingerprint + parameters +
  existing output file → returns `CacheHit` synchronously with no ffmpeg
  spawn. Any mismatch invalidates and re-extracts.

## 13. Speech recognition (Phase 3)

* The whole faster-whisper dependency lives inside
  `movie_translator_worker.stt`. The rest of the application only ever
  imports the `SpeechToTextProvider` protocol and the shape-only
  dataclasses from `movie_translator_worker.stt.models`. That gives us
  a clean swap-in seam for whisper.cpp, sherpa, ggml, or any future
  provider without touching orchestrator code.
* Supported model names: `tiny`, `base`, `small`, `medium`, `large-v2`,
  `large-v3` (also aliased as `large`) and `turbo`. Only models with
  the required snapshot files (`model.bin`, `config.json`) present
  under `<models_root>/whisper/<name>/` are surfaced as
  `installed: true` in `stt.list_models`. Downloads only happen when
  the user explicitly asks for them.
* Model files land under `<app_data>/models/whisper/<name>/`. Rust
  passes that root to the worker via `initialize.models_root`. Tests
  and CI never commit models to Git.
* Device detection surfaces `cpu` unconditionally, `cuda` when
  CTranslate2 reports at least one CUDA device, and `metal` as an
  *architectural placeholder* on Apple Silicon (marked `supported:false`
  until CTranslate2 exposes Metal). Defaults: `cuda` if present,
  otherwise `cpu` — never a device we don't actually support.
* Compute type defaults: `float16` on CUDA, `int8` on CPU. Anything
  else the user picks in Settings is passed through as-is.
* Progress reporting stages: `queued → loading_model → transcribing →
  finalizing → completed`. `transcribing` progress derives from the
  ratio of the last-emitted segment's `end` timestamp to the probed
  duration — real numbers only, never fake.
* Cancellation: Rust registers a job, watches the cancel token, and
  fires `jsonrpc://cancel { requestId }` when it flips. The Python
  handler polls the associated event between segments and raises
  `ProviderCancelled` promptly. The pending-jobs registry in the
  worker removes the entry so no zombie inference lingers.
* Cache invalidation is a single sha256 over `(audio hash, model,
  language, device, compute type, beam size, word timestamps flag,
  vad flag, temperature, initial prompt)`. A cache hit requires that
  hash AND the audio hash to match the saved manifest — either
  alone is not sufficient.

## 14. Translation (Phase 4)

* The whole llama-cpp-python dependency lives inside
  `movie_translator_worker.translation`. The rest of the application
  only ever imports the `TranslationProvider` protocol and the
  shape-only dataclasses from `movie_translator_worker.translation.models`.
  That gives us a clean swap-in seam for whisper.cpp-style GGML
  runtimes, Ollama, MLX, mistral.rs or cloud fallbacks without
  touching orchestrator or UI code.
* **Runtime**: `llama.cpp` via the `llama_cpp` Python binding.
  `Ollama` is explicitly allowed as a developer convenience only and
  must never become a production dependency.
* **Model management**: models live under
  `<app_data>/models/translation/` as user-installed `*.gguf` files
  (Qwen2, Llama 3, Mistral, Phi, ...). We **never** download models
  automatically. `translate.list_models` scans the directory; the
  first hit is flagged as the default. The Settings screen shows
  the directory path so the user knows where to drop files.
* **Language support**: architecturally open — the source/target
  language are just short codes that flow through to the prompt.
  MVP priority is `en → vi`. Adding another target is a one-line
  change to `TRANSLATION_LANGUAGES` in the UI.
* **Prompt versioning**: prompts live in
  `movie_translator_worker.translation.prompts` and are stamped with
  a stable name (`translation_prompt_v1`). The version is baked into
  the cache key so bumping the prompt doesn't invalidate previously
  cached translations until the user explicitly re-translates. The
  system prompt covers: preserve meaning/intent/names/terminology,
  match tone/politeness/register, keep profanity level intact, avoid
  literal translation, avoid explanations and translator notes, keep
  subtitles concise, never invent or drop meaning, and return a
  strict JSON object of the shape `{"segments":[{"id":..,"translation":".."}]}`.
* **Structured output**: the LLM is invoked with
  `response_format={"type":"json_object"}` and the response is parsed
  strictly. The provider verifies every requested `id` is present;
  missing ids raise `TRANSLATE_INCOMPLETE_RESPONSE`, unparseable JSON
  raises `TRANSLATE_INVALID_JSON`. Extra ids the model volunteers are
  silently dropped. The LLM only ever receives — and returns —
  translation content; `id`, `start`, `end` never round-trip through
  the model.
* **Chunking**: the transcript is split into windows of `chunk_size`
  segments (default 30). Each request additionally shows the model
  `context_before` (default 4) previous and `context_after` (default
  2) next segments as scene context; the response covers the *chunk*
  only. Users configure chunk/context size in the Translation panel;
  everything is part of the cache key.
* **Long movies**: chunks flow through the worker sequentially, and
  each completed chunk is emitted as a `translate.chunk_completed`
  notification. Rust persists the batch to `translation.json`
  immediately, then emits `translation://chunk_completed` to the
  frontend. The React store lazy-loads the doc for the editor; the
  panel counters (`translatedCount / segmentCount`) come from a
  compact summary so the sidebar never rehydrates the whole segment
  list.
* **Progress**: three stages — `planning → translating → finalizing`.
  `translating` progress is `(baseline + completedSoFar) /
  totalSegments`, where `baseline` accounts for any segments already
  translated from a partial resume. Never faked.
* **Cancellation**: cooperative between chunks — the worker checks
  the cancel event at the top of every chunk and after inference, and
  raises `ProviderCancelled` promptly. Already completed chunks
  remain on disk and become the resume baseline for the next run.
* **Cache & resume**: see §4.5. Complete → return synchronously.
  Resumable → the service seeds `translation.json` with the previous
  translations, computes which ids are still missing, and only asks
  the LLM about those. User-edited segments (`edited: true`) are
  preserved even across option changes.
* **Manual edits**: the editor writes each blurred change via
  `update_translation_segment`. The service marks the row
  `edited: true` and rewrites the file atomically. Subsequent LLM
  runs treat edited rows as already-satisfied.
* **Performance**: only one GGUF model is loaded at any time and it
  is released before switching. The worker reads model files from
  disk (mmap where llama.cpp supports it); no bytes cross IPC. The
  frontend holds the doc only while the editor is open.

## 15. Subtitle system & editor (Phase 5)

* **Canonical model.** `SubtitleSegment` (`id, start, end, sourceText,
  translatedText, speaker?, voiceId?`) is *the* single subtitle
  representation for the rest of the pipeline. Transcription and
  translation manifests still exist — they're the sources this doc
  derives from — but Phase 6 (TTS), Phase 8 (mix) and Phase 9
  (render) read from `subtitles.json`, not from the older files.
* **Rust-only.** Subtitle math is CPU-cheap and needs no AI or
  external tools; the whole subsystem lives in
  `src-tauri/src/subtitles/` with zero Python involvement.
* **Modules.**
  * `models` — `SubtitleSegment`, `SubtitleDoc`, `DirtyFlags`,
    `DerivedFrom`, patch DTOs, format enums.
  * `cache` — atomic read/write of `subtitles/subtitles.json`.
  * `derive` — build a fresh doc from transcript + optional
    translation, then merge preserving prior human edits.
  * `srt` / `ass` — parse and write the two supported wire formats
    (see §4.6 for scope). Deliberately lightweight; no libass.
  * `service` — `SubtitleService` with `get_doc`, `get_summary`,
    `rebuild_from_sources`, `update_segment`, `add_segment`,
    `delete_segment`, `split_segment`, `merge_segment`,
    `clear_dirty`, `import_from_file`, `export_to_file`. All
    mutations grab a per-service mutex before load-modify-save so
    overlapping UI requests can't lose an edit.
* **Editing operations.** All eight are exposed through IPC (§5):
  edit any field, add / delete rows, split at a timestamp (halves
  the text on the closest whitespace to the midpoint), merge with
  the following row (concatenates text and extends `end`).
* **Video sync.** The Project screen keeps one `<video>` ref and
  broadcasts `timeupdate` events to (a) the subtitle overlay
  renderer and (b) the list, which highlights the active row.
  Clicking a row's id sets `video.currentTime = seg.start`. The
  overlay uses a plain absolutely-positioned `<div>` — we NEVER
  burn subtitles into pixels during editing (that's a Phase 9
  render step).
* **Virtualization.** The subtitle list is a hand-rolled windowed
  renderer (~120 lines, no dep) with fixed row height and 4-row
  overscan. Movies with 3000+ segments stay smooth without
  react-window.
* **Persistence.** Atomic write on every mutation (tmp + rename).
  Nothing about a subtitle edit ever depends on process shutdown
  hooks; a hard kill mid-edit costs at most the last unblurred
  change.
* **Dependency tracking.** `dirty.tts | mix | render` bits sit on
  the doc; edits that could invalidate downstream flip them. The
  UI shows a warning banner while any bit is set; Phase 6+ services
  will call `clear_subtitle_dirty` after a clean pass. `sourceText`
  edits never set dirty bits (nothing downstream reads it).
* **Error surface.** `SubtitleError` → stable UI codes:
  `SUBTITLE_NO_TRANSCRIPT`, `SUBTITLE_NO_DOCUMENT`,
  `SUBTITLE_SEGMENT_NOT_FOUND`, `SUBTITLE_INVALID_TIMING`,
  `SUBTITLE_INVALID_SPLIT`, `SUBTITLE_NO_MERGE_TARGET`,
  `SUBTITLE_INVALID_FILE`, `SUBTITLE_UNSUPPORTED_FORMAT`,
  `SUBTITLE_INVALID_EXPORT_PATH`, `SUBTITLE_IO`, `SUBTITLE_DB`.

## 16. Local TTS / voice synthesis (Phase 6)

The TTS subsystem turns the canonical `SubtitleSegment.translatedText`
into one small WAV file per segment. It is 100% offline after
model installation, integrates with the job registry, and is
strictly hidden behind a provider abstraction so any future engine
can be added without touching the pipeline.

### 16.1 Provider abstraction

Python defines `movie_translator_worker.tts.provider.TTSProvider`:

```python
class TTSProvider(Protocol):
    name: str
    def get_voices(self) -> list[VoiceInfo]: ...
    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        settings: TTSSettings,
    ) -> SynthesisResult: ...
    def unload(self) -> None: ...
```

Every foreseeable failure is raised as `ProviderError` carrying a
stable code (`voice_missing`, `model_invalid`, `engine_failure`,
`out_of_memory`, `disk_full`, `invalid_text`, `unavailable`,
`cancelled`). The RPC layer maps those to `RpcErrorCode.TTS_*`,
and Rust maps them onto `AppError` variants with hints
(e.g. "install piper-tts in the worker environment").

The first concrete backend is `PiperTTSProvider`. Piper was chosen
because it is offline, MIT-licensed, ships small ONNX models
(< 100 MB per voice), runs comfortably on CPU/Apple Silicon, has
first-class Vietnamese voices, and doesn't drag in a heavy Python
runtime stack. Nothing outside of `tts/piper_provider.py` imports
`piper` — the runtime is optional (`pip install .[tts]`).

### 16.2 Voice registry

```
<data>/models/tts/
├── piper/
│   ├── vi_male_01/
│   │   ├── model.onnx
│   │   ├── model.onnx.json    ← Piper's config
│   │   └── voice.json         ← optional metadata override
│   └── vi_female_01/
│       ├── model.onnx
│       └── model.onnx.json
```

* Adding a voice is dropping a folder into place and clicking
  **Rescan** in the TTS panel. No auto-download, no bundled model
  files, no telemetry.
* `voice.json` overrides the display name, gender, language and
  quality tier when the raw `.onnx.json` metadata is thin or wrong.
* `VoiceInfo` is the canonical wire type:

  ```json
  {
    "id": "vi_male_01",
    "name": "Vietnamese Male 01",
    "language": "vi",
    "gender": "male",
    "engine": "piper",
    "modelPath": "/…/model.onnx",
    "configPath": "/…/model.onnx.json",
    "sampleRate": 22050,
    "installed": true,
    "quality": "medium",
    "supportedSettings": ["speed"]
  }
  ```

  The `supportedSettings` list tells the UI which sliders to render
  — Piper honours `speed` only, so `pitch`/`volume` are silently
  hidden for it (but still hashed into the cache key so a future
  engine that supports them gets a proper miss).

### 16.3 Cache identity

The manifest at `voices/voices.json` (schema `TtsCacheFile` in
Rust, mirror of the Python wire) lists one entry per generated
segment:

```json
{
  "segmentId": 12,
  "engine": "piper",
  "voiceId": "vi_male_01",
  "modelName": "model.onnx",
  "cacheKey": "sha256:…",
  "textHash": "sha256:…",
  "text": "Xin chào, tôi tên là Long.",
  "speed": 1.0,
  "pitch": 0.0,
  "volume": 1.0,
  "file": "voices/000012.wav",
  "durationSecs": 3.42,
  "sampleRate": 22050,
  "channels": 1,
  "sizeBytes": 150844,
  "generatedAt": "2026-08-11T09:12:33Z"
}
```

`cacheKey = sha256(engine | voice_id | model_name | text_hash |
speed | pitch | volume)`. It's computed identically in Python and
Rust so both sides always agree. A row counts as *generated* only
when every dimension matches; otherwise the UI flags it as *stale*
until it's regenerated. The literal `text` field is duplicated
purely so the UI can do cheap staleness checks without a crypto
hash on the render path (the authoritative comparison is still the
SHA-256 backend-side).

### 16.4 Preview vs. batch

* **Preview** (`preview_tts_segment` command → `tts.synthesize_one`
  RPC) is synchronous. It first inspects the manifest; if the
  entry is still fresh, it just returns the cached path — nothing
  is regenerated. Only cache misses / stale entries invoke the
  engine, and the result is written straight back into the
  manifest.
* **Batch** (`generate_tts` command → `tts.synthesize_batch` RPC)
  runs as a background job with a `CancelToken` in the shared
  `JobRegistry` (same infrastructure as STT and translation).
  Modes: `all` (regenerate everything from scratch), `missing`
  (skip anything already fresh), and `selected` (regenerate a
  specific ids list). Cancellation stops between segments and
  leaves already-generated files on disk — restarting continues
  from missing segments.

### 16.5 Progress and incremental persistence

The Python worker emits one `tts.progress` notification when it
picks a segment up, plus one `tts.segment_completed` per finished
file. Rust folds each completion into `voices.json` atomically and
re-emits it to the frontend, which updates the "Synthesising
42/350" counter and the per-row badges in real time — no
end-of-batch bulk refresh, no lost work on crash.

### 16.6 Model lifecycle & memory

* Piper's ONNX model is loaded lazily on the first `synthesize`
  call and released via `provider.unload()` after every batch
  (and after every one-off preview). No idle worker holds a TTS
  model in RAM.
* Generated WAVs never enter memory. Rust streams them from disk,
  the UI plays them via `asset://` URLs — the IPC channel carries
  metadata only. React state stores the *manifest*, not the audio.
* Piper's `speed` is applied natively; `volume` is honoured
  post-synthesis by scaling PCM samples in-place inside
  `tts/wav_io.py` (no FFmpeg dependency for the intermediate WAVs).
  `pitch` is currently ignored by Piper but reserved in the
  contract for future engines.

### 16.7 Dependency tracking

`SubtitleDoc.dirty.tts` becomes `true` when the user edits any of
`translatedText`, `voiceId`, `speed`, `pitch`, or when the default
voice changes. After a successful batch,
`TtsService::maybe_clear_dirty` clears the flag **only** when every
non-empty segment has a matching (`generated`) manifest entry —
partial coverage keeps the doc dirty so the pipeline UI shows a
warning.

### 16.8 Offline & privacy

Everything after `pip install .[tts]` and a one-time voice download
runs with the network disabled. No telemetry, no analytics, no
remote API calls, no crash reporting. The manifest, the WAVs, and
the models all live inside the project's own data directory.

## 17. Voice synchronisation (Phase 7)

Phase 7 turns the per-segment TTS WAVs from Phase 6 into
timing-aligned WAVs whose duration exactly matches the subtitle
window (`subtitle.end − subtitle.start`). It never concatenates
segments — the final mix (Phase 8) is responsible for placing each
aligned WAV at its absolute `subtitle.start` on the master timeline,
so the natural silences between subtitles are preserved by
construction.

Aligned WAVs are written per-segment under:

```text
<project>/voices/synced/000001.wav
<project>/voices/synced/000002.wav
…
<project>/voices/synced/sync.json   # manifest, atomic writes
```

### 17.1 Provider abstraction

Sync is implemented entirely in the Python worker via FFmpeg
(`atempo` for stretching, `apad` for trailing silence, `wave` for
the empty case). Rust never touches audio bytes — it dispatches
`sync.apply_batch` / `sync.apply_one` RPCs with source paths, target
paths, and the resolved plan. Extending to a different DSP backend
means implementing a new pair of RPC handlers.

### 17.2 Plan and status

Each segment is classified purely from durations in
`SyncPlanner.plan_segment`:

| Status      | Condition                                                    | Action                                    |
|-------------|--------------------------------------------------------------|-------------------------------------------|
| `empty`     | segment text empty or window ≤ 0                             | write silence of `targetDurationSecs`     |
| `fits`      | `actual ≤ target` after 1.0× baseline                        | copy + `apad` to `targetDurationSecs`     |
| `adjusted`  | `minSpeed ≤ speed ≤ maxSpeed` (default 0.85–1.20)            | `atempo=speed` (chained if outside 0.5–2) |
| `too_long`  | required `speed > maxSpeed`                                  | stretch at `maxSpeed`, tag warning        |

`too_long` intentionally still writes a WAV (at `maxSpeed`) so the
mix stage has *something* to place on the timeline, but the UI
badges it as `⚠ Too long` and the user is expected to shorten the
translation or extend the subtitle window.

### 17.3 Cache identity

Both sides derive the same `cacheKey` from:

```text
sha256(
  ttsCacheKey                  # invalidates when TTS changes
  || round_ms(targetDuration)  # invalidates when timing changes
  || min_speed || max_speed    # invalidates when speed policy changes
  || sample_rate || channels   # invalidates when output shape changes
)
```

Rounding the target duration to milliseconds keeps the key stable
against sub-millisecond floating-point drift in the subtitle timing.
A hit means "the file on disk was produced from this exact
combination"; a miss triggers a fresh FFmpeg run.

### 17.4 Preview vs. batch

* `preview_sync_segment` (Tauri command, Python RPC) is a single-shot
  synchronous call: it checks the manifest and only re-runs FFmpeg
  on a miss. The response includes the absolute path so the UI can
  point an `<audio>` element at it.
* `apply_sync` fans out through the `JobsRepo`/`JobRegistry` as a
  `JobStage::Sync` job. The Python worker processes segments one at
  a time and emits `sync.progress` + `sync.segment_completed`
  notifications; Rust persists the manifest after every completion
  so cancelling mid-run leaves a consistent `sync.json` on disk.

### 17.5 Progress and incremental persistence

`sync.segment_completed` carries `syncedCount`, `subtitleCount`,
`status`, and the relative path, so the UI can update per-row
badges without reloading the whole manifest. The batch handler
never buffers audio in memory — it hands file paths to FFmpeg and
reads the resulting WAV headers only to fill in `sampleRate`,
`channels`, `sizeBytes`, and `finalDurationSecs`.

### 17.6 Dependency tracking

`DirtyFlags` is extended with a `sync` bit. `SubtitleService`
distinguishes between content edits and timing edits:

* Content edits (translated text, speaker, voice) →
  `mark_content_dirty()` sets `tts | sync | mix | render`.
* Timing edits (start / end) → `mark_timing_dirty()` sets
  `sync | mix | render` only — transcription and translation stay
  clean, and Phase 6 TTS WAVs are still reusable because the audio
  content hasn't changed.

After a successful sync batch,
`SyncService::maybe_clear_dirty` calls
`SubtitleService::clear_dirty_flags(project_id, DirtyFlags::only_sync())`
so only the `sync` bit is cleared. Mix and render stay dirty until
their own stages run.

### 17.7 Offline & privacy

Sync only depends on FFmpeg's `atempo` / `apad` / `aresample`
filters — all bundled in every mainstream FFmpeg build. No
network, no ML models, nothing to download.

## 18. Audio mixing (Phase 8)

Phase 8 folds every Phase 7 synced voice WAV back into the original
movie soundtrack at each subtitle's absolute start time and writes the
result to `<project>/audio/mixed_vi.wav`. Unlike the AI stages,
mixing is deterministic FFmpeg work, so it lives entirely on the
Rust host — no Python worker in the loop.

### 18.1 Provider abstraction

There is no provider trait here (yet). The FFmpeg command builder
(`crate::mix::ffmpeg_cmd::build_mix_command`) is a pure function of
`(source_video, voice_segments[], MixSettings, output)`. Swapping in
a dialogue-separator preprocessor later (Demucs / Spleeter / etc.)
means adding a stage that produces additional `MixVoiceInput` /
`MixOriginalInput` layers — the mix core does not have to change.

### 18.2 Filter graph shape

For N non-empty synced voices with ducking enabled, the graph is:

```
[0:a:0]aformat=…                                             -> [orig]
[i:a:0]adelay=t_i|t_i:all=1,aformat=…                        -> [v_i]   (× N)
[v_1][v_2]…amix=inputs=N:normalize=0:duration=longest        -> [voice_raw]
[orig]volume=orig_vol                                        -> [orig_g]
[voice_raw]volume=voice_vol                                  -> [voice_g]
[orig_g][voice_g]sidechaincompress=threshold=…:ratio=…       -> [orig_ducked]
[orig_ducked][voice_g]amix=inputs=2:normalize=0:duration=…   -> [mix]
```

With ducking off, `sidechaincompress` is skipped and the two
gain-stages feed straight into the final `amix`. `-shortest` bounds
the output to the source video's length so the voice-track's silence
tail can't drag it out.

### 18.3 Volume and ducking knobs

Both live in [`MixSettings`](src-tauri/src/mix/models.rs) with the
defaults from the phase spec:

* `original_volume` — 0.70 (70%). Linear multiplier applied to the
  original stream.
* `voice_volume` — 1.00 (100%). Linear multiplier applied to the
  merged voice stream.
* `ducking_enabled` — on by default. When on, an FFmpeg
  `sidechaincompress` node keyed by `[voice_g]` clamps `[orig_g]`
  down while the voice speaks.
* `ducking_depth_db`, `ducking_threshold_db`, `ducking_attack_ms`,
  `ducking_release_ms` — expose the compressor knobs the UI lets the
  user tune. `ducking_depth_db` is mapped monotonically into a
  compressor ratio via `duck_makeup_ratio_from_depth_db`.
* `output_sample_rate` / `output_channels` — output shape. Default
  is stereo, source sample rate (falls back to whatever FFmpeg's WAV
  encoder prefers when not specified).

Every field is clamped by `MixSettings::normalised` so a hand-edited
`mix.json` cannot hand FFmpeg absurd values.

### 18.4 Cache identity

The mix cache key is deterministic over:

* the source video fingerprint (Phase 2 `SourceFingerprint`),
* every synced voice's `(segment_id, target_start_secs, sync_cache_key)`,
  sorted by id so ordering doesn't matter,
* every gain / ducking / channel / sample-rate setting.

Full formula lives in
[`build_mix_cache_key`](src-tauri/src/mix/models.rs). The Rust tests
(`src-tauri/src/mix/tests.rs`) pin the invariants: reorder-stable,
changes on any single setting or fingerprint bit, immune to
sub-millisecond floating-point wobble on `target_start`.

### 18.5 Progress and cancellation

`MixService::run_ffmpeg` spawns one `ffmpeg` process with `-progress
pipe:1` and reuses the existing `ffmpeg::progress::feed_line`
parser (originally written for Phase 2 audio extraction). Fractions
are throttled to one emit per 100ms and persisted to the `jobs` row
every 5% of progress. Cancellation from the shared `JobHandle`
token races the child via `SIGTERM` (Unix) and cleans up any partial
`mixed_vi.wav` on the way out.

### 18.6 Dependency tracking

`DirtyFlags::mix` is set whenever any of the following change:

* Any subtitle segment's timing or content (already sets `sync` +
  `mix` via `mark_timing_dirty` / `mark_content_dirty`).
* Any TTS regenerated (Phase 6 already fans out into `sync` + `mix`).
* Any sync WAV regenerated (Phase 7 upserts change the cache key that
  feeds into `build_mix_cache_key`).
* Any volume / ducking / channel setting change (folded into the
  cache key so a mismatch flips `MixSummary::status` to `stale`).

After a successful mix pass, `MixService::maybe_clear_dirty` calls
`SubtitleService::clear_dirty_flags(project_id, DirtyFlags::only_mix())`
so only the `mix` bit is cleared. `render` stays dirty until the
Phase 9 mux stage runs. Transcription and translation are never
touched by mix.

### 18.7 Offline & privacy

Mix only depends on FFmpeg's `amix`, `adelay`, `volume`,
`sidechaincompress`, and `aformat` filters — all bundled in every
mainstream FFmpeg build. No network, no ML models, nothing to
download.

## 19. Final video rendering (Phase 9)

Phase 9 wraps the pipeline: it takes the original video (untouched),
the Phase 8 mixed Vietnamese audio (`audio/mixed_vi.wav`), and the
current Vietnamese subtitle document, and produces the final movie
under `<project>/output/movie_vi.<ext>` (mp4 by default). Like Phase
8, rendering is deterministic FFmpeg work, so it lives entirely on
the Rust host — no Python worker in the loop.

### 19.1 Filter graph & muxer shape

The FFmpeg command lives in
[`crate::render::ffmpeg_cmd::build_render_command`](src-tauri/src/render/ffmpeg_cmd.rs)
and is a pure function of
`(source_video, mixed_audio, burn_subtitle?, RenderSettings, output)`.
Three shapes exist, keyed off `SubtitleMode`:

* **External** — video stream-copied (`-c:v copy`, no re-encode),
  Vietnamese SRT exported alongside as `output/movie_vi.srt`. Audio
  re-encoded from the mixed WAV via the configured codec (default
  `aac @ 192k`).
* **Burned** — subtitles baked into pixels via FFmpeg's `subtitles=…`
  filter. Because that's a video filter, this forces a video
  re-encode; `RenderSettings::normalised` silently upgrades
  `VideoCodec::Copy` to `libx264` (preset `medium`, CRF 20, yuv420p)
  when the mode requires it.
* **None** — same as External but no sidecar SRT.

In every shape we `-map 0:v:0` (source video only) and `-map 1:a:0`
(mixed WAV). The source's original audio is deliberately dropped so
the final file has exactly one Vietnamese-dubbed track. MP4 outputs
get `-movflags +faststart` to move the moov atom to the front for
streaming/web playback; MKV does not.

### 19.2 Progress and cancellation

`RenderService::run_ffmpeg` spawns one `ffmpeg` process with
`-progress pipe:1` and reuses the shared
`ffmpeg::progress::feed_line` parser. Fractions are throttled to
one emit per 100ms on the `job://progress` bus and persisted to the
`jobs` row every 5% of progress. Cancellation from the shared
`JobHandle` token races the child via `SIGTERM` (Unix) and removes
the incomplete `movie_vi.<ext>` on the way out so the user is never
left with a broken half-file dressed up as a successful render.

### 19.3 Post-render validation

Right before persisting the manifest,
`RenderService::run_ffmpeg` runs `ffprobe` on the produced file
and checks:

* file exists and `size > 0`;
* `duration > 0`;
* at least one video and one audio stream (subtitle stream count is
  recorded but not required, since we ship the SRT as a sidecar
  rather than muxing it into MP4/MKV in this release).

Any failure removes the partial file and marks the job `Failed`
with a stable code (`RENDER_OUTPUT_INVALID` /
`RENDER_VALIDATION_FAILED`). The Phase 9 spec's "never silently
report success" rule is enforced here.

### 19.4 Cache identity

The render cache key is deterministic over:

* the source video fingerprint (Phase 2 `SourceFingerprint`),
* the current Phase 8 `MixEntry::cache_key` (so any upstream change
  ripples into the render key without re-probing everything),
* every render setting after normalisation — output format, video
  codec, audio codec + bitrate, subtitle mode.

Full formula lives in
[`build_render_cache_key`](src-tauri/src/render/models.rs). Rust
tests (`src-tauri/src/render/tests.rs`) pin the invariants: the key
changes on any single upstream or setting bit, and the burn-mode
video-codec promotion collapses `Copy` and explicit `libx264` to the
same key so the UI can advertise either without splitting the cache.

### 19.5 Output & source protection

* Default output is always
  `<project>/output/movie_vi.<extension>`.
* Users can pick a custom absolute path via
  `RenderSettings::output_path`. Relative paths are refused
  (`RENDER_INVALID_OUTPUT_PATH`).
* The original source video is opened read-only and never modified;
  the resolver in `render::service::resolve_output_path` only ever
  returns a candidate to write, never rewriting the source. All
  outputs live in either the project's `output/` folder or the
  user-chosen absolute path.

### 19.6 Dependency tracking

`DirtyFlags::render` is set whenever any of the following change:

* Any subtitle segment's timing or content (already sets `sync +
  mix + render` via `mark_timing_dirty` / `mark_content_dirty`).
* Any TTS / sync / mix output regenerated (their cache keys feed
  the render key).
* Any render setting change (folded into the cache key so a mismatch
  flips `RenderSummary::status` to `stale`).

After a successful render pass, `RenderService::maybe_clear_dirty`
calls `SubtitleService::clear_dirty_flags(project_id,
DirtyFlags::only_render())` so only the `render` bit is cleared —
never the upstream `tts / sync / mix` bits. No upstream stage is
re-run by render, and re-runs of upstream stages do not trigger
render.

### 19.7 Offline & privacy

Rendering only depends on FFmpeg's `subtitles`, `libx264` /
`libx265` / `libvpx-vp9` encoders, and the standard MP4/MKV muxers
— all bundled in every mainstream FFmpeg build. No network, no ML
models, nothing to download.

## 20. Local Model Manager & Offline-First (Phase 10)

Phase 10 makes model management a **first-class**, cross-stage concern
and codifies the app's offline-first stance. The core principle is
that **AI models are external assets** — they never ship inside the
application executable, and the app never fetches them for the user
without an explicit action.

### 20.1 Model directory layout

Models live under a **single user-controlled root**, discovered via the
OS-standard app-data directory (or a user override — see §20.4):

```
<models_root>/
├── whisper/          # per-model directories (CTranslate2 layout)
│   └── <name>/{model.bin, config.json, tokenizer.json, ...}
├── translation/      # GGUF files consumed by llama.cpp
│   └── *.gguf
└── tts/
    └── <engine>/     # e.g. piper, xtts
        └── <voice>/  # ONNX + JSON pair for Piper
```

The per-stage Python providers (Phases 3, 4, 6) already own the scan
logic for their own subdirectories. Phase 10 does **not** re-implement
those scanners; it aggregates their output.

### 20.2 Central registry (`src-tauri/src/models/`)

New Rust module (`crate::models`) with three files:

* `registry.rs` — the `ModelRegistry` handle held by `AppState`.
  `list()` returns a cached `Vec<LocalModel>`; `rescan()` invalidates
  the cache and re-queries the worker's per-stage registries
  (`stt.list_models`, `translation.list_models`, `tts.list_voices`)
  concurrently via `tokio::join!`, then normalises the results into
  a single flat list with a stable `status` (`available` / `missing`
  / `invalid`). Validation is **lightweight**: existence + required
  file names (`model.bin`, `config.json` for Whisper; `.gguf`
  extension + non-zero size for translation; `.onnx` non-zero for
  voices). No model is ever loaded to validate it.
* `import.rs` — `import_local_model(models_root, spec)` places a
  user-picked source (folder or file) under the correct subdirectory.
  Two strategies:
  - **`Link`** (default) — creates a symlink so a multi-GB Whisper
    snapshot isn't duplicated. Falls back to `Copy` if the platform
    refuses (Windows without dev mode).
  - **`Copy`** — walks the source and copies bytes. Slow but
    self-contained.
  The importer **never overwrites** an existing destination; the
  user gets a clean `MODEL_ALREADY_EXISTS`.
* `errors.rs` — `ModelManagerError` with stable codes:
  `MODEL_UNSUPPORTED_KIND`, `MODEL_INVALID_SOURCE_PATH`,
  `MODEL_SOURCE_NOT_FOUND`, `MODEL_UNSUPPORTED_SOURCE`,
  `MODEL_MISSING_REQUIRED_FILE`, `MODEL_UNREADABLE`,
  `MODEL_INVALID_NAME`, `MODEL_ALREADY_EXISTS`,
  `MODEL_PERMISSION_DENIED`, `MODEL_DIR_NOT_WRITABLE`,
  `MODEL_NETWORK_DISABLED`, `MODEL_IO`.

`LocalModel` carries **only metadata** — `id`, `name`, `kind`,
`engine`, `language`, `path`, `sizeBytes`, `version`, `status`, `hint`.
Never binaries; never model weights held in Rust.

### 20.3 Offline Mode is a hard gate

`AppSettings.offline_mode` was already the default. Phase 10 turns it
into an explicit contract:

* The **only** network entry point in the entire app is
  `stt.download_model` (Hugging Face Hub via `huggingface_hub`).
  `commands::download_whisper_model` now refuses to enqueue when
  `offline_mode == true`, returning a `MODEL_NETWORK_DISABLED`
  error the UI renders with a "Turn off Offline Mode in Settings,
  or install the model manually via Add Local Model" hint.
* The Python worker never phones home for anything else — the STT
  registry only downloads on explicit `stt.download_model`, the
  translation registry only scans local files (`_hf_download`
  is not used), and the TTS registry is fully offline.
* No telemetry, no analytics, no "check for updates" job. There is
  no HTTP client hidden anywhere else in the codebase.

### 20.4 Models directory override + first-run

Two new `AppSettings` fields:

* `models_dir_override: Option<String>` — user-picked absolute
  path. When present, replaces the OS-default resolved by
  `AppPaths::models_dir`. `AppState::effective_models_dir()`
  centralises the resolution so every consumer (worker init,
  registry scan, importer, Settings panel) agrees on one path.
  The `set_model_directory` command writes the setting, calls
  `WorkerSupervisor::reinitialize_models_root` so the change takes
  effect without an app restart, and invalidates the registry cache.
* `first_run_completed: bool` — set once the user dismisses the
  Dashboard's first-run banner. The banner never triggers a
  download; it only nudges the user to Settings › AI Models. The
  app is fully usable if the banner is ignored forever.

### 20.5 Per-project model configuration

Migration `004_add_project_model_config` adds four nullable columns
to `projects`: `whisper_model`, `translation_model`, `tts_engine`,
`tts_voice_id`. All optional; pre-Phase-10 projects (and freshly
created ones) fall through to the global default in `AppSettings`.

* `db::Db::set_project_models(id, patch)` — targeted UPDATE.
* `projects::service::update_models(id, patch)` — validates, writes
  the DB, and refreshes the `project.json` mirror on disk so the
  project folder stays self-describing.
* `commands::update_project_models` — Tauri entry point.
* Front-end store auto-persists the chosen model whenever the user
  starts transcription / translation / TTS generation, so opening
  the same project tomorrow uses the same model.
* When a referenced model is missing at run time, the per-stage
  service raises `MODEL_NOT_INSTALLED` (existing behaviour) — no
  silent substitution.

### 20.6 Lifecycle & memory management

Every AI stage already loads-on-demand:

* Whisper: loaded inside the transcription RPC handler, dropped
  when the handler returns.
* Translation: `llama.cpp` context created for the job, freed when
  it finishes.
* TTS: engines cached with an explicit `tts.unload` RPC.

Phase 10 wires an **explicit "Unload all" surface** into the UI
(`unload_all_models` → `TtsService::unload_all`) so the user can
free the worker's resident memory between long idle periods without
restarting the app. Concurrency-safety is already provided by the
existing job queue (`AppSettings.max_concurrent_jobs`, default 1)
which stops two jobs from touching the same engine simultaneously.

### 20.7 Front-end surface

* `src/screens/Settings.tsx` — new **AI Models** panel: model
  directory (with "Change…" and "Reset to default"), filter dropdown,
  aggregated model table (badge per kind, status pill, path, size,
  hint), **Scan Models**, **Add Local Model** (kind picker, source
  browser, name override, symlink-first import), **Unload all**.
* `src/screens/Dashboard.tsx` — dismissible **first-run banner** that
  nudges the user to Settings › AI Models on first launch. Never
  triggers a download. Counts locally-available models when
  present.
* `src/ipc/types.ts` + `src/ipc/bridge.ts` — `LocalModel`,
  `ModelDirectoryInfo`, `ImportModelSpec`, `ProjectModelPatch`,
  `pickDirectory()`, `pickModelSource()`, and the seven new
  Tauri commands (`listLocalModels`, `rescanLocalModels`,
  `getModelDirectory`, `setModelDirectory`, `importLocalModel`,
  `unloadAllModels`, `updateProjectModels`).
* `src/state/store.ts` — cached `localModels` + `modelDirectory`
  state, refresh-on-mount, and `markFirstRunCompleted` helper.

### 20.8 Performance

* The registry is **cache-first**. Any UI read (`listLocalModels`)
  returns the cached list instantly. Rescans happen only on
  user action (Scan Models button), on `set_model_directory`, and
  on the first call after `import_local_model`.
* Aggregation is O(N) over the flat lists returned by the per-stage
  scanners — no on-demand file walks in Rust beyond the
  lightweight validation described above.
* No new database rows per scan — the metadata is transient.

## 21. Performance, RAM & CPU (Phase 11)

Phase 11 pushes on the *cost* of the pipeline rather than adding new
capabilities. The rule of thumb: **large data stays on disk, IPC
carries only summaries, models load lazily and release on idle**.

### 21.1 IPC & React memory

The React store holds only cheap surfaces: `AppSettings`, project
list, per-stage *summaries*, per-job *snapshots*, live progress
fractions. The heavy artefacts (transcript, translation, TTS
manifest, sync manifest, mix manifest, source video, generated
audio, rendered movie) live on disk. The UI asks for them by
project id and receives compact JSON documents — never raw audio
or video bytes over IPC.

Media playback uses `convertFileSrc` (Tauri's `asset:` protocol) so
the video/audio player streams straight from disk with no
per-render `URL.createObjectURL` blob copy.

### 21.2 Progress event throttling

Progress notifications flow **Python → Rust → UI**. Two throttles
short-circuit the flood before it reaches the DOM:

* **Python `HandlerContext.emit_progress`** coalesces `*.progress`
  notifications at ~20 Hz per method (50 ms window). Fraction
  jumps ≥1% always pass so Whisper's per-segment ticks reach the
  UI; frame-by-frame inner-loop callbacks do not. Semantic events
  (`*.segment_completed`, `*.chunk_completed`) bypass throttling —
  they drive incremental UI refreshes.
* **Rust service layer** (`stt`, `translation`, `tts`, `sync`,
  `mix`, `render`, `audio/extractor`) re-throttles emitted
  `job://progress` events to 10 Hz per job so a burst of Whisper
  segments never hammers the Tauri event bus.

The frontend `store.ts` additionally **debounces** the
completion-driven refresh calls (`loadTranslationDoc`,
`loadTtsManifest`, `loadSyncManifest`) at 350 ms per project so a
translation run with 30-segment chunks coalesces its 100 chunk
notifications into a couple of doc reloads rather than 100.

### 21.3 Long subtitle & translation lists

Both `SubtitleList` and `TranslationEditor` are windowed: only the
rows intersecting the scroll viewport (`viewport / rowHeight +
overscan`) are in the DOM. A 5 000-segment movie renders ~20 rows
at a time, not 5 000.

### 21.4 Model lifecycle & auto-unload

The Python worker is a single **persistent process** with a
per-stage provider (Whisper, llama.cpp, Piper). Each provider now
exposes an explicit `unload()` method that releases the resident
model. The Rust side wraps them as `stt.unload`, `translate.unload`
and the existing `tts.unload` handlers.

The UI drives the policy: the store schedules a
`unload_stage_models(stage)` command `autoUnloadAfterSecs` after
the *last* job of that stage settles (default 90 s). A new job of
the same stage cancels the pending unload — a typical
translate→tts→sync pipeline never unloads mid-flight.

A user-facing "Unload all" button in *Settings › AI Models* remains
as the immediate release surface.

### 21.5 Advanced perf settings

New global settings, all optional, all safe by default:

| Setting                | Default | Effect                                    |
| ---------------------- | ------- | ----------------------------------------- |
| `autoUnloadAfterSecs`  | 90      | Grace period before per-stage auto-unload |
| `cpuThreads`           | *auto*  | `n_threads` for llama.cpp; `cpu_threads` for faster-whisper |
| `gpuAcceleration`      | true    | Enables Metal / CUDA (`n_gpu_layers=-1`) when engine supports it |

Changes propagate to the running worker via a fresh `initialize`
RPC (`reinitialize_perf`) so the next model load picks them up
without restart. Providers ignore unsupported knobs — a machine
with no Metal silently falls back to CPU regardless of the flag.

### 21.6 Source fingerprint reuse

`get_project_media` runs on every job update; hashing a multi-GB
movie on every call — even the cheap 128 KiB sentinel window —
adds up over a long day. Phase 11 short-circuits: the fingerprint
stored on `ProjectRecord` is trusted while the file's `size` and
`mtime` still match; only stat drift triggers a fresh
`fingerprint_file` read.

### 21.7 Orphan `.tmp` sweep

Every cache writer (`config`, `render/cache`, `mix/cache`,
`sync/cache`, `subtitles/cache`, `tts/cache`, `translation/cache`,
`stt/cache`, `audio/cache`) uses the "write to `.tmp` then rename"
pattern for atomic replacement. On startup a background task
recursively sweeps `<projects>/**/*.tmp` (max depth 8) and removes
any orphaned sidecars from a previous crashed run.

### 21.8 Runtime resource monitor

`get_runtime_stats` returns a cheap snapshot: active job count,
active project count, host RSS, worker RSS, worker uptime. The
Settings › Performance panel polls it every 3 seconds; on Linux we
read `/proc/<pid>/statm`, on macOS / Windows we shell out to
`ps -o rss=`. No `sysinfo` dependency, no continuous background
polling.

### 21.9 Long-movie invariants

* Transcript, translation, subtitle, TTS, sync, mix and render
  manifests are chunked writes — nothing keeps the full document
  in a single buffer beyond what's needed for the current pass.
* FFmpeg is invoked exactly **once per stage** (extract, mix,
  render) with a filter graph that streams through the file
  rather than materialising intermediates.
* SQLite stores metadata and small structured data only; audio,
  video and model blobs live on the filesystem behind cache keys.

## 22. Production build, packaging & release (Phase 12)

Phase 12 turns the dev app into a lightweight, offline-first desktop
application without touching the core architecture. Everything below
is packaging, reliability, security and release configuration.

### 22.1 Application identity & bundle config

* `productName = "Local Movie Translator"`, `identifier =
  "app.localmovietranslator"`, `version = "0.1.0"` — the same
  version string flows through `Cargo.toml`, `package.json`,
  `tauri.conf.json`, `AppInfo.appVersion` (from `CARGO_PKG_VERSION`)
  and the bundled artefact names. Bumping the version happens in one
  place.
* `bundle.targets = "all"` — Tauri picks the native target(s) for
  the current host: `.dmg` + `.app` on macOS, `.msi` + `.exe` on
  Windows, `.deb` + `.AppImage` on Linux.
* `bundle.fileAssociations` maps `.mp4`, `.mkv`, `.mov`, `.m4v`,
  `.webm`, `.avi` as *Viewer* and `.srt`, `.ass`, `.ssa` as
  *Editor* — double-clicking a movie in Finder / Explorer / Files
  opens the app.
* Per-platform bundle blocks:
  * **macOS** — `minimumSystemVersion: 11.0`; `signingIdentity` and
    `entitlements` left `null` (fill in for distribution).
  * **Windows** — `webviewInstallMode: downloadBootstrapper` (the
    installer fetches WebView2 if it's missing);
    `allowDowngrades: true`.
  * **Linux** — `.deb depends` on `libwebkit2gtk-4.1-0`, `libssl3`,
    `ffmpeg`, `python3`. AppImage does *not* bundle a media
    framework — FFmpeg is a runtime dependency, not a shipped
    binary.

### 22.2 Release profile

`src-tauri/Cargo.toml`:

```toml
[profile.release]
strip = true
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
debug = false
```

Deliberately not `opt-level = "z"`/`"s"` — the Rust binary still
does non-trivial JSON parsing, hashing and subtitle math on hot
paths, and size-optimising them measurably slowed transcript →
subtitle conversion. The binary is small anyway (< 20 MB stripped).

A parallel `release-with-debug` profile inherits from release but
keeps line tables, so an opt-in symbolicated crash report can be
produced later without shipping debug bloat to normal users.

### 22.3 Security surface

* **Least-privilege capabilities** (`capabilities/default.json`):
  only `core:default`, `core:event:allow-listen`,
  `core:event:allow-unlisten`, `dialog:allow-open`,
  `dialog:allow-save`. No filesystem, shell, or window
  permissions — all persistence goes through Rust.
* **CSP** — `default-src 'self'; script-src 'self'; style-src
  'self' 'unsafe-inline'; img-src 'self' data: asset:
  http://asset.localhost; media-src 'self' asset:
  http://asset.localhost; connect-src ipc: http://ipc.localhost`.
  The frontend cannot reach any origin except the Tauri IPC bridge
  and the asset protocol.
* **Path validation** — every path that touches disk goes through
  `paths::validate_within(root, candidate)`. Frontend never passes
  raw paths for storage commands; instead it names a
  `PathKind::{Data,Config,Log,Cache,Projects,Models}` and Rust
  resolves it against the known-safe root.
* **No shell interpolation** — every subprocess (FFmpeg, Python
  worker, `open`/`explorer`/`xdg-open` for "reveal in file
  manager") is invoked with a typed `Command` + `arg()` array.

### 22.4 Python worker: deterministic location & graceful missing-python UX

`worker/supervisor.rs`:

* `detect_python_bin()` probes in order: `LMT_PYTHON` env override
  → `<bundle>/Contents/Resources/python-embed/bin/python3` (macOS)
  → `<exe>/python-embed/bin/python3` (Linux/portable Windows) →
  `<exe>/python-embed/python.exe` (Windows) → `python3` on `PATH`
  → `python` on `PATH` → literal `"python3"` (spawn will fail
  cleanly with a hint).
* `detect_worker_root()` prefers
  `<bundle>/Contents/Resources/python` in a packaged macOS app so
  the packaged app can never accidentally reach a stale dev checkout
  next to it, then walks up from the executable up to 6 levels
  looking for `python/src/movie_translator_worker/`, then falls
  back to `CARGO_MANIFEST_DIR/../python` (dev).
* Spawn failures are classified in `errors.rs`:
  `WORKER_PYTHON_MISSING` for `ErrorKind::NotFound`,
  `WORKER_SPAWN_DENIED` for `PermissionDenied`, generic
  `WORKER_SPAWN` otherwise. Each carries an actionable
  `hint`.

### 22.5 Logs, retention & sanitisation

`logging.rs`:

* Daily-rotated JSON lines at `<cache>/logs/rust.log[.YYYY-MM-DD]`.
* `LOG_RETENTION_DAYS = 14` — startup calls `prune_old_logs` to
  drop older `rust.log.*` / `worker.log.*` rotations. Other files
  in the log directory are left alone.
* `clear_active_logs(log_dir)` truncates in place so the tracing
  appender's open file handle stays valid — used by the
  `clear_logs` Tauri command.
* **Sanitisation contract** — every `tracing` call in the codebase
  logs method names, timings, sizes and counts. Transcripts, LLM
  prompts/responses, subtitle text, filenames of user media, and
  file contents are never logged. `RUST_LOG=trace` unlocks verbose
  developer output for on-demand debugging without changing any
  of these payload rules.

### 22.6 Storage, cache & log surface

New Tauri commands (registered in `lib.rs`):

* `open_app_path(kind)` — reveals one of the app-owned dirs in the
  OS file manager (`open` / `explorer` / `xdg-open`). `PathKind`
  is validated on the Rust side; unknown values return
  `STORAGE_UNKNOWN_PATH_KIND`.
* `get_storage_stats()` → `StorageStats{ data*, cache*, log*,
  projects*, models* }`. Sizes computed with a bounded recursive
  walk (`MAX_WALK_DEPTH = 6`) off the async runtime, so a
  pathological tree can't hang the UI.
* `clear_cache()` — removes files under `<cache>/…` **except** the
  log subdir (its own command owns that) and returns bytes
  freed. Never touches per-project data, models or rendered
  outputs.
* `clear_logs()` — prunes rotations older than the retention
  window and truncates active files.

Frontend counterpart: new **Settings › Storage & Logs** panel
listing each root with size and per-row "Open" button, plus
"Recompute sizes", "Clear cache", "Clear logs" actions.

### 22.7 Crash recovery UX

* `JobsRepo::reap_orphans` (existing since Phase 1) runs at startup
  and flips any `queued`/`running`/`paused` rows to `failed` with
  `error_code = 'JOB_ORPHANED'`.
* Phase 12 adds `JobsRepo::list_orphaned` + a `list_orphaned_jobs`
  command exposing the most recent 50 rows so the Dashboard can
  render a per-project banner ("Some jobs didn't finish last
  time…"). Users click through and re-run the affected stage —
  every stage is idempotent and honours the on-disk cache, so
  completed segments (e.g. translation 1–280 of 350) are always
  preserved.
* Phase 11's orphan-`.tmp` sweep (`app.rs::sweep_orphan_temp_files`)
  keeps running at startup so stale atomic-write temporaries don't
  accumulate.

### 22.8 User-friendly errors

* Every service returns `AppError { code, stage, message,
  recoverable, hint }`. Phase 12 tightened three surfaces:
  * `WorkerError::Spawn` split into `WORKER_PYTHON_MISSING` /
    `WORKER_SPAWN_DENIED` / `WORKER_SPAWN` with hints.
  * New `STORAGE_UNKNOWN_PATH_KIND`, `STORAGE_OPEN_FAILED` codes.
  * Dashboard's `formatDashboardError` and Settings' `errorMessage`
    helpers now include `hint` below the message so users see
    actionable text, not a bare code.
* Raw stack traces never reach the UI — `From<std::io::Error>`
  and `From<serde_json::Error>` wrap into `AppError` with generic
  codes (`IO`, `SERIALIZATION`) so a panic path can't leak
  implementation details.

### 22.9 Offline promise

* `AppSettings::offline_mode` defaults to `true`.
* Zero HTTP client crates in `Cargo.toml` for production paths; the
  frontend has no `fetch`/`axios`/`WebSocket` calls. The only
  network the CSP allows is Tauri's own `ipc://` and `asset://`
  loopback origins.
* `ModelManager::download_whisper_model` respects
  `AppSettings.offline_mode` — turning offline mode off doesn't
  auto-download anything today, it just unlocks a future opt-in
  action. There is no telemetry, analytics, or update checker.

### 22.10 Build & release

```bash
pnpm tauri:build:mac              # aarch64-apple-darwin
pnpm tauri:build:mac-universal    # universal-apple-darwin
pnpm tauri:build:windows          # x86_64-pc-windows-msvc
pnpm tauri:build:linux            # x86_64-unknown-linux-gnu
```

Artefacts land in `src-tauri/target/release/bundle/`:
* macOS — `.dmg`, `.app`
* Windows — `.msi`, `.exe`
* Linux — `.deb`, `.AppImage`

Icons: `icons/icon.png` is the source. Run
`pnpm tauri icon icons/icon.png` before a release to fan it out
into `icon.icns` (macOS), `icon.ico` (Windows) and the various
Linux PNG sizes.

---

## 23. Deferred by design

The following are intentionally NOT implemented. Their hooks exist
(data model rows, IPC method names reserved, provider interfaces
sketched) so later work adds code without changing the architecture:

* **Bundled Python interpreter.** The packaging story supports it
  (`bundled_python_candidates` probes the right locations) but the
  bundling itself is a distribution choice, not an app-code
  concern.
* **Bundled FFmpeg sidecar.** FFmpeg is a runtime dependency
  today. `Settings › FFmpeg` accepts a custom path; the Linux
  `.deb` `depends` on it; macOS/Windows users install via Homebrew
  / winget.
* **Code-signed installers.** `macOS.signingIdentity` and Windows
  Authenticode are left `null` in `tauri.conf.json` — the
  distributing team fills them in.
* **Auto-updater.** By design not shipped. If added later it must
  be opt-in and clearly separated from offline functionality.
* **Cloud AI fallback.** Excluded by product principle — the app
  is local-first, forever.
