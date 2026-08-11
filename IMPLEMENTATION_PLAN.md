# Implementation Plan — Local Movie Translator

Ten phases, one at a time. Each phase has clear acceptance criteria and
ends with a self-review. **No phase advances automatically** — the next
phase begins only after explicit approval.

Legend: ✅ done · 🚧 in progress · ⏳ pending

---

## Phase 1 — Foundation 🚧

### Goal

Ship a runnable Tauri 2 + React + Rust + Python-worker shell that already
enforces the architecture (typed IPC, structured errors, SQLite migrations,
offline defaults, path validation) — but does no AI work yet.

### Deliverables

1. **Repo scaffolding**
   * `src/` React + TS + Vite
   * `src-tauri/` Rust + Tauri 2
   * `python/movie_translator_worker/` stdlib-only worker
   * `scripts/` helper scripts
   * `.gitignore`, `README.md`
2. **Tauri 2 shell** with a single window, three placeholder screens
   (Dashboard / Project / Settings) using React Router. No animations.
3. **Rust backend** with:
   * typed Tauri commands (§5.1)
   * SQLite via `rusqlite` (bundled) + hand-rolled migrations
   * OS-appropriate data / config / cache / log directories via `dirs`
   * path validation helpers with unit tests
   * structured `AppError` returned to the UI
   * `tracing` logger writing JSON lines to `<cache>/logs/`
   * Python worker supervisor: spawn, health-ping, restart, kill
4. **Python worker (stdlib only)**
   * JSON-RPC 2.0 line-delimited over stdio
   * Methods: `initialize`, `ping`, `env_info`, `shutdown`
   * Clean SIGTERM handling, structured logs on stderr
   * `pytest` unit tests for RPC framing and dispatch
5. **Settings**
   * `AppSettings` persisted to `<config>/config.json`
   * Defaults: `offline_mode = true`, `max_concurrent_jobs = 1`,
     `source_language = "en"`, `target_language = "vi"`, `log_level = "info"`
6. **Frontend**
   * Dashboard listing projects with name/status/updated_at, "+ New Project"
   * New-project modal → creates project via IPC
   * Project screen: reads project by id, shows metadata + a
     "Worker: running / pid / uptime" chip fed by `worker://status` events
   * Settings screen: shows resolved data/log paths + offline toggle
7. **Tests & verification** (see `Acceptance` below).

### Deliberate non-goals for Phase 1

* No video import (Phase 2)
* No FFmpeg calls (Phase 2)
* No AI models (Phases 3–6)
* No job execution (only the queue table exists)
* No installer / codesign (Phase 10)

### Acceptance criteria

* `pnpm install && pnpm tauri dev` starts the app.
* A new project can be created and appears on the dashboard.
* Killing the process and re-launching shows the same project.
* Opening the project fetches its data from SQLite via IPC.
* The Python worker starts on window open, `worker_ping` returns
  `{ pong: true, pid, uptime_ms }` in the Settings screen.
* Killing the worker externally causes Rust to restart it and emit
  `worker://status` transitions.
* `cargo test --manifest-path src-tauri/Cargo.toml` passes.
* `pytest -q` in `python/` passes.
* `pnpm build` (Vite) succeeds.
* `cargo check --manifest-path src-tauri/Cargo.toml` succeeds.
  (A full `pnpm tauri build` is not required to close Phase 1 because
   codesigning / packaging is a Phase 10 concern.)

### Self-review checklist (run at end of phase)

* [ ] No frontend value reaches a shell command
* [ ] No large binary payloads on IPC
* [ ] All paths canonicalized + validated
* [ ] Offline default respected
* [ ] Errors have `code / message / recoverable`
* [ ] No dependency is optional / unused
* [ ] Logs rotate and never leak PII
* [ ] Every SQLite write goes through a migration-tracked schema

---

## Phase 2 — Video import & audio extraction ✅

Implemented. See `ARCHITECTURE.md` §4.3 and §12 for the finished design.

### Deliverables

* `media` module: supported-extension registry, `SourceFingerprint`
  (`sha256(size ‖ first 64 KiB ‖ last 64 KiB)`), cross-platform
  `available_space` disk-check.
* `ffmpeg` module: `FfmpegService` (locate + verify + version),
  `probe_video` (ffprobe → typed `VideoMetadata`), `run_extraction`
  (real `-progress pipe:1` parsing, `SIGTERM`-based cancellation with a
  2 s grace period, partial-output cleanup).
* `jobs` module: `JobRegistry` (in-memory cancel tokens) + `JobsRepo`
  (SQLite CRUD, orphan reap on startup).
* `audio` module: `AudioExtractor` orchestrates fingerprint → cache
  lookup → probe → disk-space check → ffmpeg → cache write, and
  publishes `job://progress` / `job://update` events.
* `projects.import_media` supports `reference` (default) and `copy`.
* New DB migration `003_add_source_fingerprint` (`source_hash`,
  `source_size`, `source_modified_at`, `source_import_mode`).
* Tauri IPC commands: `get_ffmpeg_availability`, `refresh_ffmpeg`,
  `probe_media`, `import_media`, `get_project_media`, `extract_audio`,
  `cancel_job`, `list_active_jobs`.
* Tauri config: enabled `protocol-asset` + `tauri-plugin-dialog` (native
  file picker, no shell interpolation).
* Frontend: project screen with import (reference / copy), metadata
  grid, native `<video>` preview via `convertFileSrc`, extract button
  with real progress bar + cancel, cached-audio panel that survives
  app restart. Settings screen shows FFmpeg availability, version and
  configured path.

### Acceptance

Verified by the automated tests below (57 unit + 4 integration in Rust,
13 in Python). The manual acceptance flow — create project → import
`.mkv` → probe metadata → play in `<video>` → extract to
`audio/original.wav` → close app → reopen → cache detected, no
re-extraction — is covered by:

* `audio::cache::tests::round_trip`,
* `audio::cache::tests::hit_requires_matching_source_and_file`,
* `audio::cache::tests::hit_misses_on_deleted_file`,
* `projects::service::tests::imports_source_by_reference_and_persists_fingerprint`,
* `projects::service::tests::imports_source_by_copy_lands_in_project`,
* `tests/ffmpeg_roundtrip.rs::detect_probe_extract_end_to_end`,
* `tests/ffmpeg_roundtrip.rs::cancellation_terminates_ffmpeg_and_deletes_partial`.

### Non-goals for Phase 2

* Transcription, translation, TTS, subtitle editing, mixing, muxing,
  packaging (all still deferred).
* Bundled FFmpeg sidecar (planned for Phase 10 packaging; detection
  code already supports it).

---

## Phase 3 — Transcription ✅

Implemented. See `ARCHITECTURE.md` §4.4 and §13 for the finished design.

### Deliverables

* **Provider architecture** — `SpeechToTextProvider` protocol and
  transport-shaped dataclasses (`Transcript`, `Segment`, `Word`,
  `TranscribeOptions`) in `movie_translator_worker.stt`; the rest of
  the app only imports from that seam.
* **`FasterWhisperProvider`** with lazy `WhisperModel` load, in-memory
  cache across calls, and structured `ProviderError`s for every
  failure listed in the spec (whisper missing, model missing, load
  fail, invalid audio, OOM, cancel).
* **Model registry** — `tiny/base/small/medium/large-v2/large-v3
  (=large)/turbo`; only snapshots present under
  `<models_root>/whisper/<name>/` are surfaced as `installed`. No
  automatic downloads — the user has to click "Download".
* **Device detection** — `cpu` always, `cuda` when CTranslate2 reports
  a compatible device, `metal` shown as an unsupported placeholder on
  Apple Silicon. Default: `cuda` if present, else `cpu`.
* **Async RPC + notifications** — the Python worker now supports async
  handlers (background thread per request), `stt.progress` /
  `stt.download_progress` notifications, and `jsonrpc://cancel`
  control frames. Rust side gains `request_no_timeout_with_id`,
  `notify`, `cancel_request` and `subscribe`/`unsubscribe`.
* **Rust `SttService`** — orchestrates fingerprint → cache lookup →
  job register → worker call → progress forwarding → cancellation →
  transcript write → job finalise, and reuses the Phase-2 `jobs`
  registry unchanged.
* **Transcript file** — `<project>/transcription/transcription.json`
  with the schema in ARCHITECTURE §4.4. Cache key covers audio hash
  + every material option; cache hit requires cacheKey AND audio
  hash to match.
* **Tauri commands** — `get_stt_env`, `list_whisper_models`,
  `download_whisper_model`, `transcribe`, `get_project_transcript`.
  `cancel_job` is reused.
* **UI** — Project screen has a "Speech recognition" panel with
  Model / Language / Device / Word-timestamps selectors, install
  affordance for missing models, live progress + cancel, and a
  "42 segments detected" summary post-completion. Settings shows
  the worker's STT environment (whisper installed? device list?
  models root?) and per-model install status with a Download button.

### Acceptance

Verified by the automated tests below (78 Rust unit + Rust integration,
83 Python — see §Verification). The manual acceptance flow —
`movie.mp4 → audio/original.wav → faster-whisper →
transcription/transcription.json`, close app, reopen, transcript still
shown — is covered by:

* `stt::cache::tests::round_trip_preserves_fields`,
* `stt::cache::tests::hit_matches_only_when_both_match`,
* `stt::service_tests::tests::cache_key_*` (5 stability tests),
* `stt::errors::tests::*` (RPC → SttError mapping),
* Python `test_stt_handlers.py::test_transcribe_end_to_end_emits_progress_and_result`,
* Python `test_stt_handlers.py::test_transcribe_cancellation`,
* Python `test_stt_handlers.py::test_transcribe_maps_provider_error`,
* Python `test_stt_handlers.py::test_download_model_*`,
* Python `test_stt_models.py::*` (schema + timestamp validation).

### Non-goals for Phase 3

* Translation, TTS, subtitle editing, mixing, muxing, packaging.
* Bundling a Whisper model with the app (Phase 10).
* Word-timestamps in the UI — the data model supports them; the panel
  exposes the toggle but no per-word editor exists yet (Phase 5).

---

## Phase 4 — Translation ✅

Local, offline, resumable translation of the Whisper transcript with a
llama.cpp-hosted GGUF model. English → Vietnamese is the MVP pair; the
architecture and UI treat languages as codes so extending is a
one-line change.

* **Provider abstraction** — `TranslationProvider` protocol in
  `movie_translator_worker.translation.provider`. The only concrete
  implementation is `LlamaCppTranslationProvider`; nothing in the
  service or UI imports `llama_cpp` directly.
* **Runtime** — `llama-cpp-python` (lazy import, gracefully degrades
  when missing). Ollama is dev-only, never a production dependency.
* **Model management** — user-installed `*.gguf` files under
  `<app_data>/models/translation/`. `translate.list_models` scans the
  directory; the first hit is the default. No auto-download.
* **Versioned prompt** — `translation_prompt_v1` in
  `movie_translator_worker.translation.prompts`. Baked into the cache
  key so bumping doesn't invalidate the cache until the user
  explicitly re-runs.
* **Structured output** — chat completion with
  `response_format={"type":"json_object"}`. The provider strictly
  validates every requested `id` is present; extra ids are dropped;
  `id`/`start`/`end` are never sent back to the model.
* **Chunking + context** — window of ~30 segments, with configurable
  `context_before`/`context_after` scene padding. Only the chunk is
  returned; context is read-only for the LLM.
* **Incremental persistence** — every completed chunk is streamed
  back to Rust via a `translate.chunk_completed` notification and
  written to `translation/translation.json` atomically. A crash
  mid-run leaves the completed portion intact.
* **Progress** — `translate.progress` notifications carry
  `completedSegments / totalSegments`; Rust forwards them as
  `job://progress` events with real fractions.
* **Cancellation** — cooperative cancel between chunks; the worker
  raises `ProviderCancelled` and the service marks the job cancelled
  without touching already-persisted segments.
* **Cache & resume** — cache key is
  `sha256(transcript_key, audio_hash, model, prompt_version, source
  lang, target lang, chunk/context/temperature/top_p/max_tokens)`.
  Full-hit returns synchronously; partial hits resume from the last
  translated segment. User edits (`edited: true`) survive across
  option changes.
* **Editor** — inline per-segment textarea in the Project screen.
  Every blurred change calls `update_translation_segment`, which
  writes the row atomically and flips `edited: true` so subsequent
  LLM runs won't clobber it.
* **Errors** — structured `TranslationError` variants mapped to UI
  codes: `TRANSLATE_MODEL_NOT_INSTALLED`,
  `TRANSLATE_LLAMA_NOT_INSTALLED`, `TRANSLATE_MODEL_LOAD_FAILED`,
  `TRANSLATE_INVALID_JSON`, `TRANSLATE_INCOMPLETE_RESPONSE`,
  `TRANSLATE_OUT_OF_MEMORY`, `TRANSLATE_LLM_FAILURE`,
  `TRANSLATE_WORKER_CRASH`, `TRANSLATE_CANCELLED`.
* **Acceptance:** `transcription.json` → `translation.json`, resume
  after quit continues from the last completed chunk, edits survive
  a re-run, no network access, no bundled model, timestamps
  unchanged.

### Non-goals for Phase 4

* Automatic model download (user installs models by hand).
* Speaker/gender/glossary hints beyond the versioned prompt
  (extensible via a new `translation_prompt_vN`).
* Multi-model ensembling or draft-then-refine passes.
* Editor UX beyond per-segment textareas (segment split/merge lives
  in Phase 5).
* Bundling llama.cpp binaries (Phase 10 packaging concern).

---

## Phase 5 — Subtitle editor ✅

**Implemented:**

* Canonical subtitle model (`SubtitleSegment`: `id, start, end,
  sourceText, translatedText, speaker?, voiceId?`) persisted at
  `<project>/subtitles/subtitles.json` — a single source of truth
  for every downstream stage.
* Rust `subtitles` module (models, cache, derive, srt, ass,
  service, errors) with zero worker/Python involvement.
* Derivation from transcript (+ optional translation) with a
  merge-preserving-edits path so re-running "Build subtitles"
  never clobbers hand-edits.
* IPC surface (11 Tauri commands): `get_project_subtitles(_doc)`,
  `rebuild_project_subtitles`, `update_subtitle_segment`,
  `add_subtitle_segment`, `delete_subtitle_segment`,
  `split_subtitle_segment`, `merge_subtitle_segment`,
  `clear_subtitle_dirty`, `import_subtitles`, `export_subtitles`.
* SRT importer + writer (`HH:MM:SS,mmm` timing, inline-tag
  stripping, missing-index tolerance).
* Lightweight ASS/SSA importer + writer (parses `[Events]`
  `Dialogue:` rows, strips `{...}` overrides and translates
  `\N`/`\n`/`\h`; export emits a fixed `[Script Info]` +
  `[V4+ Styles]` header with a single Default style so downstream
  tooling still recognises the file).
* Structured errors: `SUBTITLE_NO_TRANSCRIPT`,
  `SUBTITLE_NO_DOCUMENT`, `SUBTITLE_SEGMENT_NOT_FOUND`,
  `SUBTITLE_INVALID_TIMING`, `SUBTITLE_INVALID_SPLIT`,
  `SUBTITLE_NO_MERGE_TARGET`, `SUBTITLE_INVALID_FILE`,
  `SUBTITLE_UNSUPPORTED_FORMAT`, `SUBTITLE_INVALID_EXPORT_PATH`,
  plus IO/DB fallbacks — all recoverable, all with UI hints.
* Dependency tracking: any edit to `translatedText`, `start`,
  `end`, `speaker`, or `voiceId`, plus add/delete/split/merge,
  sets `dirty.tts = dirty.mix = dirty.render = true`. Edits to
  `sourceText` alone don't dirty anything. The doc exposes
  `clear_dirty` for Phase 6+ to call after a successful pass.
* React frontend:
  * Zustand state (`currentSubtitleDoc`) mirrors the on-disk doc;
    every mutation replaces it wholesale to avoid drift.
  * `SubtitlePanel` with summary badges (segment/translated/
    speaker/overlap counts, dirty banner), toolbar (filter, add
    row, rebuild, import, export with format + kind selectors).
  * Hand-rolled virtualized `SubtitleList` (fixed row height +
    overscan) — no react-window dependency, handles thousands of
    rows without dropping frames.
  * `SubtitleRow` inline editor: HH:MM:SS.ms timecodes with blur-
    to-commit, source + translation textareas, speaker + voice
    inputs, split-at-playhead / merge-next / delete actions.
  * `<video>` overlay renders the active segment's translated
    text (with source fallback) — plain positioned `<div>`, never
    burned into pixels (Phase 9 is the render stage).
  * Native "Open subtitle file" / "Save subtitle" dialogs via
    `tauri-plugin-dialog` (already wired for the video picker).
* Persistence: atomic write on every mutation (`tmp + rename`)
  through the existing project storage layout — no new database
  tables required (subtitles live in the project folder).
* **Acceptance:** subtitle → JSON round-trip preserves ids and
  timing; SRT/ASS round-trip preserves timing to within one
  quantum (1 ms / 10 ms). Frontend `pnpm build` passes cleanly;
  `cargo test subtitles` covers derivation, SRT/ASS
  parse+write, id allocation, dirty flags, overlap detection.

**Non-goals (Phase 5):**

* Full ASS style fidelity — we emit and honour a single Default
  style; users needing complex styling should keep their original
  ASS as the render input (Phase 9 will support that).
* NLE-style multi-track timeline. The spec explicitly asks for a
  lightweight list + overlay.
* Auto-fix of overlapping segments. Overlaps are surfaced as a
  counter; the user decides.
* Frame-accurate scrubbing beyond the HTML5 video element's own
  `currentTime` precision. Frame stepping arrives with a dedicated
  transport control in a later polish pass.

---

## Phase 6 — Local TTS ✅

**What Phase 6 delivers**

Convert the translated dialogue in `subtitles/subtitles.json` into
per-segment WAV files under `voices/`, driven by a pluggable
`TTSProvider` abstraction. Piper is the first concrete backend
because it satisfies every ranked constraint from the spec (offline,
CPU/Apple-Silicon friendly, small RAM, permissive MIT license, and
has good Vietnamese voices), but the pipeline never imports Piper
directly — everything goes through `TTSProvider`.

**Provider architecture**

* Python: `movie_translator_worker.tts.provider.TTSProvider` (a
  runtime-checkable protocol) plus `PiperTTSProvider`. The provider
  contract is deliberately minimal: `get_voices`, `synthesize`,
  `unload`. Every foreseeable failure is raised as `ProviderError`
  with a stable code so the host can map it to `RpcErrorCode` and
  therefore to a structured `AppError`.
* Rust: `src-tauri/src/tts` owns the orchestration surface
  (`TtsService`), the manifest (`voices/voices.json`), the error
  mapping, and the IPC contract. The service does **not** know
  anything Piper-specific; it just talks to the worker.

**Voice registry**

* Layout: `<models>/tts/<engine>/<voice_id>/…`. Piper wants an
  `.onnx` and an `.onnx.json` in that directory; an optional
  `voice.json` can override the display metadata (name, gender,
  quality tag) without editing the model files.
* The registry is filesystem-driven — installing a voice is
  "drop a folder in place, click **Rescan**". No models are ever
  bundled in the executable, and nothing is auto-downloaded.

**Cache identity**

* The manifest key is `sha256(engine | voice_id | model_name |
  text_hash | speed | pitch | volume)`, mirrored bit-for-bit in
  Python (`build_segment_cache_key`) and Rust (same name).
* A row is *generated* only when every dimension still matches;
  a text edit, a slider tweak, or a voice swap silently reclassifies
  the row as *stale* until it's regenerated.
* Because the cache lives on disk as `voices/voices.json`, closing
  and reopening the project preserves progress; cancelled or crashed
  runs simply resume from the missing segments.

**Preview vs. batch generation**

* `preview_tts_segment` is a synchronous RPC — it hits the manifest
  first, and only re-invokes the engine if the cache entry is stale
  or absent. Result is played back with an HTML `<audio>` element
  from an `asset://` URL, so no PCM ever crosses the IPC boundary.
* `generate_tts` starts a background job (registered with the same
  `JobRegistry` used by STT and translation). The Python side emits
  a `tts.segment_completed` notification per file, which Rust folds
  into the manifest and forwards to the frontend so the UI updates
  incrementally. `progress` events drive the "Synthesising 42/350"
  counter.

**Dependency tracking**

* `SubtitleDoc.dirty.tts` is set whenever the user touches a field
  the cache key depends on (`translatedText`, `voiceId`, `speed`,
  `pitch`). After a successful batch, `TtsService::maybe_clear_dirty`
  only clears the flag once every non-empty segment has a matching
  manifest entry — partial coverage keeps it dirty.

**Model lifecycle**

* `PiperTTSProvider` lazy-loads the ONNX model on the first
  `synthesize` call and releases it via `unload()` at the end of
  every batch, so no idle worker holds a TTS model in RAM. The
  provider also picks the fastest backend it can find (Python
  bindings when installed, `piper` CLI otherwise), which keeps
  the runtime dependency optional.

**Errors**

* Every TTS failure surfaces through the existing structured error
  system: `TTS_ENGINE_UNAVAILABLE`, `TTS_VOICE_MISSING`,
  `TTS_MODEL_INVALID`, `TTS_INVALID_TEXT`, `TTS_ENGINE_FAILURE`,
  `TTS_OUT_OF_MEMORY`, `TTS_DISK_FULL`, plus `CANCELLED` when the
  user hits stop. `AppError` conversions include actionable hints
  (e.g. "install piper-tts in the worker environment").

**Acceptance:** subtitle → wav, offline, cached per segment, dirty
flags cleared only at full coverage. `pnpm build`, `cargo test tts`,
and `pytest` all pass.

---

## Phase 7 — Voice synchronisation ✅

**What Phase 7 delivers**

* Per-segment timing-aligned WAVs written to
  `voices/synced/000001.wav … 000NNN.wav`, plus a canonical
  `voices/synced/sync.json` manifest (atomic writes).
* Python worker `sync` module — pure planner
  (`sync/planner.py`) + FFmpeg apply layer
  (`sync/ffmpeg_apply.py`) built on `atempo` (chained for extreme
  ratios), `apad` for trailing silence, and Python's `wave` for the
  empty-window case. Streams file paths only — no in-memory audio.
* JSON-RPC handlers (`sync.env`, `sync.apply_one`,
  `sync.apply_batch`) with `sync.progress` and
  `sync.segment_completed` notifications for incremental UI updates
  and cancellation support.
* Rust `SyncService`, `SyncCacheFile`, `SyncError`, and typed
  `SyncEnv` / `SyncSettings` / `SyncManifest` / `SyncSummary`
  models. Tauri commands: `get_sync_env`, `get_project_sync_summary`,
  `get_project_sync_manifest`, `preview_sync_segment`, `apply_sync`.
* Cache identity keyed on
  `sha256(ttsCacheKey || round_ms(targetDuration) || min/maxSpeed || sampleRate || channels)`
  so a hit means "produced from this exact combination", and any
  drift triggers a fresh FFmpeg run.
* `JobStage::Sync` added to the job registry / SQLite persistence /
  event bus so the pipeline UI and cancellation work uniformly with
  the other stages.
* Granular dirty tracking in `SubtitleService`: content edits
  (`mark_content_dirty`) invalidate TTS + sync + mix + render, but
  **timing-only edits** (`mark_timing_dirty`) invalidate sync + mix
  + render only — transcription and translation stay clean, and
  Phase 6 TTS WAVs are reused. `clear_dirty_flags(project_id, mask)`
  lets each service clear only its own bit.
* Frontend: `SyncPanel` (env, coverage, fit breakdown, min/max
  speed sliders, mono/stereo, actions, progress, preview player),
  per-row `sync-badge` with the required states
  (`✓ Fits`, `⚠ Adjusted`, `⚠ Too long`, plus `⚠ Outdated` /
  `⚠ Not synced` / `· Empty`), and a "▶ Preview synced" /
  "Resync" pair next to the existing TTS row actions.
* Docs: `ARCHITECTURE.md` §17 covers the full Phase 7 model.

**Acceptance:** all segments produce a synced WAV or are flagged
`too_long`; timing-only edits invalidate sync/mix/render without
regenerating transcription, translation, or TTS; per-row badges
match the spec (`✓ Fits`, `⚠ Adjusted`, `⚠ Too long`); the
`voices/synced/` output is per-segment (never a concatenated
stream), so the eventual mix stage places each WAV at its absolute
`subtitle.start` and inter-subtitle silences are preserved by
construction. `pnpm test` (tsc), `cargo test`, and `pytest` all
green.

---

## Phase 8 — Audio mixing ✅

* `mix` module lives entirely on the host (Rust) — audio mixing is
  pure FFmpeg, so no Python worker involvement. `MixService` spawns
  one ffmpeg per generate, parses `-progress pipe:1`, and honours
  cancellation through the shared `JobHandle` token.
* Voice timeline: every synced WAV is placed at its absolute
  `subtitle.start` via `adelay=<ms>|<ms>`. No concatenation, so
  inter-subtitle silences are preserved by construction. `SyncStatus::Empty`
  segments are skipped (silence + anything = anything).
* Original audio: read straight from the source video's first audio
  stream. No source-separation MVP; the argv builder is factored so a
  future dialogue-separator can slot in as an extra input.
* Configurable **Original Volume** (default 70%) and **Voice Volume**
  (default 100%) via the mix panel; each is a linear FFmpeg `volume`
  filter and folded into the cache key.
* Optional **ducking** via `sidechaincompress` keyed by the merged
  voice track. Depth / threshold / attack / release are all
  configurable and part of the cache key.
* Output: a single WAV at `audio/mixed_vi.wav`, PCM16, stereo 44.1 kHz
  by default (channels / sample rate are configurable).
* `audio/mix.json` manifest tracks the cache identity (source
  fingerprint + voice sync cache keys + volume + ducking) so a re-run
  with the same inputs is a no-op.
* Frontend: `MixPanel` with sliders + ducking toggle + Play button
  and a "Mixing" progress row in the pipeline panel. Volume, ducking,
  voice, or timing changes invalidate `dirty.mix` and (transitively)
  `dirty.render`; transcription / translation are untouched.
* **Acceptance:** mixed audio file plays cleanly with Vietnamese
  dialog on top; `cargo test --lib mix::` covers cache-key stability,
  argv shape (with & without ducking, single vs. batched voices),
  voice-input collection, and dirty-flag masking. `pnpm tsc`,
  `cargo test`, and `pytest` all green.

---

## Phase 9 — Final video rendering ✅

* `render` module lives entirely on the host (Rust) — final muxing
  is pure FFmpeg, so no Python worker involvement. `RenderService`
  spawns one ffmpeg per render, parses `-progress pipe:1`, and
  honours cancellation through the shared `JobHandle` token.
* **Video**: `-c:v copy` by default so the source video is
  stream-copied byte-for-byte (fastest, lossless, no CPU cost).
  Re-encoding is only used when the subtitle mode requires it
  (`Burned` silently promotes `Copy → libx264 (preset medium, CRF
  20, yuv420p)`) or when the user explicitly picks a re-encode codec.
* **Audio**: always re-encoded from `audio/mixed_vi.wav` via the
  configured codec (default `aac @ 192k`, also `libopus / ac3 / mp3`).
  Source's original audio track is intentionally dropped so the
  file has exactly one Vietnamese-dubbed audio stream.
* **Subtitle modes** — `External` writes the Vietnamese SRT next to
  the movie (`output/movie_vi.srt`) and stream-copies the video;
  `Burned` bakes them into the pixels via FFmpeg's `subtitles=…`
  filter; `None` skips subtitles entirely. Path escaping for the
  filter is unit-tested (`build_subtitles_filter`).
* **Output settings** — `RenderSettings` exposes only what the spec
  asks for: output format (`mp4` / `mkv`), video codec (`copy` or a
  re-encode codec name), audio codec + bitrate, subtitle mode, and
  optional custom absolute output path. Low-level ffmpeg knobs stay
  hidden.
* **Progress** — real ffmpeg `-progress pipe:1` output surfaces on
  `job://progress` throttled to 100 ms; the UI `Render` row shows
  the actual encoder progress fraction.
* **Cancellation** — `JobHandle::cancel` fires SIGTERM, ffmpeg exits,
  the partial output file is removed (no zombie processes, no
  half-written movies dressed up as successes).
* **Validation** — every successful render is re-probed with
  `ffprobe`: file exists, size > 0, duration > 0, at least one
  video + audio stream. Any mismatch removes the output and fails
  the job with a stable code (`RENDER_OUTPUT_INVALID` /
  `RENDER_VALIDATION_FAILED`).
* **Source protection** — the original video is opened read-only;
  outputs always land in `<project>/output/` (default) or a
  user-picked absolute path; relative paths are refused.
* **Dependency tracking** — `dirty.render` is set whenever subtitle,
  translation, TTS, sync, mix, or render settings change; cleared
  only after a successful render pass via
  `DirtyFlags::only_render()`. No unaffected upstream stage is
  re-run.
* **Frontend** — `RenderPanel` with subtitle-mode dropdown, output
  format / video codec / audio codec pickers, output-path picker,
  render / cancel / regenerate buttons, and a "Rendering" row in the
  Pipeline panel that mirrors ffmpeg's own progress.
* **Acceptance:** the final movie plays back with Vietnamese audio
  and either a burned-in or sidecar SRT; `cargo test --lib render::`
  covers cache-key stability, argv shape (external / burned / none,
  mp4 vs mkv, faststart), path-escaping, summary contract, and
  dirty-flag masking. `pnpm tsc`, `cargo test`, and `pytest` all
  green.

---

## Phase 10 — Local Model Manager & Offline-First Architecture ✅

Delivered a centralised, offline-first model manager that treats AI
models as external assets — nothing is bundled into the executable,
nothing is downloaded silently, everything runs locally after install.

### Rust (`src-tauri/src/models/`)

* `mod.rs` — module entry; re-exports `LocalModel`,
  `ModelDirectoryInfo`, `ModelKind`, `ModelStatus`, `ModelRegistry`,
  `ImportSpec`, `ImportStrategy`, `ModelManagerError`.
* `registry.rs` — `ModelRegistry` aggregates the per-stage worker
  registries (Whisper via `stt.list_models`, GGUF via
  `translation.list_models`, voices via `tts.list_voices`) into a
  single flat `Vec<LocalModel>`. In-memory cache with explicit
  `rescan()` + `invalidate()`. Lightweight validators
  (`validate_whisper_dir`, `validate_gguf_file`, `validate_voice_file`)
  check existence + required files + non-zero size — never load a
  model. Also exposes `probe_writable(dir)` used when the user
  changes the models directory.
* `import.rs` — `import_local_model(models_root, spec)` places a
  user-picked source under the correct subdirectory. Two strategies
  (`Link` default, `Copy` fallback); symlink helpers are gated by
  `#[cfg(unix)]` / `#[cfg(windows)]` and gracefully fall back to a
  full copy when Windows refuses. Never overwrites — returns
  `MODEL_ALREADY_EXISTS`.
* `errors.rs` — `ModelManagerError` with stable codes
  (`MODEL_UNSUPPORTED_KIND`, `MODEL_INVALID_SOURCE_PATH`,
  `MODEL_SOURCE_NOT_FOUND`, `MODEL_UNSUPPORTED_SOURCE`,
  `MODEL_MISSING_REQUIRED_FILE`, `MODEL_UNREADABLE`,
  `MODEL_INVALID_NAME`, `MODEL_ALREADY_EXISTS`,
  `MODEL_PERMISSION_DENIED`, `MODEL_DIR_NOT_WRITABLE`,
  `MODEL_NETWORK_DISABLED`, `MODEL_IO`). Mapped to `AppError`
  with UI-friendly hints (e.g. Offline-mode explainer for
  `NetworkDisabled`).

### Wiring

* `AppState.models: ModelRegistry` created in `bootstrap()`, sharing
  the existing `stt`/`translation`/`tts` service handles.
* `AppState::effective_models_dir()` — single source of truth for
  the resolved models directory (`models_dir_override` in settings
  wins, otherwise `AppPaths::models_dir`). Used by
  `bootstrap()`, `WorkerConfig`, `commands::get_app_info`,
  `commands::get_model_directory`, `commands::import_local_model`.
* `WorkerSupervisor::reinitialize_models_root(path)` — re-runs the
  `initialize` RPC so a live `set_model_directory` change takes
  effect without an app restart. Best-effort; on failure we log and
  ask the user to restart.
* `config::AppSettings` — new `models_dir_override: Option<String>`
  and `first_run_completed: bool` fields, with matching
  `AppSettingsPatch` entries that trim empty strings to `None`.
* `db::migrations` — migration `004_add_project_model_config` adds
  `whisper_model`, `translation_model`, `tts_engine`, `tts_voice_id`
  to `projects` (all nullable, no defaults).
* `db::models::ProjectRecord` — same four optional fields, camelCase
  on the wire. New `ProjectModelPatch` for partial updates.
* `db::Db::set_project_models(id, patch)` — targeted UPDATE that
  bumps `updated_at`.
* `projects::service::update_models` — validates the id, writes
  the DB, refreshes the `project.json` mirror.

### Tauri commands (`commands.rs` + `lib.rs`)

* `list_local_models` — cached read.
* `rescan_local_models` — force refresh.
* `get_model_directory` / `set_model_directory` — read + write the
  effective directory (with absolute-path + writable probe on set).
* `import_local_model` — `spawn_blocking` wrapper around the
  importer.
* `unload_all_models` — proxies to `TtsService::unload_all`
  (Whisper and llama.cpp already release state when idle).
* `update_project_models` — persists per-project selection.
* `download_whisper_model` — now hard-fails with
  `MODEL_NETWORK_DISABLED` when `offline_mode == true`, so the only
  network path in the app is honestly gated.

### Frontend

* `src/ipc/types.ts` — `LocalModel`, `ModelKind`, `ModelStatus`,
  `ModelDirectoryInfo`, `ImportModelSpec`, `ImportStrategy`,
  `ProjectModelPatch`; extended `AppSettings` with
  `modelsDirOverride` + `firstRunCompleted`; extended `Project`
  with `whisperModel` / `translationModel` / `ttsEngine` /
  `ttsVoiceId`.
* `src/ipc/bridge.ts` — seven new API calls plus
  `pickDirectory(title)` and `pickModelSource(directory, filters)`
  wrappers over the Tauri dialog plugin.
* `src/state/store.ts` — `localModels`, `localModelsLoading`,
  `modelDirectory`, `modelImportBusy`; actions
  `refreshLocalModels`, `refreshModelDirectory`, `setModelDirectory`,
  `importLocalModel`, `unloadAllModels`, `updateProjectModels`,
  `markFirstRunCompleted`. `startTranscribe`/`startTranslate`/
  `startGenerateTts` auto-persist the chosen model per project.
* `src/screens/Settings.tsx` — new **AI Models** panel: model
  directory row (Change… / Reset to default), filter dropdown,
  aggregated model table with per-kind badges + status pills +
  "invalid" hints, **Scan Models**, **Add Local Model** (kind
  picker, Browse…, name override, Link import), **Unload all**.
* `src/screens/Dashboard.tsx` — dismissible **first-run banner**
  that reports "N local models installed" (or nudges the user to
  Settings when none), never triggers a download, disappears once
  the user clicks Got it.
* `src/styles/global.css` — `.first-run-banner`, `.model-dir-row`,
  `.model-filter`, `.model-table`, `.model-row`, `.model-badge`
  (per-kind colours), `.status-badge` (ok/warn/err), and
  `.import-model-form` styles.

### Documentation

* **ARCHITECTURE.md** — new §20 *Local Model Manager & Offline-First
  (Phase 10)* documenting the directory layout, registry contract,
  import strategies, offline gate, override + first-run flow,
  per-project config, lifecycle/memory story, front-end surface,
  and performance guarantees. "Deferred by design" bumped to §21.
* **IMPLEMENTATION_PLAN.md** — this section.
* **LICENSES.md** — clarified that Phase 10 introduces no new
  runtime dependencies; the sole network entrypoint (Whisper
  download via `huggingface_hub`) is now gated by Offline Mode.

### Verification

* `cargo fmt --check` — clean
* `cargo clippy -D warnings` — clean
* `cargo test --lib` — all pre-existing tests still green
* `pnpm tsc --noEmit` + `pnpm build` — clean
* `pytest` — unchanged (Phase 10 is a Rust/UI aggregation; the
  Python-side per-stage registries were not touched)

---

## Phase 11 — Performance, RAM & CPU optimization ✅

Focused on making the app *cheap* — smaller working set, less IPC
noise, faster startup — without touching feature scope. Every
intervention started from an inspection and targeted a concrete
bottleneck (no speculative rewrites).

**IPC & memory**

* Debounced the `translation.chunk_completed`,
  `tts.segment_completed` and `sync.segment_completed` handlers in
  `state/store.ts` at 350 ms per project. Previously a chunk-heavy
  translation run triggered one full doc reload per chunk (up to
  100 IPC round-trips carrying the full doc); now it coalesces to
  a handful.
* `TranslationEditor` in `screens/Project.tsx` is now virtualised
  (same windowed strategy `SubtitleList` uses since Phase 5). A
  5 000-segment movie renders ~20 textareas at a time instead of
  5 000.
* React state continues to hold only summaries, snapshots and
  live progress; large artefacts (video, audio, transcripts,
  manifests) remain on disk and are streamed through
  `convertFileSrc`.

**Python worker throttling**

* `HandlerContext.emit_progress` in `python/rpc.py` now coalesces
  `*.progress` notifications at 20 Hz per method with a 1%
  fraction-jump escape hatch. Semantic events
  (`*.chunk_completed`, `*.segment_completed`) always pass so the
  UI still refreshes per-chunk / per-segment.

**Source fingerprint reuse**

* `commands::get_project_media` no longer re-fingerprints the
  source video on every job progress tick. The fingerprint on
  `ProjectRecord` is trusted while the file's stat matches; only
  a real content change triggers a fresh `fingerprint_file` read
  (still the cheap 128 KiB sentinel-window hash — never a full
  hash).

**Model lifecycle**

* Added `stt.unload` / `translate.unload` RPC methods and
  matching `SttService::unload` / `TranslationService::unload`
  wrappers on the Rust side (TTS already had `tts.unload`).
* `unload_all_models` now genuinely unloads Whisper, llama.cpp
  and every Piper voice — not just TTS.
* New `unload_stage_models(stage)` command lets the frontend free
  one stage's model without disturbing another warm one.
* `state/store.ts` schedules a per-stage auto-unload
  `autoUnloadAfterSecs` seconds after the last job of that stage
  settles; a follow-up job of the same stage cancels the pending
  unload. Default is 90 s; setting it to 0/null disables the
  timer entirely.

**Advanced performance settings**

* New `AppSettings` fields: `autoUnloadAfterSecs`, `cpuThreads`,
  `gpuAcceleration`. Rust patches, worker config and Python
  `handlers._PERF` all propagate the values.
* `WorkerSupervisor::reinitialize_perf` reuses the existing
  `initialize` handshake to push perf changes live — providers
  reload their model on the next call so users don't need to
  restart the app.
* `LlamaCppTranslationProvider` reads `n_gpu_layers=-1` when
  `gpuAcceleration` is on (Metal / CUDA offload where the build
  supports it) and honours the user-supplied `cpu_threads`.
* `FasterWhisperProvider` honours the user's `cpu_threads` and
  respects `gpuAcceleration=false` by downgrading automatic
  device selection to CPU.

**Startup**

* Bootstrap remains lightweight — no models loaded, no
  recursive scans, no ffmpeg probe on the critical path.
* Added a best-effort recursive sweep (max depth 8) over
  `<projects>/**/*.tmp` in the background so orphaned atomic-write
  sidecars from a crashed previous run don't accumulate. The
  sweep never delays the UI.

**Runtime monitor**

* New `get_runtime_stats` command exposes `activeJobs`,
  `activeProjects`, `hostRssBytes`, `workerRssBytes`,
  `workerUptimeSecs`. On Linux we read `/proc/<pid>/statm`; on
  macOS / Windows we shell out to `ps -o rss=`. No `sysinfo`
  dependency; no continuous background polling on the Rust side.
* Settings › Performance polls it every 3 s so the user can see
  worker RAM shrink after auto-unload.

**Dependencies**

* No new crates, no new npm packages, no new pip packages. The
  `release` profile in `Cargo.toml` (LTO + strip + `codegen-units
  = 1` + `panic = "abort"`) is unchanged and already optimal.

**Docs**

* `ARCHITECTURE.md §21` added — Phase 11 performance rules, IPC
  discipline, model lifecycle, auto-unload policy, orphan sweep,
  runtime monitor. "Deferred by design" moved to §22.
* `IMPLEMENTATION_PLAN.md` — this section.
* `LICENSES.md` — no new components; Phase 12 markers updated.

**Verification**

* `cargo fmt --check` — clean
* `cargo clippy --lib -- -D warnings` — clean
* `cargo test --lib` — 169 passed (existing suite, no new tests
  added per the phase brief)
* `pnpm tsc --noEmit` + `pnpm build` — clean
* `pytest` — 49 passed (Python throttler + unload handlers
  covered by existing async / registry tests)

---

## Phase 12 — Production build, packaging & release ✅

### Goal

Turn the running dev app into a lightweight production desktop
application — offline-first, self-contained, cross-platform ready,
prioritising macOS Apple Silicon. No new product features; only
packaging, reliability, security, and release configuration.

### What shipped

* **Tauri production config**
  * `tauri.conf.json` — `bundle.targets = "all"`, explicit
    `productName`, publisher, copyright, category, short/long
    description; `fileAssociations` for `.mp4`/`.mkv`/`.mov`/`.m4v`/
    `.webm`/`.avi` (Viewer) and `.srt`/`.ass`/`.ssa` (Editor).
  * `resources: [python worker sources]` so the packaged app carries
    the `movie_translator_worker` package next to the executable.
  * Per-platform bundle blocks: macOS `minimumSystemVersion: 11.0`,
    Windows `downloadBootstrapper` for the WebView2, Linux `.deb`
    depends line (`libwebkit2gtk-4.1-0`, `libssl3`, `ffmpeg`,
    `python3`) and non-media-framework AppImage.
  * `withGlobalTauri: false` — the frontend uses the ES modules
    directly, no `window.__TAURI__` shim.
  * Window `label: "main"`, `dragDropEnabled: true`,
    `acceptFirstMouse: true`.

* **Rust release profile (`Cargo.toml`)**
  * `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`,
    `strip = true`, `opt-level = 3`, `debug = false`. Kept
    `opt-level = 3` (not `z`/`s`) because Rust still does non-trivial
    JSON parsing, hashing and cache computation on hot paths.
  * New `release-with-debug` profile mirrors release but keeps line
    tables for opt-in symbolicated crash reports.

* **Least-privilege capabilities**
  * `capabilities/default.json` restricted to `core:default`,
    `core:event:allow-listen`, `core:event:allow-unlisten`,
    `dialog:allow-open`, `dialog:allow-save` — no filesystem, shell,
    or window permissions.
  * CSP already scoped to `self` + `ipc:` + `asset:`; the frontend
    never talks to any origin except the Tauri IPC bridge.

* **Deterministic Python worker location**
  * `detect_python_bin()` now probes bundled interpreters first:
    `<bundle>/Contents/Resources/python-embed/bin/python3`,
    `<exe>/python-embed/bin/python3`, then `python-embed/python.exe`
    on Windows, then `LMT_PYTHON` env var, then `PATH`.
  * `detect_worker_root()` prefers `<bundle>/Contents/Resources/python`
    when running from a macOS app bundle, then walks up from the
    executable for AppImage / portable layouts.
  * `WorkerError::Spawn` is now classified: `WORKER_PYTHON_MISSING`
    with an install hint when the interpreter isn't found;
    `WORKER_SPAWN_DENIED` for permission errors.

* **Logs & log rotation** (`logging.rs`)
  * Daily rotation with a 14-day retention window; startup calls
    `prune_old_logs` so a long install never grows unbounded.
  * `clear_active_logs` truncates active files (doesn't unlink)
    so the tracing appender's file handle stays valid.
  * Documented promise: `tracing` calls elsewhere log method names,
    timings and counts only — never transcripts, prompts, subtitle
    text, or file contents.

* **Storage / cache / log surface**
  * New commands: `open_app_path`, `get_storage_stats`,
    `clear_cache`, `clear_logs`, `list_orphaned_jobs`.
  * `PathKind` enum on the Rust side restricts the frontend to
    named app-owned roots — passing an arbitrary path returns
    `STORAGE_UNKNOWN_PATH_KIND`.
  * `StorageStats` computed with a bounded walk
    (`MAX_WALK_DEPTH = 6`) so a pathological tree can't hang the UI.
  * `clear_cache` never touches the log subdir, per-project data,
    models or rendered outputs.
  * New **Settings › Storage & Logs** panel: per-root size, "Open"
    (reveals in OS file manager), "Clear cache", "Clear logs".

* **Crash recovery UX**
  * `reap_orphans` (Phase 1) already marks incomplete jobs on
    startup. Phase 12 adds `list_orphaned_jobs` + a Dashboard
    banner ("Some jobs didn't finish last time.") linking to each
    affected project, so the user can resume where they left off
    (completed segments are always preserved).

* **User-friendly errors**
  * `AppError.hint` was already threaded through every service; the
    Settings/Dashboard `errorMessage()` helpers now render it
    below the message so users see actionable next steps instead
    of a bare code.
  * `WORKER_PYTHON_MISSING` hint points at the install docs;
    `STORAGE_OPEN_FAILED`, `STORAGE_UNKNOWN_PATH_KIND` and
    `MODEL_DIR_NOT_ABSOLUTE` all carry hints.

* **Icons & file associations**
  * `tauri.conf.json` `bundle.fileAssociations` maps `.mp4`, `.mkv`,
    `.mov`, `.m4v`, `.webm`, `.avi`, `.srt`, `.ass`, `.ssa` to the
    app. Existing `icons/icon.png` is the source; run
    `pnpm tauri icon icons/icon.png` before release to fan it out
    into per-platform variants.

* **Docs**
  * New root-level `README.md` — system requirements, install
    (FFmpeg + Python + models), storage locations table, project
    layout, supported formats, offline promise, troubleshooting,
    build-from-source, security & privacy.
  * `ARCHITECTURE.md` — new **§22 Production build, packaging &
    release** covering everything above; "Deferred by design"
    moved to §23.
  * `LICENSES.md` — Phase 12 markers bumped to shipped; Python
    interpreter treated as a runtime dependency, not a bundled
    binary; FFmpeg treated the same way with a note on the Linux
    `.deb` `depends`.
  * `package.json` — added `productName`, `license`, `homepage`,
    `keywords`, `typecheck` alias, `tauri:build:mac`,
    `tauri:build:mac-universal`, `tauri:build:windows`,
    `tauri:build:linux` scripts.

* **No new mandatory network paths**
  * Grepped the tree — no `fetch`, `axios`, `WebSocket`, analytics,
    telemetry, or update-check code was added or exists in the
    production paths. Offline Mode is on by default; turning it off
    does not enable any downloader today.

### Not in scope (deferred)

* Signed installer identities (`macOS.signingIdentity`, Windows code
  signing) — placeholders left `null`; team must fill in when
  distributing.
* Auto-updater — the design forbids a mandatory one; if added later
  it must be opt-in and clearly separated from offline functionality.
* Bundled Python interpreter — the packaging story supports it
  (`bundled_python_candidates` probes the right paths) but the
  bundling itself is a distribution-time choice, not an app-code
  concern.

### Verification

* `cargo fmt --check` — clean
* `cargo clippy -- -D warnings` — clean
* `cargo test --lib` — all tests pass
* `pnpm build` / `pnpm typecheck` — clean
* `pytest -q python` — all tests pass

