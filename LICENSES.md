# Third-party licenses — Local Movie Translator

This file lists every third-party runtime component the application will
depend on. It is updated at the end of every phase and re-audited before
release. **Nothing may enter the shipped artefact without appearing here
first.**

Legend: **Runtime** = shipped in the installer or required at runtime.
**Dev-only** = only present in developer or CI environments.

---

## 1. Phases 1–6 — currently declared

### Runtime

| Component                          | Version           | License                         | Ship?             | Notes                                                  |
| ---------------------------------- | ----------------- | ------------------------------- | ----------------- | ------------------------------------------------------ |
| Tauri (core, tauri-build)          | 2.x               | MIT / Apache-2.0                | Yes               | Framework.                                             |
| tauri-plugin-log                   | 2.x               | MIT / Apache-2.0                | Yes               |                                                        |
| tauri-plugin-dialog                | 2.x               | MIT / Apache-2.0                | Yes               | Native file picker (Phase 2).                          |
| serde / serde_json                 | 1.x               | MIT / Apache-2.0                | Yes               |                                                        |
| thiserror / anyhow                 | 1.x               | MIT / Apache-2.0                | Yes               |                                                        |
| tracing / tracing-subscriber / tracing-appender | 0.1.x  | MIT                             | Yes               |                                                        |
| tokio                              | 1.x               | MIT                             | Yes               |                                                        |
| rusqlite (with bundled sqlite)     | 0.31+             | MIT (rusqlite) · Public Domain (SQLite) | Yes       | Bundled build avoids OS libsqlite.                     |
| parking_lot                        | 0.12              | MIT / Apache-2.0                | Yes               |                                                        |
| uuid                               | 1.x               | MIT / Apache-2.0                | Yes               |                                                        |
| chrono                             | 0.4.x             | MIT / Apache-2.0                | Yes               |                                                        |
| dirs                               | 5.x               | MIT / Apache-2.0                | Yes               |                                                        |
| once_cell                          | 1.x               | MIT / Apache-2.0                | Yes               |                                                        |
| sha2                               | 0.10              | MIT / Apache-2.0                | Yes               | Source-fingerprint hash (Phase 2).                     |
| fs4                                | 0.9               | MIT / Apache-2.0                | Yes               | Cross-platform free-disk-space (Phase 2).              |
| libc (unix targets)                | 0.2               | MIT / Apache-2.0                | Yes (unix only)   | `SIGTERM` for graceful FFmpeg cancel (Phase 2).        |
| React / React-DOM                  | 18.x              | MIT                             | Yes               |                                                        |
| React Router                       | 6.x               | MIT                             | Yes               |                                                        |
| zustand                            | 4.x               | MIT                             | Yes               |                                                        |
| @tauri-apps/api                    | 2.x               | MIT / Apache-2.0                | Yes               |                                                        |
| @tauri-apps/plugin-dialog          | 2.x               | MIT / Apache-2.0                | Yes               | Frontend binding for native pickers (Phase 2).         |
| Python 3.11+                       | 3.11 / 3.12       | PSF                             | Runtime dependency (Phase 12)   | Not bundled in the current installer — the app locates a user-installed interpreter via `LMT_PYTHON` env var, then bundled candidates (`<bundle>/Contents/Resources/python-embed/…`), then `python3`/`python` on `PATH`. Bundling a self-contained interpreter is supported by the packaging layout (`bundled_python_candidates` probes it first) but left to the distributing team so a normal install stays lightweight. PSF is compatible with commercial redistribution. |
| FFmpeg / ffprobe                   | 4.x — 8.x         | LGPL 2.1+ (default builds)      | External runtime (Phase 12)     | Not bundled in the installer. macOS/Windows users install via Homebrew/winget; the Linux `.deb` `depends: ffmpeg`. `Settings › FFmpeg` accepts a custom path via `AppSettings.ffmpeg_path`. Invoked as CLI only (Phase 2 extract, Phase 7 sync via `atempo` / `apad` / `aresample`, Phase 8 mix via `amix` / `adelay` / `volume` / `sidechaincompress` / `aformat`, Phase 9 render via `-c:v copy` mux, `subtitles=` filter for burn-in, `libx264` / `libx265` / `libvpx-vp9` for optional re-encode, `aac` / `libopus` for audio, and `ffprobe -show_streams` for post-render validation) — never linked into the app binary, so LGPL relinking obligations do not attach to us. See §3 for the `--enable-nonfree` exclusion. |
| faster-whisper                     | 1.x               | MIT                             | Optional runtime (Phase 3) | Imported lazily by `FasterWhisperProvider`; the worker degrades gracefully if it is missing. |
| CTranslate2 (via faster-whisper)   | 4.x               | MIT                             | Optional runtime (Phase 3) | Transitive dependency of faster-whisper. |
| huggingface_hub (via faster-whisper) | 0.20+           | Apache-2.0                      | Optional runtime (Phase 3) | Only touched during explicit Whisper model download. As of Phase 10, that entry point is hard-gated by `AppSettings.offline_mode`; when Offline Mode is on (the default), the app refuses the request with `MODEL_NETWORK_DISABLED` before any HTTP call is made. Phase 11 added no new network callers — the performance work is entirely local. |
| tokenizers (via faster-whisper)    | 0.15+             | Apache-2.0                      | Optional runtime (Phase 3) | Transitive. |
| onnxruntime (via faster-whisper)   | 1.x               | MIT                             | Optional runtime (Phase 3) | Transitive; used for optional VAD. |
| OpenAI Whisper model weights       | small/medium/large-v2/large-v3/turbo | MIT | User-installed (Phase 3) | Downloaded from HuggingFace on explicit user action. Not committed to Git. |
| llama.cpp                          | latest            | MIT                             | Optional runtime (Phase 4) | Native inference engine; loaded via `llama-cpp-python`. Never bundled with the app in current phases. |
| llama-cpp-python                   | 0.2+              | MIT                             | Optional runtime (Phase 4) | Imported lazily by `LlamaCppTranslationProvider`; the worker degrades gracefully if it is missing. |
| GGUF translation model (user-supplied) | user's choice | **Per-model** — user's responsibility | User-installed (Phase 4) | Dropped into `<data>/models/translation/` by the user. Common choices: Qwen2 (Apache-2.0), Llama 3 (Meta community license — verify), Mistral (Apache-2.0). Never bundled or auto-downloaded. |
| Subtitle parsers (SRT + ASS)       | in-house (Phase 5) | MIT (this crate)                | Yes               | Lightweight in-house Rust parser/writer. No third-party subtitle library required, so no GPL exposure from `pysrt`/`pyass`. |
| piper-tts                          | 1.2+              | MIT                             | Optional runtime (Phase 6) | Imported lazily by `PiperTTSProvider`; the worker degrades gracefully if it is missing. The provider can also drive the `piper` CLI if only the binary is installed. |
| onnxruntime                        | 1.17+             | MIT                             | Optional runtime (Phase 6) | Transitive of piper-tts; only loaded when a Piper voice is synthesised in-process. |
| espeak-ng phoneme data (via piper) | 1.51+             | GPLv3                           | External runtime (Phase 6) | Piper uses eSpeak NG **only** as an external phonemiser executable invoked out-of-process. We do not link against libespeak, so the GPL does not attach to the shipped binary. Users install eSpeak NG themselves. |
| Piper Vietnamese voice models      | user's choice     | **Per-model** — user's responsibility | User-installed (Phase 6)  | Dropped into `<data>/models/tts/piper/<voice_id>/` by the user. Voices from `rhasspy/piper-voices` on HuggingFace are typically MIT / CC0 but must be re-checked per voice — some third-party voices are trained on data with restrictive licenses. Never bundled or auto-downloaded. |

### Dev-only

| Component                          | Version           | License                         | Notes                                                     |
| ---------------------------------- | ----------------- | ------------------------------- | --------------------------------------------------------- |
| TypeScript                         | 5.x               | Apache-2.0                      |                                                           |
| Vite                               | 5.x               | MIT                             |                                                           |
| @vitejs/plugin-react               | 4.x               | MIT                             |                                                           |
| @tauri-apps/cli                    | 2.x               | MIT / Apache-2.0                |                                                           |
| pytest                             | 8.x               | MIT                             |                                                           |
| clippy / rustfmt                   | with Rust         | MIT / Apache-2.0                |                                                           |

---

## 2. Reserved / to be reviewed before landing (post-1.0)

These components have **not** been added to the repo. Their licensing
implications must be re-checked if a future release introduces them,
and this file updated at that time.

| Component                          | Status (Phase 12)             | Expected license                | Concern                                                |
| ---------------------------------- | ----------------------------- | ------------------------------- | ------------------------------------------------------ |
| Bundled FFmpeg sidecar             | **Not shipped.** FFmpeg is a runtime dependency (see §1). Bundling is left to the distributing team. | LGPL 2.1+ | If a future release ships a bundled FFmpeg build, ensure it is the default `--enable-lgpl` build (no `--enable-nonfree`, no `--enable-gpl` unless deliberately re-licensing the shipped artefact). |
| Bundled Python interpreter         | **Not shipped.** Runtime dependency (see §1). Packaging layout (`bundled_python_candidates`) already probes `<bundle>/Contents/Resources/python-embed/…`. | PSF | PSF is redistribution-friendly; add per-platform build recipes when shipping. |
| Auto-updater                       | **Excluded by design (Phase 12).** No auto-update code is shipped. | tauri-plugin-updater = MIT / Apache-2.0 | If added later, must be opt-in and clearly separated from offline functionality (per Phase 12 spec). |
| Code-signing (macOS notarisation, Windows Authenticode) | **Not configured.** `tauri.conf.json` leaves `macOS.signingIdentity` and Windows signing empty. | Apple Developer / Sectigo / DigiCert | Fill in per-team credentials before public distribution; unrelated to source-code licensing. |
| numpy / soundfile                  | **Not shipped.** Every audio path stays FFmpeg-based. | BSD-3-Clause / BSD-3-Clause | Only added if a future TTS/sync provider needs them. Phases 6–12 intentionally avoid them — Phase 6 writes WAVs via Python's stdlib `wave` module, Phase 7 delegates all resampling / stretching / padding to FFmpeg (`atempo` / `apad` / `aresample`), Phase 8 does the same for mixing (`amix` + `adelay` + `sidechaincompress`), Phase 9 muxes / burns / re-encodes via FFmpeg (`-c:v copy`, `subtitles=`, `libx264`, `aac`), Phase 10 (Model Manager) is metadata-only and imports models via symlink/copy, Phase 11 (performance) added no audio processing, and Phase 12 added no dependencies at all — only Tauri config, log rotation, storage/cache commands and docs. |

---

## 3. Explicit exclusions (must never appear)

* Any GPLv3-only library that would infect the shipped binary.
* Any AI model whose license forbids commercial use or redistribution.
* `openssl` linked dynamically to a system copy (Windows portability).
* `--enable-nonfree` FFmpeg builds.

---

## 4. Audit workflow

1. When a new dependency is proposed, list it in the pull-request
   description with source, version, license and redistribution status.
2. Update this file in the same PR.
3. Before a release, run `cargo tree`, `pnpm licenses ls`, `pip freeze`
   and diff against this file. `cargo deny check licenses` is a
   recommended optional step for teams that add CI (not shipped by
   this repo).
