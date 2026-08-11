# Tech stack — Local Movie Translator

The stack is chosen to satisfy the primary objectives in order:
offline, lightweight shell, low idle RAM, efficient CPU/GPU when active,
maintainable, and cross-platform (macOS Apple Silicon → Windows x64).

---

## 1. Detected development environment

Verified on this machine on 2026-08-11.

| Component        | Version                            | Path                            | Notes                                   |
| ---------------- | ---------------------------------- | ------------------------------- | --------------------------------------- |
| OS               | macOS 26.5.2 (Darwin 25.5.0)       | —                               | Apple Silicon                           |
| Arch             | arm64 (Apple M-series)             | —                               |                                         |
| Free disk        | 111 GB                             | `/System/Volumes/Data`          | plenty for models & renders             |
| Xcode / clang    | Xcode 26.2, Apple clang 17         | `/Applications/Xcode.app`       | required by `cargo` on macOS            |
| Rust             | rustc 1.97.1                       | `~/.cargo/bin/rustc`            | needs `source ~/.cargo/env`             |
| Cargo            | 1.97.1                             | `~/.cargo/bin/cargo`            |                                         |
| Rust target      | `aarch64-apple-darwin` (installed) | —                               |                                         |
| Node.js          | v20.16.0                           | `~/.nvm/…`                      | via nvm                                 |
| pnpm             | 9.15.9                             | global                          | corepack broken in this env → use pnpm  |
| Python           | 3.14.5                             | `/opt/homebrew/bin/python3`     | Homebrew                                |
| SQLite (Python)  | 3.53.1                             | stdlib                          |                                         |
| FFmpeg / FFprobe | 8.1.1                              | `/opt/homebrew/bin/ffmpeg`      | videotoolbox, audiotoolbox, x264, x265  |
| Homebrew         | 6.0.11                             | `/opt/homebrew`                 |                                         |
| Tauri CLI        | not installed                      | —                               | installed via `@tauri-apps/cli` devDep  |

### Compatibility notes

* **Python 3.14 is very new.** `faster-whisper`, `ctranslate2` and
  `llama-cpp-python` usually only publish wheels a few weeks after a Python
  release. Phase 1's worker uses **only the standard library**, so 3.14 is
  fine. From Phase 3 onward the worker will run inside a project-local
  virtualenv pinned to **Python 3.11 or 3.12** for guaranteed wheel
  availability. This will be documented in the worker README and
  auto-selected by the `scripts/setup.sh` bootstrap.
* **No CUDA on Apple Silicon.** GPU acceleration is via Metal
  (`ctranslate2` Metal backend, `llama.cpp` Metal, `whisper.cpp` Metal,
  Piper CPU). Never assume CUDA; runtime detection lives in the worker's
  `env_info` response.
* **FFmpeg is present system-wide.** For distribution we will bundle a
  static FFmpeg build (Phase 10) so users don't need Homebrew.

---

## 2. Chosen stack

### 2.1 Desktop shell

| Layer          | Choice                         | Why                                                        |
| -------------- | ------------------------------ | ---------------------------------------------------------- |
| App framework  | **Tauri 2**                    | Small binary, native WebView, Rust core. No Electron.      |
| Language       | **Rust (edition 2021)**        | Safety, no GC, mature process/IPC/SQLite ecosystem.        |
| Frontend       | **React 18 + TypeScript 5**    | Widely known, ecosystem for virtualization / routing.      |
| Build tool     | **Vite 5**                     | Fast dev, tiny prod bundle.                                |
| Routing        | **react-router-dom 6**         | Standard, tiny.                                            |
| State          | **zustand**                    | ~1 kB, no context boilerplate, great for a normalized store. |
| Styling        | Plain CSS + CSS variables       | No Tailwind/UI kit → keep bundle small and predictable.    |
| Node PM        | **pnpm 9**                     | Fastest, deterministic, hard-links to save disk.           |

### 2.2 Rust crates (Phase 1)

| Crate                  | Purpose                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `tauri = "2"`          | window, IPC, event emitter                                    |
| `tauri-build = "2"`    | build-time codegen                                            |
| `tauri-plugin-log`     | uniform log routing to file + console (dev)                   |
| `serde` / `serde_json` | typed IPC payloads                                            |
| `thiserror`            | structured error enum                                         |
| `anyhow`               | internal error propagation only (never leaks to frontend)     |
| `tracing`              | structured logs                                               |
| `tracing-subscriber`   | JSON formatter                                                |
| `tracing-appender`     | daily rolling file appender                                   |
| `tokio`                | async runtime (`rt-multi-thread`, `macros`, `process`, `sync`)|
| `rusqlite` (`bundled`) | zero system-lib dependency; small; sync API in blocking task  |
| `parking_lot`          | fast Mutex/RwLock                                             |
| `uuid` (`v4`)          | project ids                                                   |
| `chrono` (`serde`)     | RFC3339 timestamps                                            |
| `dirs`                 | OS-native data/config/cache dirs                              |
| `once_cell`            | lazy singletons                                               |

Explicitly not added yet (avoiding premature complexity):

* No `sqlx` — heavier than `rusqlite`; we get sync + `spawn_blocking`.
* No `reqwest` — offline-first, nothing to fetch in Phase 1.
* No `openssl` / `native-tls`.
* No `serde_with` etc. — small surface, small deps.

### 2.3 Frontend packages (Phase 1)

| Package                    | Purpose                        |
| -------------------------- | ------------------------------ |
| `react`, `react-dom`       | UI                             |
| `react-router-dom`         | routing                        |
| `zustand`                  | state                          |
| `@tauri-apps/api`          | typed `invoke` / `listen`      |
| `@tauri-apps/cli` (dev)    | tauri dev/build                |
| `typescript`, `vite`, `@vitejs/plugin-react` | build |
| `@types/react`, `@types/react-dom` | types |

Not added yet: `react-window` (Phase 5), `wavesurfer.js` (Phase 5), any
component library (never — YAGNI for this UI).

### 2.4 Python worker (Phase 1)

| Package    | Purpose                                                    |
| ---------- | ---------------------------------------------------------- |
| stdlib only | JSON-RPC dispatch, logging, subprocess helpers            |
| `pytest` (dev only) | unit tests                                        |

From Phase 3+ (pinned in a separate `requirements-ml.txt` at that time):

* `faster-whisper`
* `llama-cpp-python` (or a llama.cpp CLI subprocess wrapped by the worker)
* `piper-tts` and Piper Vietnamese voice bundles
* `soundfile`, `numpy` — only if strictly needed
* `pysrt` / `pyass` — small, permissive licenses

### 2.5 Media

* FFmpeg (system for dev, bundled binary for release). LGPL 2.1+ build
  chosen for release — avoid `--enable-nonfree`.

### 2.6 Persistence

* SQLite 3 via `rusqlite` (bundled). One file `<app_data>/db.sqlite3`.
* Filesystem for all binary artefacts (media, models, voices, logs).

---

## 3. Excluded technologies (with reason)

| Excluded                    | Why                                                            |
| --------------------------- | -------------------------------------------------------------- |
| Electron                    | Explicit product rule.                                         |
| Next.js                     | Not a web-server-shaped app.                                   |
| Any cloud STT/LLM/TTS API   | Offline-first, no telemetry.                                   |
| Ollama (as runtime dep)     | Optional in dev; production uses llama.cpp directly.           |
| Tailwind / MUI / Chakra     | Bundle size and hidden runtime cost for a small UI.            |
| Redux / RTK                 | Overkill vs. zustand for our screens.                          |
| `serde_yaml`, TOML for data | JSON is enough; SQLite is the source of truth.                 |

---

## 4. Version pinning strategy

* **Node** — pinned via `.nvmrc` (`20`) — LTS.
* **pnpm** — via `packageManager` field in `package.json`.
* **Rust** — pinned via `rust-toolchain.toml` to `stable-1.97` (matches
  the installed toolchain on this machine; will be re-evaluated before
  release).
* **Python** — Phase 1 accepts any 3.11+; Phase 3+ pins to 3.11 or 3.12
  inside a venv, controlled by `scripts/setup.sh`.
* **FFmpeg** — Phase 2+ requires ≥ 6.0.

---

## 5. Environment bootstrap (developer)

```bash
# from repo root
source ~/.cargo/env                     # if not already on PATH
pnpm install                            # frontend + Tauri CLI
scripts/setup.sh                        # verifies tools, warns on Python version
pnpm tauri dev                          # runs everything
```

For CI (later):

* macOS runner with Xcode → `cargo test`, `pnpm build`, `pnpm tauri build`
* Linux runner (self-hosted or GitHub) for lint only
* Windows runner (Phase 10) for MSI packaging
