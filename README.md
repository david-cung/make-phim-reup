# Local Movie Translator

Offline-first desktop app that translates and dubs movies — **entirely on
your computer**. No cloud, no telemetry, no mandatory internet.
Built on Tauri 2 (Rust + React) with a local Python worker for AI.

> Prioritised for macOS Apple Silicon, cross-platform ready for Windows and Linux.

---

## What it does

```
New Project → Import Movie → Transcribe → Translate → Edit Subtitle
                                                           ↓
Render ← Mix Audio ← Sync Voice ← Generate Voice ←────────┘
```

All nine stages run against **local models** you install once. After
setup, you can pull the network cable and everything keeps working.

---

## System requirements

| Component  | Minimum                                | Recommended                      |
| ---------- | -------------------------------------- | -------------------------------- |
| OS         | macOS 11, Windows 10, Ubuntu 22.04     | macOS 14, Windows 11, Ubuntu 24  |
| Arch       | Intel x86_64 / Apple Silicon aarch64   | Apple Silicon aarch64            |
| RAM        | 8 GB                                   | 16 GB+ (Whisper large / 7B LLM)  |
| Disk       | 5 GB free (models excluded)            | 30 GB+ for a mid-size model set  |
| Python     | 3.11+                                  | 3.12 (same interpreter, faster)  |
| FFmpeg     | 6.0+ (`ffmpeg` and `ffprobe` on PATH)  | 7.0+                             |

The installer itself is small (< 30 MB). Everything heavy — AI models,
FFmpeg, Python inference libraries — is installed **separately** and
lives outside the app bundle.

---

## Installation

### 1. Install the app

Download the appropriate installer from the [releases page](#build-from-source)
or build from source (see below):

- **macOS:** `Local Movie Translator.dmg` — drag to `/Applications`.
- **Windows:** `local-movie-translator_<version>_x64_en-US.msi`.
- **Linux:** `.AppImage` or `.deb` package.

### 2. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install --id=Gyan.FFmpeg -e
```

If FFmpeg isn't on your `PATH`, point at it in **Settings → FFmpeg**.

### 3. Install Python 3.11+ and worker dependencies

Only the interpreter is required system-wide — the worker's Python
packages live inside `python/pyproject.toml` and get installed into
a venv the app can find. **All three ML extras must be installed**
or the corresponding pipeline stage refuses to start with a
`STT_WHISPER_NOT_INSTALLED` / `TRANSLATION_LLAMA_NOT_INSTALLED` /
similar error (the stage-start buttons remain disabled):

```bash
python3.11 -m venv .venv-worker
source .venv-worker/bin/activate            # cmd.exe: .venv-worker\Scripts\activate
pip install -e './python[stt,translation,tts]'
```

Then either:

- Set `LMT_PYTHON` to `<repo>/.venv-worker/bin/python3` (or its
  Windows equivalent), **or**
- Ensure that interpreter is the first `python3` / `python` on `PATH`.

If you skip an extra on purpose (e.g. you only need transcription,
not TTS), the app still starts; only the disabled stage's button
surfaces the hint pointing back at this section.

### 4. Install models

The app never downloads models on your behalf. Install them yourself:

```
<app data>/models/
├── whisper/       # CTranslate2 Whisper snapshot dirs (e.g. large-v3)
├── translation/   # GGUF llama.cpp files (Qwen, Llama, Mistral, …)
├── tts/           # Piper (or other) engine subdirs
└── voices/        # Piper voice pairs (.onnx + .onnx.json)
```

Find the exact location under **Settings → Application → Models dir**.
Drop your files in, then click **Settings → AI Models → Scan Models**
(or use **Add Local Model** to register a folder outside the tree).

### 5. That's it

Turn on **Settings → Preferences → Offline mode** to make it explicit —
the app makes zero network requests regardless.

---

## Storage locations

Everything the app owns lives under OS-standard directories (no
hard-coded paths):

| Kind     | macOS                                                       | Linux                                     | Windows                                            |
| -------- | ----------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| Data     | `~/Library/Application Support/local-movie-translator/`     | `~/.local/share/local-movie-translator/`  | `%APPDATA%\local-movie-translator\`                |
| Config   | `~/Library/Application Support/local-movie-translator/`     | `~/.config/local-movie-translator/`       | `%APPDATA%\local-movie-translator\`                |
| Cache    | `~/Library/Caches/local-movie-translator/`                  | `~/.cache/local-movie-translator/`        | `%LOCALAPPDATA%\local-movie-translator\`           |
| Logs     | `<cache>/logs/`                                             | `<cache>/logs/`                           | `<cache>\logs\`                                    |
| Projects | `<data>/projects/`                                          | `<data>/projects/`                        | `<data>\projects\`                                 |
| Models   | `<data>/models/` (override in Settings)                     | `<data>/models/`                          | `<data>\models\`                                   |

**Settings → Storage & Logs** shows the current sizes, lets you open
any of them in the OS file explorer, clear the cache safely (per-project
data, models and rendered outputs are never touched), and truncate log
rotations.

---

## Project layout

Each project is a self-contained folder — copy it to a USB drive and it
still works on another machine (as long as the same models are
installed there):

```
<projects>/<uuid>/
├── project.json          # metadata mirror (name, source path, models)
├── audio/                # extracted PCM for Whisper
├── cache/                # scratch (safe to delete via "Clear cache")
├── output/
│   ├── movie_vi.mp4      # final rendered video
│   ├── movie_vi.srt      # exported subtitle sidecar
│   ├── render.json       # render manifest
│   └── mix.wav           # mixed dubbed track
├── subtitles/            # SRT/ASS working copies
├── transcript/           # Whisper JSON
├── translation/          # LLM output
├── tts/                  # per-segment voice WAVs + manifest
├── sync/                 # per-segment time-stretched WAVs + manifest
└── source.<ext>          # symlink (or reference) to the original movie
                          # — never copied by default
```

---

## Supported formats

- **Video import:** `.mp4`, `.mkv`, `.mov`, `.m4v`, `.avi`, `.webm`
- **Subtitle import/export:** `.srt`, `.ass`, `.ssa`
- **Video render:** MP4 (H.264/H.265) or MKV, with either external SRT
  sidecar or hard-burned subtitles

---

## Offline mode

`Settings → Preferences → Offline mode` is on by default. With it on:

- No HTTP requests.
- No telemetry, analytics or update checks.
- No automatic model downloads.
- No external services of any kind.

Turning it off does **not** enable model downloads or telemetry — it
just unlocks features that would benefit from network (e.g. a future
manual "check for updates" button; not present today).

---

## Troubleshooting

| Symptom                                              | Fix                                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `WORKER_PYTHON_MISSING` on startup                   | Install Python 3.11+ and either put it on `PATH` or set `LMT_PYTHON=/full/path/python`.   |
| `FFMPEG_NOT_FOUND`                                   | Install FFmpeg (see step 2) or point at it in **Settings → FFmpeg**.                       |
| `STT_MODEL_NOT_INSTALLED` / no Whisper models        | Drop a CTranslate2 Whisper snapshot into `<models>/whisper/<name>/`.                       |
| `TRANSLATE_MODEL_NOT_INSTALLED` / no LLMs            | Drop a `.gguf` file into `<models>/translation/`.                                          |
| `STT_OUT_OF_MEMORY` / `TRANSLATE_OUT_OF_MEMORY`      | Pick a smaller model, or lower the context in **Settings → Performance**.                  |
| App feels sluggish on 8 GB RAM                       | Enable **Auto-unload models after** (90 s default) so models don't stay resident.          |
| "Some jobs didn't finish last time" banner           | Reopen the listed project and re-run the affected stage — completed work is preserved.    |
| Rendered video has no audio                          | Ensure sync + mix stages completed. Retry **Mix Audio** then **Render**.                   |

Full logs live under **Settings → Storage & Logs → Open** (Logs row).
Rotated daily, retained for 14 days.

---

## Build from source

```bash
# Prerequisites
brew install ffmpeg python@3.12 pnpm rustup

# Clone + install
git clone https://github.com/local-movie-translator/local-movie-translator
cd local-movie-translator
pnpm install

# Worker venv. `[stt,translation,tts]` pulls in faster-whisper +
# huggingface_hub + llama-cpp-python + piper-tts + onnxruntime.
# Without them the corresponding pipeline stage refuses to start.
python -m venv .venv-worker && source .venv-worker/bin/activate
pip install -e './python[stt,translation,tts]'
export LMT_PYTHON="$(pwd)/.venv-worker/bin/python3"

# Dev
pnpm tauri:dev

# Production build (macOS Apple Silicon)
pnpm tauri:build:mac

# Universal macOS binary
pnpm tauri:build:mac-universal

# Windows / Linux
pnpm tauri:build:windows
pnpm tauri:build:linux
```

Release artefacts land in `src-tauri/target/release/bundle/`.

### Full verification

```bash
cargo fmt --check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo test  --manifest-path src-tauri/Cargo.toml --lib
pnpm build
pytest -q python
```

---

## Security & privacy

- **Least-privilege Tauri capabilities.** The webview can invoke IPC
  commands, listen to events, and open OS file/save dialogs — nothing
  else. All filesystem work goes through Rust and every path is
  validated against an allow-listed root (see `paths::validate_within`).
- **No shell interpolation.** FFmpeg and Python are always invoked with
  typed argv arrays — user input is never concatenated into a command
  line.
- **No sensitive data in logs.** Transcripts, LLM prompts/responses,
  subtitle text and file contents are never logged. Logs record method
  names, timings and counts only. `RUST_LOG` can enable verbose logging
  for advanced debugging.
- **No telemetry.** No analytics, crash reporters or "phone home"
  probes are shipped.

---

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design. Short
version:

```
                 ┌──────────────────────┐
                 │      Tauri App       │
                 │  React UI  ⇄  Rust   │
                 └──────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           FFmpeg    Python Worker      SQLite
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Whisper       LLM          TTS
                          │
                          ▼
                    Local AI Models
```

## License

MIT (app code). Third-party components — FFmpeg, Whisper models,
llama.cpp, Piper voices, etc. — retain their own licences. See
[`LICENSES.md`](./LICENSES.md).
