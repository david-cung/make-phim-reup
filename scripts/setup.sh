#!/usr/bin/env bash
# Local Movie Translator — developer bootstrap.
# Verifies the required toolchains and prepares a Python virtualenv for the
# worker. Safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf "\033[1;34m▸\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

require() {
  local bin="$1" hint="${2:-}"
  command -v "$bin" >/dev/null 2>&1 || die "'$bin' not found on PATH. $hint"
}

log "checking toolchains"

require node "Install Node.js 20 LTS (nvm recommended)."
NODE_MAJOR=$(node --version | sed -E 's/^v([0-9]+)\..*/\1/')
[ "$NODE_MAJOR" -ge 20 ] || warn "Node ≥ 20 recommended (found $(node --version))"

require pnpm "Install pnpm: 'npm i -g pnpm'."

# Rust may live in ~/.cargo/bin; source the env if the caller hasn't.
if ! command -v cargo >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
fi
require cargo "Install Rust: https://rustup.rs"

require python3 "Install Python 3.11 or 3.12."
PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PY_VER" in
  3.11|3.12) ;;
  3.13|3.14|3.15) warn "Python $PY_VER: AI wheels (faster-whisper, ctranslate2, llama-cpp-python) may not exist yet. Phase 1 works, later phases will need 3.11 or 3.12." ;;
  *) warn "Python $PY_VER may be too old — 3.11+ required.";;
esac

if ! command -v ffmpeg >/dev/null 2>&1; then
  warn "ffmpeg not on PATH — Phase 2 will require it. On macOS: 'brew install ffmpeg'."
else
  log "ffmpeg: $(ffmpeg -version | head -1)"
fi

log "creating python venv at .venv-worker"
python3 -m venv .venv-worker
# shellcheck disable=SC1091
source .venv-worker/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -e "python[dev,stt,translation,tts]" >/dev/null
log "python worker and local AI runtimes installed (editable) in .venv-worker"

log "all set. Run: pnpm install && pnpm tauri:dev"
