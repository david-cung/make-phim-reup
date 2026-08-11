"""Top-level RPC handlers (Phase 1 core + Phase 3 wiring)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, logging as log
from .errors import RpcError, RpcErrorCode
from .stt import handlers as stt_handlers
from .sync import handlers as sync_handlers
from .translation import handlers as translation_handlers
from .tts import handlers as tts_handlers

_STARTED_AT = time.monotonic()

# Phase 11 — the most-recent performance snapshot from the host. Read
# by providers when they load a model (`n_threads`, `n_gpu_layers`,
# `cpu_threads`, ...). Kept intentionally simple: providers ask for
# the field they care about with a default, no per-engine glue here.
_PERF: dict[str, Any] = {
    "cpu_threads": None,
    "gpu_acceleration": True,
}


def _uptime_ms() -> int:
    return int((time.monotonic() - _STARTED_AT) * 1000)


def get_perf() -> dict[str, Any]:
    """Return the current perf snapshot. Providers use this at model
    load time so a settings change flows through the next reload."""
    return dict(_PERF)


def initialize(params: dict[str, Any]) -> dict[str, Any]:
    """Idempotent one-time setup handshake with the host.

    Accepts:
      - `app_version` (str, optional): host app version, logged for correlation.
      - `log_level`   (str, optional): "trace|debug|info|warn|error".
      - `data_root`   (str, optional): where the app stores per-project data.
      - `perf`        (dict, optional, Phase 11):
                       ``{"cpu_threads": int|null, "gpu_acceleration": bool}``.
                       Providers pick these up on their next model load
                       so a settings change takes effect without a
                       worker restart.
    """
    app_version = params.get("app_version")
    log_level = params.get("log_level")
    data_root = params.get("data_root")

    if isinstance(log_level, str):
        os.environ["LMT_WORKER_LOG_LEVEL"] = log_level.upper()
        log.reload_level()
    if isinstance(data_root, str):
        os.environ["LMT_DATA_DIR"] = data_root

    models_root = params.get("models_root")
    if isinstance(models_root, str) and models_root:
        os.environ["LMT_MODELS_DIR"] = models_root

    perf = params.get("perf")
    if isinstance(perf, dict):
        cpu_threads = perf.get("cpu_threads")
        if cpu_threads is None or isinstance(cpu_threads, int):
            _PERF["cpu_threads"] = cpu_threads if isinstance(cpu_threads, int) and cpu_threads > 0 else None
        gpu = perf.get("gpu_acceleration")
        if isinstance(gpu, bool):
            _PERF["gpu_acceleration"] = gpu

    models_root = _resolve_models_root()
    stt_handlers.configure(models_root=models_root)
    translation_handlers.configure(models_root=models_root)
    tts_handlers.configure(models_root=models_root)

    ffmpeg_bin = params.get("ffmpeg_bin")
    if isinstance(ffmpeg_bin, str) and ffmpeg_bin:
        sync_handlers.configure(ffmpeg_bin=ffmpeg_bin)
    else:
        sync_handlers.configure()

    log.info(
        "initialized",
        app_version=app_version,
        worker_version=__version__,
        python=sys.version.split()[0],
        data_root=data_root,
        models_root=os.environ.get("LMT_MODELS_DIR"),
        cpu_threads=_PERF["cpu_threads"],
        gpu_acceleration=_PERF["gpu_acceleration"],
    )

    return {
        "ok": True,
        "workerVersion": __version__,
        "pythonVersion": sys.version.split()[0],
    }


def _resolve_models_root() -> Path:
    env = os.environ.get("LMT_MODELS_DIR")
    if env:
        return Path(env)
    data_dir = os.environ.get("LMT_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "models"
    return Path.home() / ".cache" / "movie-translator" / "models"


def ping(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "pong": True,
        "pid": os.getpid(),
        "uptimeMs": _uptime_ms(),
    }


def env_info(_params: dict[str, Any]) -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_version = _probe_ffmpeg_version(ffmpeg_path) if ffmpeg_path else None
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "ffmpegAvailable": ffmpeg_path is not None,
        "ffmpegVersion": ffmpeg_version,
        "cpuCount": os.cpu_count() or 1,
    }


def shutdown(_params: dict[str, Any]) -> dict[str, Any]:
    log.info("shutdown handler invoked")
    return {"ok": True}


def _probe_ffmpeg_version(bin_path: str) -> str | None:
    try:
        out = subprocess.run(
            [bin_path, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warn("ffmpeg probe failed", error=str(e))
        return None
    line = (out.stdout or "").splitlines()[:1]
    if not line:
        return None
    # e.g. "ffmpeg version 8.1.1 Copyright ..." → "8.1.1"
    parts = line[0].split()
    if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
        return parts[2]
    return line[0].strip() or None


# ---- registration ----

def install(dispatcher) -> None:
    dispatcher.register("initialize", initialize)
    dispatcher.register("ping", ping)
    dispatcher.register("env_info", env_info)
    dispatcher.register("shutdown", shutdown)
    stt_handlers.install(dispatcher)
    translation_handlers.install(dispatcher)
    tts_handlers.install(dispatcher)
    sync_handlers.install(dispatcher)
    # Reject anything else with a stable code.
    _ = RpcError, RpcErrorCode  # keep imports live for later handlers
