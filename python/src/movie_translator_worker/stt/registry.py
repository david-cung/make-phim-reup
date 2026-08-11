"""Catalogue of Whisper models we know how to run.

We intentionally hard-code the list of supported names — silently
running whatever the user types would be a footgun. Adding a new model
is a code change plus a version bump of the on-disk cache format.

Model files live under ``<models_root>/whisper/<slug>/``. The slug is
the model *name* the user picks; the directory is a CTranslate2
Whisper snapshot (``model.bin``, ``config.json``, ...). Downloading is
gated behind :func:`ensure_downloaded`, which is the only function in
this package that touches the network.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Repo IDs on the HuggingFace hub for CTranslate2-compatible Whisper builds.
# See https://github.com/SYSTRAN/faster-whisper#supported-model-names.
_KNOWN_MODELS: dict[str, dict[str, str | int]] = {
    "tiny":      {"repo": "Systran/faster-whisper-tiny",      "params_m": 39},
    "base":      {"repo": "Systran/faster-whisper-base",      "params_m": 74},
    "small":     {"repo": "Systran/faster-whisper-small",     "params_m": 244},
    "medium":    {"repo": "Systran/faster-whisper-medium",    "params_m": 769},
    "large-v2":  {"repo": "Systran/faster-whisper-large-v2",  "params_m": 1550},
    "large-v3":  {"repo": "Systran/faster-whisper-large-v3",  "params_m": 1550},
    "large":     {"repo": "Systran/faster-whisper-large-v3",  "params_m": 1550},
    "turbo":     {"repo": "Systran/faster-whisper-large-v3-turbo", "params_m": 809},
}

# Files that must be present for a snapshot to count as "installed".
_REQUIRED_FILES = ("model.bin", "config.json")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    params_m: int
    installed: bool
    size_bytes: Optional[int]
    path: Optional[str]

    def to_dict(self) -> dict:
        # Rust `stt::ModelInfo` uses `#[serde(rename_all = "camelCase")]`,
        # so the JSON contract on the wire is camelCase (`paramsM`,
        # `sizeBytes`). We emit that shape explicitly rather than
        # relying on `asdict` — otherwise the frontend's model
        # registry scan fails with `missing field paramsM` and every
        # Whisper model shows up as "not installed", which in turn
        # disables the "Start transcription" button.
        return {
            "name": self.name,
            "repo": self.repo,
            "paramsM": self.params_m,
            "installed": self.installed,
            "sizeBytes": self.size_bytes,
            "path": self.path,
        }


def known_model_names() -> list[str]:
    return list(_KNOWN_MODELS.keys())


def is_known(name: str) -> bool:
    return name in _KNOWN_MODELS


def repo_for(name: str) -> str:
    if name not in _KNOWN_MODELS:
        raise ValueError(f"unknown whisper model: {name}")
    return str(_KNOWN_MODELS[name]["repo"])


def model_dir(models_root: Path, name: str) -> Path:
    if name not in _KNOWN_MODELS:
        raise ValueError(f"unknown whisper model: {name}")
    return Path(models_root) / "whisper" / name


def is_installed(models_root: Path, name: str) -> bool:
    d = model_dir(models_root, name)
    return all((d / f).is_file() for f in _REQUIRED_FILES)


def snapshot_size(dir_path: Path) -> Optional[int]:
    if not dir_path.exists():
        return None
    total = 0
    for root, _dirs, files in os.walk(dir_path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                continue
    return total


def list_models(models_root: Path) -> list[ModelInfo]:
    out: list[ModelInfo] = []
    for name, meta in _KNOWN_MODELS.items():
        d = model_dir(models_root, name)
        installed = is_installed(models_root, name)
        out.append(
            ModelInfo(
                name=name,
                repo=str(meta["repo"]),
                params_m=int(meta["params_m"]),
                installed=installed,
                size_bytes=snapshot_size(d) if installed else None,
                path=str(d) if installed else None,
            )
        )
    return out


def remove_model(models_root: Path, name: str) -> None:
    d = model_dir(models_root, name)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def ensure_downloaded(
    models_root: Path,
    name: str,
    *,
    hf_downloader,  # noqa: ANN001 - injected for testability
    progress=None,  # noqa: ANN001 - Optional[Callable[[float, str], None]]
) -> Path:
    """Download the model if it is not already present, then return
    its directory. ``hf_downloader`` is a callable that takes
    ``(repo_id, local_dir, progress)``; :mod:`.handlers` supplies the
    real HuggingFace implementation while tests inject a stub.
    """
    if not is_known(name):
        raise ValueError(f"unknown whisper model: {name}")
    dest = model_dir(models_root, name)
    if is_installed(models_root, name):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    repo = repo_for(name)
    hf_downloader(repo, dest, progress)
    if not is_installed(models_root, name):
        raise RuntimeError(
            f"download of {name!r} finished but required files are missing under {dest}",
        )
    return dest
