"""Local GGUF model registry.

Model files live under ``<models_root>/translation/`` and are ALWAYS
installed by the user — the app never downloads them automatically.
This module scans the directory and returns whatever ``*.gguf`` files
it finds, so users can drop in any llama.cpp-compatible model they
prefer (Qwen2, Llama 3, Mistral, Phi, ...).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

TRANSLATION_SUBDIR = "translation"

# Curated list of translation models the app can auto-download from
# HuggingFace when the user has nothing installed. Keys are the
# *presets* the UI knows about (short slugs, forever-stable). Values
# are: a HuggingFace repo id, a specific quantised GGUF filename
# inside that repo, an approximate on-disk size in bytes, and a
# short human-readable label. Multiple presets can share a repo.
#
# When picking presets, prefer quantisations that (a) fit comfortably
# on a 16 GB Apple Silicon Mac, (b) are strong at English↔Vietnamese
# *and* Chinese↔Vietnamese (the two languages this app targets most),
# and (c) come from a well-maintained community re-quantisation so
# they don't disappear behind auth walls.
_RECOMMENDED_MODELS: dict[str, dict[str, object]] = {
    # ~2 GB — good default for first-time users on any modern Mac.
    "qwen2.5-3b-instruct-q4_k_m": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "approx_size_bytes": 2_000_000_000,
        "label": "Qwen 2.5 3B Instruct (Q4_K_M, ~2 GB) — balanced default",
    },
    # ~4.7 GB — best quality on Mac Silicon 16 GB+.
    "qwen2.5-7b-instruct-q4_k_m": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "approx_size_bytes": 4_700_000_000,
        "label": "Qwen 2.5 7B Instruct (Q4_K_M, ~4.7 GB) — highest quality",
    },
    # ~2 GB — Llama alternative for users who prefer it.
    "llama-3.2-3b-instruct-q4_k_m": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "approx_size_bytes": 2_020_000_000,
        "label": "Llama 3.2 3B Instruct (Q4_K_M, ~2 GB)",
    },
}

# The preset the UI picks when the user has no translation model at
# all and clicks "Download & translate". Kept small so the initial
# download completes in a couple of minutes on a typical connection.
DEFAULT_RECOMMENDED_PRESET = "qwen2.5-3b-instruct-q4_k_m"


def recommended_presets() -> list[dict[str, object]]:
    """List every preset the app knows how to auto-download.

    Wire-shape (camelCase for Rust serde compatibility): each entry
    has ``preset``, ``repo``, ``filename``, ``approxSizeBytes``,
    ``label``, and ``isDefault``.
    """
    out: list[dict[str, object]] = []
    for preset, meta in _RECOMMENDED_MODELS.items():
        out.append(
            {
                "preset": preset,
                "repo": meta["repo"],
                "filename": meta["filename"],
                "approxSizeBytes": int(meta["approx_size_bytes"]),  # type: ignore[arg-type]
                "label": meta["label"],
                "isDefault": preset == DEFAULT_RECOMMENDED_PRESET,
            }
        )
    return out


def recommended_meta(preset: str) -> Optional[dict[str, object]]:
    return _RECOMMENDED_MODELS.get(preset)


@dataclass(frozen=True)
class TranslationModelInfo:
    name: str
    path: str
    size_bytes: int
    is_default: bool

    def to_dict(self) -> dict:
        # Rust `translation::TranslationModelInfo` uses
        # `#[serde(rename_all = "camelCase")]`, so the wire contract
        # is camelCase. Emit explicit keys — see the same fix in
        # `stt/registry.py::ModelInfo.to_dict` for the rationale.
        return {
            "name": self.name,
            "path": self.path,
            "sizeBytes": self.size_bytes,
            "isDefault": self.is_default,
        }


def models_dir(models_root: Path) -> Path:
    return Path(models_root) / TRANSLATION_SUBDIR


def list_models(models_root: Path) -> list[TranslationModelInfo]:
    """Enumerate ``*.gguf`` files under ``<models_root>/translation/``.

    Returns them sorted by name. The first one is flagged as the
    default so the UI can pre-select it.
    """
    root = models_dir(models_root)
    if not root.is_dir():
        return []
    entries: list[TranslationModelInfo] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and entry.suffix.lower() == ".gguf":
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            entries.append(
                TranslationModelInfo(
                    name=entry.name,
                    path=str(entry),
                    size_bytes=size,
                    is_default=False,
                )
            )
    if entries:
        entries[0] = TranslationModelInfo(
            name=entries[0].name,
            path=entries[0].path,
            size_bytes=entries[0].size_bytes,
            is_default=True,
        )
    return entries


def resolve_model_path(models_root: Path, name: str) -> Optional[Path]:
    """Look up a model file by its bare ``name.gguf`` (the value the UI
    hands us). Returns ``None`` if it's missing.
    """
    if not name:
        return None
    root = models_dir(models_root)
    if not root.is_dir():
        return None
    # Reject anything trying to escape the translation directory.
    if "/" in name or "\\" in name or ".." in name or name in (".",):
        return None
    candidate = root / name
    if candidate.is_file() and candidate.suffix.lower() == ".gguf":
        return candidate
    return None


def is_installed(models_root: Path, name: str) -> bool:
    return resolve_model_path(models_root, name) is not None


def default_model(models_root: Path) -> Optional[str]:
    models = list_models(models_root)
    return models[0].name if models else None


def llama_cpp_installed() -> bool:
    """Cheap ``importlib`` probe so callers can degrade gracefully when
    ``llama-cpp-python`` is missing.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("llama_cpp") is not None
    except Exception:
        return False


def ensure_models_dir(models_root: Path) -> Path:
    d = models_dir(models_root)
    os.makedirs(d, exist_ok=True)
    return d
