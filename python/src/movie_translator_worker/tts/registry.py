"""Local voice-model registry.

Voice models are always installed by the user — the app never
downloads them automatically. Layout on disk::

    <models_root>/tts/<engine>/<voice_dir>/
        model.onnx            (or engine-specific primary file)
        model.onnx.json       (Piper's phoneme/inference config)
        voice.json            (optional metadata override)

If a directory does not contain a ``voice.json`` the registry infers
one from the engine-specific config file, so users can just drop a
Piper voice folder in and it appears immediately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .models import VoiceInfo

TTS_SUBDIR = "tts"


def tts_root(models_root: Path) -> Path:
    return Path(models_root) / TTS_SUBDIR


def engine_root(models_root: Path, engine: str) -> Path:
    return tts_root(models_root) / engine


def ensure_tts_root(models_root: Path) -> Path:
    root = tts_root(models_root)
    os.makedirs(root, exist_ok=True)
    return root


# ------------------------------------------------------------------ piper


def list_piper_voices(models_root: Path) -> list[VoiceInfo]:
    """Scan ``<models>/tts/piper/`` for installed voices."""
    root = engine_root(models_root, "piper")
    if not root.is_dir():
        return []
    voices: list[VoiceInfo] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        info = _piper_voice_from_dir(entry)
        if info is not None:
            voices.append(info)
    return voices


def resolve_piper_voice(models_root: Path, voice_id: str) -> Optional[VoiceInfo]:
    """Look up a Piper voice by id.

    ``voice_id`` corresponds to the on-disk directory name so it must
    not contain path separators or ``..`` components.
    """
    if not voice_id or "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        return None
    root = engine_root(models_root, "piper")
    if not root.is_dir():
        return None
    entry = root / voice_id
    if not entry.is_dir():
        return None
    return _piper_voice_from_dir(entry)


def _piper_voice_from_dir(entry: Path) -> Optional[VoiceInfo]:
    onnx = _first_matching(entry, "*.onnx")
    if onnx is None:
        return None
    override = _load_voice_json(entry / "voice.json")
    config_json = _find_config_json(onnx)
    inferred = _infer_from_piper_config(config_json) if config_json else {}
    merged: dict[str, Any] = {**inferred, **(override or {})}

    voice_id = str(merged.get("id") or entry.name)
    name = str(merged.get("name") or _prettify_voice_name(entry.name))
    language = str(merged.get("language") or "unknown").lower()
    gender = str(merged.get("gender") or "unknown").lower()
    sample_rate = int(merged.get("sampleRate") or merged.get("sample_rate") or 22050)
    quality = merged.get("quality")

    return VoiceInfo(
        id=voice_id,
        name=name,
        language=language,
        gender=gender,
        engine="piper",
        model_path=str(onnx),
        config_path=str(config_json) if config_json else None,
        sample_rate=sample_rate,
        installed=True,
        quality=str(quality) if quality is not None else None,
        supported_settings=["speed"],
    )


def _first_matching(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _find_config_json(onnx: Path) -> Optional[Path]:
    """Piper ships ``<voice>.onnx`` alongside ``<voice>.onnx.json``."""
    cfg = onnx.with_suffix(onnx.suffix + ".json")
    if cfg.is_file():
        return cfg
    # Fall back to *.onnx.json in the same dir if the name doesn't line up.
    alt = _first_matching(onnx.parent, "*.onnx.json")
    return alt if alt else None


def _load_voice_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_from_piper_config(config_json: Path) -> dict[str, Any]:
    """Piper's ``.onnx.json`` carries language + sample rate metadata.

    Schema is defined at
    https://github.com/rhasspy/piper — we tolerate missing keys and
    just skip them.
    """
    try:
        with config_json.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    audio = payload.get("audio")
    if isinstance(audio, dict) and "sample_rate" in audio:
        out["sampleRate"] = audio["sample_rate"]
    lang = payload.get("language")
    if isinstance(lang, dict):
        code = lang.get("code") or lang.get("family")
        if isinstance(code, str) and code:
            out["language"] = code.split("_")[0].lower()
    dataset = payload.get("dataset")
    if isinstance(dataset, str):
        out["name"] = dataset
    quality = payload.get("quality")
    if isinstance(quality, str):
        out["quality"] = quality
    return out


def _prettify_voice_name(dirname: str) -> str:
    parts = [p for p in dirname.replace("_", "-").split("-") if p]
    return " ".join(p.capitalize() for p in parts) if parts else dirname


# --------------------------------------------------------------- aggregate


def list_all_voices(models_root: Path) -> list[VoiceInfo]:
    """Union of every engine's installed voices, sorted stably."""
    voices: list[VoiceInfo] = []
    voices.extend(list_piper_voices(models_root))
    voices.sort(key=lambda v: (v.language, v.engine, v.id.lower()))
    return voices


def resolve_voice(
    models_root: Path, engine: str, voice_id: str
) -> Optional[VoiceInfo]:
    if engine == "piper":
        return resolve_piper_voice(models_root, voice_id)
    return None


# ------------------------------------------------------------- capability


def piper_installed() -> bool:
    """Cheap ``importlib`` probe so callers can degrade gracefully."""
    try:
        import importlib.util

        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


def supported_engines() -> list[dict[str, Any]]:
    """Static description of which engines the worker knows about.

    ``available`` reflects whether the runtime dependency is importable.
    """
    return [
        {
            "id": "piper",
            "name": "Piper (local, Vietnamese-capable)",
            "available": piper_installed(),
            "supportedSettings": ["speed"],
        }
    ]
