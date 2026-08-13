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
F5_ENGINE = "f5-vietnamese"
F5_MODEL_ID = "hynt/F5-TTS-Vietnamese-ViVoice"
F5_MODEL_REVISION = "50228ccc563853f0ac628f49ed99a11f653d9ebe"
F5_MODEL_LICENSE = "CC-BY-NC-SA-4.0"
F5_MODEL_DIR_RELATIVE = Path("f5") / "vietnamese"
F5_MODEL_FILENAME = "model_last.pt"
F5_VOCAB_FILENAME = "vocab.txt"
F5_METADATA_FILENAME = "model.json"
F5_VOCODER_REPO = "charactr/vocos-mel-24khz"
F5_VOCODER_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"


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
    gender = str(merged.get("gender") or _infer_gender(entry.name, name)).lower()
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
        supported_settings=["speed", "volume"],
        model_name=onnx.name,
        model_source="rhasspy/piper-voices",
        license="MIT",
        commercial_use=True,
        cache_identity=_file_identity(onnx),
    )


# ---------------------------------------------------------- F5 Vietnamese


def f5_model_dir(models_root: Path) -> Path:
    return tts_root(models_root) / F5_MODEL_DIR_RELATIVE


def f5_model_paths(models_root: Path) -> dict[str, Path]:
    root = f5_model_dir(models_root)
    vocoder = root / "vocos"
    return {
        "root": root,
        "checkpoint": root / F5_MODEL_FILENAME,
        "vocab": root / F5_VOCAB_FILENAME,
        "metadata": root / F5_METADATA_FILENAME,
        "vocoder_config": vocoder / "config.yaml",
        "vocoder_model": vocoder / "pytorch_model.bin",
        "voices": root / "voices",
    }


def f5_model_status(models_root: Path) -> dict[str, Any]:
    paths = f5_model_paths(models_root)
    required = (
        paths["checkpoint"],
        paths["vocab"],
        paths["metadata"],
        paths["vocoder_config"],
        paths["vocoder_model"],
    )
    installed = all(path.is_file() and path.stat().st_size > 0 for path in required)
    any_present = any(path.exists() for path in required)
    metadata = _load_voice_json(paths["metadata"]) or {}
    return {
        "id": "f5-vietnamese-vivoice",
        "engine": F5_ENGINE,
        "name": "F5-TTS Vietnamese ViVoice",
        "installed": installed,
        "status": "ready" if installed else "error" if any_present else "not_installed",
        "path": str(paths["root"]),
        "source": F5_MODEL_ID,
        "version": str(metadata.get("version") or F5_MODEL_REVISION),
        "license": F5_MODEL_LICENSE,
        "commercialUse": False,
        "approxSizeBytes": 5_448_727_000,
    }


def f5_runtime_installed() -> bool:
    try:
        import importlib.util

        return (
            importlib.util.find_spec("f5_tts") is not None
            and importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("soundfile") is not None
        )
    except Exception:
        return False


def list_f5_voices(models_root: Path) -> list[VoiceInfo]:
    status = f5_model_status(models_root)
    if not status["installed"]:
        return []
    paths = f5_model_paths(models_root)
    voices_root = paths["voices"]
    if not voices_root.is_dir():
        return []
    voices: list[VoiceInfo] = []
    for entry in sorted(voices_root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        info = _f5_voice_from_dir(models_root, entry)
        if info is not None:
            voices.append(info)
    return voices


def resolve_f5_voice(models_root: Path, voice_id: str) -> Optional[VoiceInfo]:
    if not voice_id or "/" in voice_id or "\\" in voice_id or ".." in voice_id:
        return None
    entry = f5_model_paths(models_root)["voices"] / voice_id
    if not entry.is_dir():
        return None
    return _f5_voice_from_dir(models_root, entry)


def _f5_voice_from_dir(models_root: Path, entry: Path) -> Optional[VoiceInfo]:
    profile = _load_voice_json(entry / "voice.json")
    if not profile:
        return None
    reference_raw = profile.get("referenceAudio")
    reference_text = profile.get("referenceText")
    if not isinstance(reference_raw, str) or not reference_raw:
        return None
    if not isinstance(reference_text, str) or not reference_text.strip():
        return None
    reference = Path(reference_raw)
    if not reference.is_absolute():
        reference = entry / reference
    if not reference.is_file():
        return None
    paths = f5_model_paths(models_root)
    if not f5_model_status(models_root)["installed"]:
        return None
    reference_hash = str(profile.get("referenceSha256") or _file_identity(reference))
    identity = ":".join(
        (
            F5_MODEL_REVISION,
            _file_identity(paths["checkpoint"]),
            _file_identity(paths["vocab"]),
            _file_identity(paths["vocoder_config"]),
            _file_identity(paths["vocoder_model"]),
            reference_hash,
            _file_identity(reference),
        )
    )
    return VoiceInfo(
        id=str(profile.get("id") or entry.name),
        name=str(profile.get("name") or _prettify_voice_name(entry.name)),
        language=str(profile.get("language") or "vi-VN"),
        gender=str(profile.get("gender") or "unknown"),
        engine=F5_ENGINE,
        model_path=str(paths["checkpoint"]),
        config_path=str(paths["vocab"]),
        sample_rate=24000,
        installed=True,
        quality="quality",
        supported_settings=["speed", "volume", "device"],
        reference_audio_path=str(reference),
        reference_text=reference_text.strip(),
        model_name="F5-TTS-Vietnamese-ViVoice",
        model_version=F5_MODEL_REVISION,
        model_source=F5_MODEL_ID,
        license=F5_MODEL_LICENSE,
        commercial_use=False,
        cache_identity=identity,
        emotion=str(profile.get("emotion") or "neutral"),
        style=str(profile.get("style") or "default"),
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


def _infer_gender(dirname: str, pretty_name: str) -> str:
    blob = f"{dirname} {pretty_name}".lower()
    if any(token in blob for token in ("female", "woman", "girl", "vais")):
        return "female"
    if any(token in blob for token in ("male", "man", "boy")):
        return "male"
    return "unknown"


def _prettify_voice_name(dirname: str) -> str:
    parts = [p for p in dirname.replace("_", "-").split("-") if p]
    return " ".join(p.capitalize() for p in parts) if parts else dirname


def _file_identity(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return path.name


# --------------------------------------------------------------- aggregate


def list_all_voices(models_root: Path) -> list[VoiceInfo]:
    """Union of every engine's installed voices, sorted stably."""
    voices: list[VoiceInfo] = []
    voices.extend(list_piper_voices(models_root))
    voices.extend(list_f5_voices(models_root))
    voices.sort(key=lambda v: (v.language, v.engine, v.id.lower()))
    return voices


def resolve_voice(
    models_root: Path, engine: str, voice_id: str
) -> Optional[VoiceInfo]:
    if engine == "piper":
        return resolve_piper_voice(models_root, voice_id)
    if engine == F5_ENGINE:
        return resolve_f5_voice(models_root, voice_id)
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
            "supportedSettings": ["speed", "volume"],
        },
        {
            "id": F5_ENGINE,
            "name": "F5-TTS Vietnamese (local, high quality)",
            "available": f5_runtime_installed(),
            "supportedSettings": ["speed", "volume", "device"],
            "license": F5_MODEL_LICENSE,
            "commercialUse": False,
        },
    ]


# --------------------------------------------------------- recommended voices
#
# Curated list of Piper voices the app can auto-download when the
# user has none installed. Mirrors the Whisper (`stt/registry.py`)
# and translation (`translation/registry.py`) presets — the goal is
# that clicking "Generate all" for the very first time Just Works
# without the user hunting for a `.onnx` file.
#
# All entries live in the ``rhasspy/piper-voices`` HuggingFace repo,
# under the path convention::
#
#     <lang>/<lang_LOCALE>/<name>/<quality>/<file>.onnx
#     <lang>/<lang_LOCALE>/<name>/<quality>/<file>.onnx.json
#
# ``target_languages`` is the set of app-level target language codes
# (see ``TRANSLATION_LANGUAGES`` on the frontend) this voice covers.
# The frontend uses it to pick a default voice for the project's
# target language when auto-downloading.

_PIPER_HF_REPO = "rhasspy/piper-voices"

_RECOMMENDED_VOICES: dict[str, dict[str, Any]] = {
    "vi_VN-vais1000-medium": {
        "engine": "piper",
        "voice_id": "vi_VN-vais1000-medium",
        "language": "vi",
        "target_languages": ["vi"],
        "quality": "medium",
        "approx_size_bytes": 63_000_000,
        "label": "Piper — Vietnamese (vais1000, medium, ~63 MB)",
        "repo": _PIPER_HF_REPO,
        "hf_dir": "vi/vi_VN/vais1000/medium",
    },
    "en_US-libritts_r-medium": {
        "engine": "piper",
        "voice_id": "en_US-libritts_r-medium",
        "language": "en",
        "target_languages": ["en"],
        "quality": "medium",
        "approx_size_bytes": 63_000_000,
        "label": "Piper — English US (libritts_r, medium, ~63 MB)",
        "repo": _PIPER_HF_REPO,
        "hf_dir": "en/en_US/libritts_r/medium",
    },
    "zh_CN-huayan-medium": {
        "engine": "piper",
        "voice_id": "zh_CN-huayan-medium",
        "language": "zh",
        "target_languages": ["zh", "yue"],
        "quality": "medium",
        "approx_size_bytes": 63_000_000,
        "label": "Piper — Chinese (huayan, medium, ~63 MB)",
        "repo": _PIPER_HF_REPO,
        "hf_dir": "zh/zh_CN/huayan/medium",
    },
    "f5-vietnamese-vivoice": {
        "engine": F5_ENGINE,
        "voice_id": "",
        "language": "vi",
        "target_languages": ["vi"],
        "quality": "quality",
        "approx_size_bytes": 5_448_727_000,
        "label": "F5-TTS Vietnamese ViVoice (~5.45 GB, non-commercial)",
        "repo": F5_MODEL_ID,
        "revision": F5_MODEL_REVISION,
        "license": F5_MODEL_LICENSE,
        "commercial_use": False,
    },
}

# Voice used as fallback when the project's target language doesn't
# have a preset — Vietnamese is the app's primary target locale.
DEFAULT_RECOMMENDED_VOICE_PRESET = "vi_VN-vais1000-medium"


def recommended_voices() -> list[dict[str, Any]]:
    """Every preset the app knows how to auto-download.

    Wire shape (camelCase, matches Rust
    ``tts::RecommendedVoicePreset``): ``preset``, ``engine``,
    ``voiceId``, ``language``, ``targetLanguages``, ``quality``,
    ``approxSizeBytes``, ``label``, ``isDefault``.
    """
    out: list[dict[str, Any]] = []
    for preset, meta in _RECOMMENDED_VOICES.items():
        out.append(
            {
                "preset": preset,
                "engine": meta["engine"],
                "voiceId": meta["voice_id"],
                "language": meta["language"],
                "targetLanguages": list(meta["target_languages"]),
                "quality": meta["quality"],
                "approxSizeBytes": int(meta["approx_size_bytes"]),
                "label": meta["label"],
                "isDefault": preset == DEFAULT_RECOMMENDED_VOICE_PRESET,
                "license": meta.get("license"),
                "commercialUse": meta.get("commercial_use"),
            }
        )
    return out


def recommended_voice_meta(preset: str) -> Optional[dict[str, Any]]:
    return _RECOMMENDED_VOICES.get(preset)
