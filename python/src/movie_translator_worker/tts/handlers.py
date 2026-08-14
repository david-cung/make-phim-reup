"""RPC handlers for the TTS subsystem.

Registered by :func:`install` from :mod:`..handlers`.

Methods:

  * ``tts.env`` — sync — which engines are importable, where voice
    models live, whether Piper's runtime is installed.
  * ``tts.list_voices`` — sync — enumerate installed voices across
    every registered engine.
  * ``tts.synthesize_one`` — **async** — synthesise a single segment
    to disk and return the resulting WAV metadata. Used for the
    "Preview Voice" flow.
  * ``tts.synthesize_batch`` — **async, cancellable** — walk through
    a list of segments the host asked to (re)generate, emit a
    ``tts.segment_completed`` notification after each one so the host
    can persist the manifest incrementally, and a coarser
    ``tts.progress`` notification for the UI progress bar.

The host (Rust ``TtsService``) is authoritative for cache-hit
decisions: it filters segments the user already has and only sends
the outstanding subset. The worker never guesses whether something
is stale.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcError, RpcErrorCode
from ..rpc import HandlerContext, request_restart
from . import registry
from .hardware import f5_hardware_capability
from .models import (
    BatchSegment,
    TTSSettings,
    build_segment_cache_key,
    text_hash,
    validate_batch,
)
from .prosody import shorten_for_duration
from .manager import TTSManager
from .provider import ProviderCancelled, ProviderError, TTSProvider
from .text_normalization import normalize_tts_text

_MODELS_ROOT: Optional[Path] = None
_MANAGER: Optional[TTSManager] = None


def configure(
    *,
    models_root: Path,
    providers: Optional[dict[str, TTSProvider]] = None,
) -> None:
    """Wire up per-process state.

    ``providers`` is optional: tests inject fakes, production leaves
    the mapping empty and real providers are constructed lazily on
    first use so importing this module stays cheap.
    """
    global _MODELS_ROOT, _MANAGER
    _MODELS_ROOT = Path(models_root)
    _MANAGER = TTSManager(_MODELS_ROOT, providers)
    registry.ensure_tts_root(_MODELS_ROOT)


def _models_root() -> Path:
    if _MODELS_ROOT is None:
        raise RpcError(RpcErrorCode.INTERNAL, "tts handlers not configured")
    return _MODELS_ROOT


def _provider_for(engine: str) -> TTSProvider:
    if _MANAGER is None:
        raise RpcError(RpcErrorCode.INTERNAL, "tts handlers not configured")
    return _MANAGER.provider(engine)


# ---------------------------------------------------------------- sync methods


def tts_env(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "engines": registry.supported_engines(),
        "modelsRoot": str(_models_root()),
        "ttsRoot": str(registry.tts_root(_models_root())),
        "piperInstalled": registry.piper_installed(),
        "f5RuntimeInstalled": registry.f5_runtime_installed(),
        "f5Model": registry.f5_model_status(_models_root()),
        "f5Hardware": f5_hardware_capability(),
        "defaultEngine": "piper",
    }


def tts_list_voices(_params: dict[str, Any]) -> dict[str, Any]:
    voices = registry.list_all_voices(_models_root())
    return {"voices": [v.to_dict() for v in voices]}


def tts_list_recommended(_params: dict[str, Any]) -> dict[str, Any]:
    """Phase 12 — voices the app can auto-download when the user
    hasn't installed any. Mirrors ``stt.list_models`` /
    ``translate.list_recommended``.
    """
    return {"presets": registry.recommended_voices()}


# --------------------------------------------------------------- async methods


def tts_download_voice(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    """Phase 12 — pull a Piper voice pair (`.onnx` + `.onnx.json`)
    from HuggingFace into ``<models>/tts/piper/<voice_id>/``.

    Emits ``tts.download_progress`` keyed on the request id so the
    Rust host can drive a progress bar just like it does for STT
    and translation model downloads. Idempotent: if both files
    already exist locally, returns a completed payload without
    hitting the network.
    """
    preset = _require_str(params, "preset")
    meta = registry.recommended_voice_meta(preset)
    if meta is None:
        raise RpcError(
            RpcErrorCode.TTS_UNKNOWN_PRESET,
            f"unknown TTS voice preset: {preset}",
        )
    if str(meta["engine"]) == registry.F5_ENGINE:
        return _download_f5_model(preset, ctx)
    if str(meta["engine"]) != "piper":
        raise RpcError(
            RpcErrorCode.TTS_UNKNOWN_PRESET,
            f"unsupported local TTS preset engine: {meta['engine']!r}",
        )
    voice_id = str(meta["voice_id"])
    repo = str(meta["repo"])
    hf_dir = str(meta["hf_dir"]).strip("/")
    onnx_name = f"{voice_id}.onnx"
    json_name = f"{voice_id}.onnx.json"
    onnx_hf_path = f"{hf_dir}/{onnx_name}"
    json_hf_path = f"{hf_dir}/{json_name}"

    root = _models_root()
    piper_root = registry.engine_root(root, "piper")
    voice_dir = piper_root / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx_dest = voice_dir / onnx_name
    json_dest = voice_dir / json_name

    def _progress(fraction: float, stage: str) -> None:
        ctx.emit_progress(
            "tts.download_progress",
            {
                "fraction": max(0.0, min(1.0, float(fraction))),
                "stage": stage,
            },
        )

    if (
        onnx_dest.is_file()
        and onnx_dest.stat().st_size > 0
        and json_dest.is_file()
        and json_dest.stat().st_size > 0
    ):
        _progress(1.0, "already_installed")
        return {
            "ok": True,
            "preset": preset,
            "voiceId": voice_id,
            "engine": "piper",
            "modelPath": str(onnx_dest),
            "configPath": str(json_dest),
            "sizeBytes": onnx_dest.stat().st_size,
            "alreadyInstalled": True,
        }

    _progress(0.0, "starting")

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            "huggingface_hub is not installed; add the [tts] extra to the worker environment",
        ) from e

    try:
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user")
        # Config first — it's ~1 KB and gives us cheap early feedback
        # if the repo path is wrong (misspelled voice name).
        _progress(0.05, "downloading_config")
        cfg_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=json_hf_path,
                local_dir=str(voice_dir),
            )
        )
        _move_into_place(cfg_path, json_dest)

        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user")

        _progress(0.15, "downloading_model")
        model_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=onnx_hf_path,
                local_dir=str(voice_dir),
            )
        )
        _move_into_place(model_path, onnx_dest)

        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user")

        _progress(1.0, "downloaded")
    except RpcError:
        _cleanup_voice(voice_dir, onnx_dest, json_dest)
        raise
    except Exception as e:
        _cleanup_voice(voice_dir, onnx_dest, json_dest)
        raise RpcError(
            RpcErrorCode.TTS_DOWNLOAD_FAILED,
            f"failed to download voice {voice_id!r} from {repo!r}: {e}",
        ) from e

    return {
        "ok": True,
        "preset": preset,
        "voiceId": voice_id,
        "engine": "piper",
        "modelPath": str(onnx_dest),
        "configPath": str(json_dest),
        "sizeBytes": onnx_dest.stat().st_size if onnx_dest.is_file() else 0,
        "alreadyInstalled": False,
    }


def _move_into_place(src: Path, dest: Path) -> None:
    """``hf_hub_download`` with ``local_dir`` may write to a nested
    subpath (mirroring the repo structure). We flatten by moving
    into the flat ``voice_dir/<file>`` layout the registry expects.
    """
    if src == dest:
        return
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)


def _cleanup_voice(voice_dir: Path, *files: Path) -> None:
    """Remove zero-byte / half-written outputs so a retry starts
    fresh instead of picking up a broken half-file. Only removes
    what we were about to write — never touches unrelated content
    already in the voice directory.
    """
    for f in files:
        try:
            if f.exists() and f.stat().st_size == 0:
                f.unlink()
        except OSError:
            pass
    try:
        # Only nuke the dir if we ended up creating it empty.
        if voice_dir.is_dir() and not any(voice_dir.iterdir()):
            voice_dir.rmdir()
    except OSError:
        pass


def _python_package_root() -> Path:
    """Directory that contains ``pyproject.toml`` (the editable worker)."""
    env = os.environ.get("LMT_WORKER_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def _ensure_f5_runtime(ctx: HandlerContext, progress) -> None:
    """Install the worker ``[f5]`` extra when F5-TTS is not importable."""
    if registry.f5_runtime_installed():
        return
    root = _python_package_root()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            f"cannot install F5-TTS: worker package root {root} has no pyproject.toml",
        )
    progress(0.02, "installing_f5_runtime")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        f"{root}[f5]",
    ]
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONUTF8"] = "1"
    log.info("installing f5 runtime", python=sys.executable, root=str(root))
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": env,
    }
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except OSError as exc:
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            f"failed to start pip to install F5-TTS: {exc}",
        ) from exc

    def _kill() -> None:
        try:
            if sys.platform != "win32" and proc.pid:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.kill()
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def _pump() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_pump, daemon=True).start()
    fraction = 0.02
    try:
        while True:
            if ctx.cancelled():
                _kill()
                raise RpcError(RpcErrorCode.CANCELLED, "F5 runtime install cancelled")
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            text = line.strip()
            if text:
                log.info("f5 pip", line=text[:300])
                fraction = min(0.12, fraction + 0.004)
                progress(fraction, "installing_f5_runtime")
        code = proc.wait()
    except RpcError:
        _kill()
        raise
    except Exception:
        _kill()
        proc.wait(timeout=10)
        raise
    if code != 0:
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            "failed to install the F5-TTS runtime into the worker environment",
        )
    importlib.invalidate_caches()
    installed = registry.f5_runtime_installed()
    # pip replaced packages (numpy, tokenizers, ...) inside the very
    # environment this process runs from, so half of `sys.modules` is now
    # stale. Reusing this process poisons unrelated engines — Whisper
    # loads have failed with `RecursionError` this way. Hand the job back
    # and let the host respawn a clean worker.
    request_restart("installed the F5-TTS runtime into the running environment")
    raise RpcError(
        RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
        "F5-TTS runtime installed. Restarting the local engine — press Download again to continue."
        if installed
        else "failed to install the F5-TTS runtime into the worker environment",
        data={"recoverable": True, "restarting": True, "runtimeInstalled": installed},
    )


def _download_f5_model(
    preset: str,
    ctx: HandlerContext,
) -> dict[str, Any]:
    paths = registry.f5_model_paths(_models_root())

    def progress(fraction: float, stage: str) -> None:
        ctx.emit_progress(
            "tts.download_progress",
            {"fraction": max(0.0, min(1.0, fraction)), "stage": stage},
        )

    _ensure_f5_runtime(ctx, progress)
    status = registry.f5_model_status(_models_root())

    if status["installed"]:
        progress(1.0, "already_installed")
        return {
            "ok": True,
            "preset": preset,
            "voiceId": "",
            "engine": registry.F5_ENGINE,
            "modelPath": str(paths["checkpoint"]),
            "configPath": str(paths["vocab"]),
            "sizeBytes": paths["checkpoint"].stat().st_size,
            "alreadyInstalled": True,
        }
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            "huggingface_hub is required to install the F5-TTS model",
        ) from exc

    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["vocoder_config"].parent.mkdir(parents=True, exist_ok=True)
    progress(0.0, "preparing")
    try:
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "model installation cancelled")
        progress(0.03, "downloading_vocabulary")
        vocab_source = Path(
            hf_hub_download(
                repo_id=registry.F5_MODEL_ID,
                revision=registry.F5_MODEL_REVISION,
                filename="config.json",
                local_dir=str(paths["root"]),
            )
        )
        _move_into_place(vocab_source, paths["vocab"])

        progress(0.06, "downloading_vocoder")
        vocoder_config = Path(
            hf_hub_download(
                repo_id=registry.F5_VOCODER_REPO,
                revision=registry.F5_VOCODER_REVISION,
                filename="config.yaml",
                local_dir=str(paths["vocoder_config"].parent),
            )
        )
        _move_into_place(vocoder_config, paths["vocoder_config"])
        vocoder_model = Path(
            hf_hub_download(
                repo_id=registry.F5_VOCODER_REPO,
                revision=registry.F5_VOCODER_REVISION,
                filename="pytorch_model.bin",
                local_dir=str(paths["vocoder_model"].parent),
            )
        )
        _move_into_place(vocoder_model, paths["vocoder_model"])

        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "model installation cancelled")
        progress(0.10, "downloading_model_5_39gb")
        checkpoint_source = Path(
            hf_hub_download(
                repo_id=registry.F5_MODEL_ID,
                revision=registry.F5_MODEL_REVISION,
                filename=registry.F5_MODEL_FILENAME,
                local_dir=str(paths["root"]),
            )
        )
        _move_into_place(checkpoint_source, paths["checkpoint"])
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "model installation cancelled")

        metadata = {
            "id": "f5-vietnamese-vivoice",
            "name": "F5-TTS Vietnamese ViVoice",
            "version": registry.F5_MODEL_REVISION,
            "source": registry.F5_MODEL_ID,
            "license": registry.F5_MODEL_LICENSE,
            "commercialUse": False,
            "checkpointSha256": "5ae8293dd09868d5758cd1edc6b74f53bd0200652d907bd43724a69c7b82ea1f",
            "vocoderSource": registry.F5_VOCODER_REPO,
            "vocoderLicense": "MIT",
        }
        tmp_metadata = paths["metadata"].with_suffix(".json.tmp")
        tmp_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_metadata.replace(paths["metadata"])
        progress(1.0, "installed")
    except RpcError:
        raise
    except Exception as exc:
        raise RpcError(
            RpcErrorCode.TTS_DOWNLOAD_FAILED,
            f"failed to install F5-TTS Vietnamese model: {exc}",
        ) from exc
    return {
        "ok": True,
        "preset": preset,
        "voiceId": "",
        "engine": registry.F5_ENGINE,
        "modelPath": str(paths["checkpoint"]),
        "configPath": str(paths["vocab"]),
        "sizeBytes": paths["checkpoint"].stat().st_size,
        "alreadyInstalled": False,
    }


def tts_create_voice_profile(
    params: dict[str, Any], _ctx: HandlerContext
) -> dict[str, Any]:
    profile_id = _slug(_require_str(params, "id"))
    name = _require_str(params, "name").strip()
    gender = (_optional_str(params, "gender") or "unknown").lower()
    emotion = (_optional_str(params, "emotion") or "neutral").lower()
    style = (_optional_str(params, "style") or "default").lower()
    reference_source = Path(_require_str(params, "referenceAudioPath"))
    reference_text = normalize_tts_text(_require_str(params, "referenceText"))
    if not reference_text:
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            "reference transcript cannot be blank",
        )
    if not registry.f5_model_status(_models_root())["installed"]:
        raise RpcError(
            RpcErrorCode.TTS_VOICE_MISSING,
            "install the F5-TTS Vietnamese model before creating a reference voice",
        )
    if not reference_source.is_file():
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"reference audio does not exist: {reference_source}",
        )
    if reference_source.suffix.lower() != ".wav":
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            "F5 reference audio must be a WAV file",
        )
    try:
        with wave.open(str(reference_source), "rb") as reader:
            duration = reader.getnframes() / max(1, reader.getframerate())
    except (OSError, wave.Error) as exc:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"invalid reference WAV: {exc}") from exc
    if duration < 1.0 or duration > 30.0:
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            "reference audio must be between 1 and 30 seconds (under 10 seconds is recommended)",
        )
    voice_dir = registry.f5_model_paths(_models_root())["voices"] / profile_id
    if voice_dir.exists():
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"voice profile {profile_id!r} already exists",
        )
    voice_dir.mkdir(parents=True, exist_ok=False)
    destination = voice_dir / "reference.wav"
    try:
        tmp_audio = voice_dir / "reference.wav.tmp"
        shutil.copy2(reference_source, tmp_audio)
        tmp_audio.replace(destination)
        reference_sha = _sha256_file(destination)
        profile = {
            "id": profile_id,
            "name": name,
            "language": "vi-VN",
            "gender": gender if gender in {"male", "female", "neutral"} else "unknown",
            "provider": registry.F5_ENGINE,
            "referenceAudio": destination.name,
            "referenceText": reference_text,
            "referenceSha256": reference_sha,
            "model": "F5-TTS-Vietnamese-ViVoice",
            "modelVersion": registry.F5_MODEL_REVISION,
            "license": registry.F5_MODEL_LICENSE,
            "commercialUse": False,
            "emotion": emotion
            if emotion
            in {
                "neutral",
                "happy",
                "sad",
                "angry",
                "afraid",
                "surprised",
                "serious",
                "excited",
                "calm",
                "whisper",
            }
            else "neutral",
            "style": style[:32],
        }
        tmp_json = voice_dir / "voice.json.tmp"
        tmp_json.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_json.replace(voice_dir / "voice.json")
    except Exception:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise
    voice = registry.resolve_f5_voice(_models_root(), profile_id)
    if voice is None:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise RpcError(RpcErrorCode.TTS_MODEL_INVALID, "created voice profile is invalid")
    return {"ok": True, "voice": voice.to_dict()}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    if not slug:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "voice profile id is empty")
    return slug[:64]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tts_synthesize_one(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    engine = _optional_str(params, "engine") or "piper"
    voice_id = _require_str(params, "voiceId")
    source_text = _require_str(params, "text").strip()
    text = normalize_tts_text(source_text)
    output_path = _require_str(params, "outputPath")
    settings = TTSSettings.from_dict(params.get("settings"))
    segment_id_raw = params.get("segmentId")
    segment_id = int(segment_id_raw) if isinstance(segment_id_raw, (int, float)) else -1

    provider = _provider_for(engine)
    _set_cancel_checker(provider, ctx.cancelled)
    if ctx.cancelled():
        raise RpcError(RpcErrorCode.CANCELLED, "cancelled before start")

    try:
        result = _synthesize_fitting(
            provider,
            text=text,
            voice_id=voice_id,
            output_path=output_path,
            settings=settings,
            target_duration=params.get("targetDurationSecs"),
        )
    except ProviderCancelled as e:
        raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user") from e
    except ProviderError as e:
        raise RpcError(e.code, e.message, data={"recoverable": e.recoverable}) from e

    voice_info = registry.resolve_voice(_models_root(), engine, voice_id)
    model_name = (
        voice_info.cache_identity or Path(voice_info.model_path).name
        if voice_info
        else voice_id
    )
    cache_key = build_segment_cache_key(
        engine=engine,
        voice_id=voice_id,
        model_name=model_name,
        text=text,
        settings=settings,
        voice_identity=model_name,
    )
    return {
        "ok": True,
        "segmentId": segment_id,
        "engine": engine,
        "voiceId": voice_id,
        "modelName": model_name,
        "cacheKey": cache_key,
        "textHash": text_hash(source_text),
        "text": source_text,
        "file": output_path,
        "durationSecs": result.duration_secs,
        "sampleRate": result.sample_rate,
        "channels": result.channels,
        "sizeBytes": result.size_bytes,
        "settings": settings.normalised().to_dict(),
    }


def tts_synthesize_batch(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    try:
        return _tts_synthesize_batch_impl(params, ctx)
    finally:
        if _MANAGER is not None:
            for _name, loaded_provider in _MANAGER.items():
                _safe_unload(loaded_provider)
                _set_cancel_checker(loaded_provider, lambda: False)


def _tts_synthesize_batch_impl(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    engine = _optional_str(params, "engine") or "piper"
    default_voice_id = _require_str(params, "defaultVoiceId")
    default_settings = TTSSettings.from_dict(params.get("settings"))
    project_root = _require_str(params, "projectRoot")
    voices_subdir = _optional_str(params, "voicesSubdir") or "voices"

    raw_segments = params.get("segments")
    if not isinstance(raw_segments, list):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "segments must be a list")
    if not raw_segments:
        ctx.emit_progress(
            "tts.progress",
            {
                "stage": "completed",
                "fraction": 1.0,
                "completedSegments": 0,
                "totalSegments": 0,
            },
        )
        return {"ok": True, "engine": engine, "totalSegments": 0, "generatedSegments": 0}

    try:
        segments = [BatchSegment.from_wire(s) for s in raw_segments]
        validate_batch(segments)
    except ValueError as e:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, str(e)) from e

    provider = _provider_for(engine)
    _set_cancel_checker(provider, ctx.cancelled)
    project_dir = Path(project_root)
    voices_dir = project_dir / voices_subdir
    voices_dir.mkdir(parents=True, exist_ok=True)

    total = len(segments)
    generated = 0
    log.info(
        "tts batch start",
        engine=engine,
        default_voice=default_voice_id,
        segments=total,
        project_root=str(project_dir),
    )
    ctx.emit_progress(
        "tts.progress",
        {
            "stage": "preparing",
            "fraction": 0.0,
            "completedSegments": 0,
            "totalSegments": total,
        },
    )

    # Cache voice metadata lookups so we don't scan the disk per segment.
    voice_cache: dict[str, Any] = {}

    for idx, seg in enumerate(segments):
        if ctx.cancelled():
            _safe_unload(provider)
            raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user")

        voice_id = seg.voice_id or default_voice_id
        settings = seg.settings or default_settings
        source_text = seg.text.strip()
        normalized_text = normalize_tts_text(source_text)
        if voice_id not in voice_cache:
            voice_cache[voice_id] = registry.resolve_voice(
                _models_root(), engine, voice_id
            )
        voice_info = voice_cache[voice_id]
        if voice_info is None:
            raise RpcError(
                RpcErrorCode.TTS_VOICE_MISSING,
                f"voice {voice_id!r} is not installed on engine {engine!r}",
                data={"voiceId": voice_id, "engine": engine, "segmentId": seg.id},
            )

        rel = f"{voices_subdir}/{seg.id:06d}.wav"
        dst = voices_dir / f"{seg.id:06d}.wav"

        ctx.emit_progress(
            "tts.progress",
            {
                "stage": "loading_model" if idx == 0 else "generating_voice",
                "fraction": idx / total,
                "completedSegments": generated,
                "totalSegments": total,
                "currentSegmentId": seg.id,
            },
        )

        try:
            result = _synthesize_fitting(
                provider,
                text=normalized_text,
                voice_id=voice_id,
                output_path=str(dst),
                settings=settings,
                target_duration=seg.target_duration_secs,
            )
        except ProviderCancelled as e:
            _safe_unload(provider)
            raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user") from e
        except ProviderError as e:
            _safe_unload(provider)
            log.warn(
                "tts segment failed",
                segment=seg.id,
                voice=voice_id,
                code=e.code,
                message=e.message,
            )
            raise RpcError(
                e.code,
                e.message,
                data={
                    "recoverable": e.recoverable,
                    "segmentId": seg.id,
                    "voiceId": voice_id,
                    "engine": engine,
                },
            ) from e

        model_name = voice_info.cache_identity or Path(voice_info.model_path).name
        cache_key = build_segment_cache_key(
            engine=engine,
            voice_id=voice_id,
            model_name=model_name,
            text=normalized_text,
            settings=settings,
            voice_identity=model_name,
        )
        generated += 1
        log.info(
            "tts segment complete",
            segment=seg.id,
            voice=voice_id,
            text_chars=len(normalized_text),
            output=str(dst),
            size_bytes=result.size_bytes,
            duration_secs=round(result.duration_secs, 3),
            sample_rate=result.sample_rate,
            channels=result.channels,
        )

        ctx.emit_progress(
            "tts.segment_completed",
            {
                "segmentId": seg.id,
                "engine": engine,
                "voiceId": voice_id,
                "modelName": model_name,
                "cacheKey": cache_key,
                "textHash": text_hash(source_text),
                "text": source_text,
                "file": rel,
                "durationSecs": result.duration_secs,
                "sampleRate": result.sample_rate,
                "channels": result.channels,
                "sizeBytes": result.size_bytes,
                "settings": settings.normalised().to_dict(),
                "completedSegments": generated,
                "totalSegments": total,
                "fraction": generated / total,
            },
        )

    # Release the model between explicit batches so long-idle projects
    # don't hold hundreds of MB.
    _safe_unload(provider)

    ctx.emit_progress(
        "tts.progress",
        {
            "stage": "completed",
            "fraction": 1.0,
            "completedSegments": generated,
            "totalSegments": total,
        },
    )
    log.info(
        "tts batch complete",
        engine=engine,
        generated_segments=generated,
        total_segments=total,
    )

    return {
        "ok": True,
        "engine": engine,
        "totalSegments": total,
        "generatedSegments": generated,
    }


def tts_unload(_params: dict[str, Any]) -> dict[str, Any]:
    """Explicitly release every loaded engine's runtime state."""
    released: list[str] = []
    if _MANAGER is None:
        return {"ok": True, "released": released}
    for name, provider in _MANAGER.items():
        try:
            provider.unload()
            released.append(name)
        except Exception as e:  # pragma: no cover
            log.warn("provider unload failed", provider=name, error=str(e))
    return {"ok": True, "released": released}


def _safe_unload(provider: TTSProvider) -> None:
    try:
        provider.unload()
    except Exception:  # pragma: no cover - unload must never fail loudly
        pass


def _set_cancel_checker(provider: TTSProvider, checker) -> None:
    setter = getattr(provider, "set_cancel_checker", None)
    if callable(setter):
        setter(checker)


# ----------------------------------------------------------------- registration


def install(dispatcher) -> None:
    dispatcher.register("tts.env", tts_env)
    dispatcher.register("tts.list_voices", tts_list_voices)
    dispatcher.register("tts.list_recommended", tts_list_recommended)
    dispatcher.register("tts.unload", tts_unload)
    dispatcher.register_async("tts.synthesize_one", tts_synthesize_one)
    dispatcher.register_async("tts.synthesize_batch", tts_synthesize_batch)
    dispatcher.register_async("tts.download_voice", tts_download_voice)
    dispatcher.register_async("tts.create_voice_profile", tts_create_voice_profile)


# ------------------------------------------------------------------ helpers


def _synthesize_fitting(
    provider,
    *,
    text: str,
    voice_id: str,
    output_path: str,
    settings: TTSSettings,
    target_duration: Any,
) -> Any:
    """Synthesize, then if the line overruns its subtitle window:

    1. shorten filler-heavy spoken text
    2. apply a small speed increase (≤ 1.12×)
    3. stop — never ram the voice to fit at any cost
    """
    result = provider.synthesize(text, voice_id, output_path, settings)
    target = None
    if isinstance(target_duration, (int, float)) and float(target_duration) > 0.2:
        target = float(target_duration)
    if target is None or result.duration_secs <= target * 1.06:
        return result

    ratio = result.duration_secs / target
    spoken = shorten_for_duration(text, ratio) or text
    next_settings = settings
    if ratio > 1.08:
        faster = min(1.12, settings.normalised().speed * min(ratio, 1.12))
        next_settings = TTSSettings(
            speed=faster,
            pitch=settings.pitch,
            volume=settings.volume,
            device=settings.device,
        )
    if spoken != text or abs(next_settings.speed - settings.speed) > 1e-3:
        result = provider.synthesize(spoken, voice_id, output_path, next_settings)
    return result


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"missing required string param: {key}")
    return value


def _optional_str(params: dict[str, Any], key: str) -> Optional[str]:
    value = params.get(key)
    if isinstance(value, str) and value:
        return value
    return None
