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

from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcError, RpcErrorCode
from ..rpc import HandlerContext
from . import registry
from .models import (
    BatchSegment,
    TTSSettings,
    build_segment_cache_key,
    text_hash,
    validate_batch,
)
from .piper_provider import PiperTTSProvider
from .provider import ProviderCancelled, ProviderError, TTSProvider

_MODELS_ROOT: Optional[Path] = None
_PROVIDERS: dict[str, TTSProvider] = {}


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
    global _MODELS_ROOT, _PROVIDERS
    _MODELS_ROOT = Path(models_root)
    _PROVIDERS = dict(providers) if providers else {}
    registry.ensure_tts_root(_MODELS_ROOT)


def _models_root() -> Path:
    if _MODELS_ROOT is None:
        raise RpcError(RpcErrorCode.INTERNAL, "tts handlers not configured")
    return _MODELS_ROOT


def _provider_for(engine: str) -> TTSProvider:
    if engine in _PROVIDERS:
        return _PROVIDERS[engine]
    if engine == "piper":
        provider = PiperTTSProvider(_models_root())
        _PROVIDERS[engine] = provider
        return provider
    raise RpcError(
        RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
        f"unknown TTS engine: {engine}",
    )


# ---------------------------------------------------------------- sync methods


def tts_env(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "engines": registry.supported_engines(),
        "modelsRoot": str(_models_root()),
        "ttsRoot": str(registry.tts_root(_models_root())),
        "piperInstalled": registry.piper_installed(),
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
    if str(meta["engine"]) != "piper":
        # Only Piper is wired up so far. Adding another engine here
        # is a matter of extending the branch — the download flow
        # itself is engine-agnostic.
        raise RpcError(
            RpcErrorCode.TTS_UNKNOWN_PRESET,
            f"auto-download is only supported for Piper voices right now (got {meta['engine']!r})",
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


def tts_synthesize_one(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    engine = _optional_str(params, "engine") or "piper"
    voice_id = _require_str(params, "voiceId")
    text = _require_str(params, "text")
    output_path = _require_str(params, "outputPath")
    settings = TTSSettings.from_dict(params.get("settings"))
    segment_id_raw = params.get("segmentId")
    segment_id = int(segment_id_raw) if isinstance(segment_id_raw, (int, float)) else -1

    provider = _provider_for(engine)
    if ctx.cancelled():
        raise RpcError(RpcErrorCode.CANCELLED, "cancelled before start")

    try:
        result = provider.synthesize(text, voice_id, output_path, settings)
    except ProviderCancelled as e:
        raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user") from e
    except ProviderError as e:
        raise RpcError(e.code, e.message, data={"recoverable": e.recoverable}) from e

    voice_info = registry.resolve_voice(_models_root(), engine, voice_id)
    model_name = Path(voice_info.model_path).name if voice_info else voice_id
    cache_key = build_segment_cache_key(
        engine=engine,
        voice_id=voice_id,
        model_name=model_name,
        text=text,
        settings=settings,
    )
    return {
        "ok": True,
        "segmentId": segment_id,
        "engine": engine,
        "voiceId": voice_id,
        "modelName": model_name,
        "cacheKey": cache_key,
        "textHash": text_hash(text),
        "text": text,
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
    project_dir = Path(project_root)
    voices_dir = project_dir / voices_subdir
    voices_dir.mkdir(parents=True, exist_ok=True)

    total = len(segments)
    generated = 0
    ctx.emit_progress(
        "tts.progress",
        {
            "stage": "starting",
            "fraction": 0.0,
            "completedSegments": 0,
            "totalSegments": total,
        },
    )

    # Cache voice metadata lookups so we don't scan the disk per segment.
    voice_cache: dict[str, Any] = {}

    for idx, seg in enumerate(segments):
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user")

        voice_id = seg.voice_id or default_voice_id
        settings = seg.settings or default_settings
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
                "stage": "synthesizing",
                "fraction": idx / total,
                "completedSegments": generated,
                "totalSegments": total,
                "currentSegmentId": seg.id,
            },
        )

        try:
            result = provider.synthesize(seg.text, voice_id, str(dst), settings)
        except ProviderCancelled as e:
            raise RpcError(RpcErrorCode.CANCELLED, "tts cancelled by user") from e
        except ProviderError as e:
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

        model_name = Path(voice_info.model_path).name
        cache_key = build_segment_cache_key(
            engine=engine,
            voice_id=voice_id,
            model_name=model_name,
            text=seg.text,
            settings=settings,
        )
        generated += 1

        ctx.emit_progress(
            "tts.segment_completed",
            {
                "segmentId": seg.id,
                "engine": engine,
                "voiceId": voice_id,
                "modelName": model_name,
                "cacheKey": cache_key,
                "textHash": text_hash(seg.text),
                "text": seg.text,
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
    try:
        provider.unload()
    except Exception:  # pragma: no cover - unload must never fail loudly
        pass

    ctx.emit_progress(
        "tts.progress",
        {
            "stage": "completed",
            "fraction": 1.0,
            "completedSegments": generated,
            "totalSegments": total,
        },
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
    for name, provider in list(_PROVIDERS.items()):
        try:
            provider.unload()
            released.append(name)
        except Exception as e:  # pragma: no cover
            log.warn("provider unload failed", provider=name, error=str(e))
    return {"ok": True, "released": released}


# ----------------------------------------------------------------- registration


def install(dispatcher) -> None:
    dispatcher.register("tts.env", tts_env)
    dispatcher.register("tts.list_voices", tts_list_voices)
    dispatcher.register("tts.list_recommended", tts_list_recommended)
    dispatcher.register("tts.unload", tts_unload)
    dispatcher.register_async("tts.synthesize_one", tts_synthesize_one)
    dispatcher.register_async("tts.synthesize_batch", tts_synthesize_batch)
    dispatcher.register_async("tts.download_voice", tts_download_voice)


# ------------------------------------------------------------------ helpers


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
