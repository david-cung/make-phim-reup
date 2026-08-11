"""RPC handlers for the STT subsystem.

Registered by :func:`install` from :mod:`..handlers`. Kept in a
separate module so the heavy lifting stays out of the top-level
handler file.

Methods:

  * ``stt.env`` — device list + whether ``faster-whisper`` is
    importable (sync, fast).
  * ``stt.list_models`` — enumerate the known Whisper models with an
    ``installed`` flag (sync, fast).
  * ``stt.download_model`` — download a model into
    ``<models_root>/whisper/<name>/`` (**async**, cancellable, emits
    ``stt.download_progress`` notifications).
  * ``stt.transcribe`` — run inference (**async**, cancellable,
    emits ``stt.progress`` notifications). Returns
    ``(detected_language, segments)`` for the host to persist.
  * ``stt.remove_model`` — delete a downloaded snapshot (sync).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcError, RpcErrorCode
from ..rpc import HandlerContext
from . import registry
from .device import default_device, detect_devices
from .faster_whisper_provider import FasterWhisperProvider
from .models import (
    Segment,
    TranscribeOptions,
    build_cache_key,
    validate_segments,
)
from .provider import (
    ProgressCallback,
    ProviderCancelled,
    ProviderError,
    SpeechToTextProvider,
    TranscribeContext,
)

# Set by :func:`configure` at startup so handlers can locate model files.
_MODELS_ROOT: Optional[Path] = None
_PROVIDER: Optional[SpeechToTextProvider] = None


def configure(*, models_root: Path, provider: Optional[SpeechToTextProvider] = None) -> None:
    """Wire up per-process state.

    ``provider`` is optional: tests inject a fake, production leaves it
    ``None`` and the real :class:`FasterWhisperProvider` is created
    lazily on first use.
    """
    global _MODELS_ROOT, _PROVIDER
    _MODELS_ROOT = Path(models_root)
    _PROVIDER = provider
    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)


def _models_root() -> Path:
    if _MODELS_ROOT is None:
        raise RpcError(RpcErrorCode.INTERNAL, "stt handlers not configured")
    return _MODELS_ROOT


def _provider() -> SpeechToTextProvider:
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    _PROVIDER = FasterWhisperProvider(_models_root())
    return _PROVIDER


# --------------------------------------------------------------------- sync


def stt_env(_params: dict[str, Any]) -> dict[str, Any]:
    devices = detect_devices()
    return {
        "devices": [d.to_dict() for d in devices],
        "defaultDevice": default_device(devices),
        "whisperInstalled": _whisper_installed(),
        "modelsRoot": str(_models_root()),
    }


def stt_list_models(_params: dict[str, Any]) -> dict[str, Any]:
    root = _models_root()
    models = registry.list_models(root)
    return {"models": [m.to_dict() for m in models]}


def stt_remove_model(params: dict[str, Any]) -> dict[str, Any]:
    name = _require_str(params, "name")
    if not registry.is_known(name):
        raise RpcError(RpcErrorCode.STT_UNKNOWN_MODEL, f"unknown whisper model: {name}")
    registry.remove_model(_models_root(), name)
    return {"ok": True, "name": name}


# ------------------------------------------------------------------- async


def stt_transcribe(params: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
    audio_path = _require_str(params, "audioPath")
    options = _options_from_params(params)

    if not registry.is_known(options.model):
        raise RpcError(RpcErrorCode.STT_UNKNOWN_MODEL, f"unknown whisper model: {options.model}")
    root = _models_root()
    if not registry.is_installed(root, options.model):
        raise RpcError(
            RpcErrorCode.STT_MODEL_NOT_INSTALLED,
            f"whisper model {options.model!r} is not installed; download it first",
            data={"model": options.model},
        )
    if not Path(audio_path).is_file():
        raise RpcError(RpcErrorCode.STT_INVALID_AUDIO, f"audio file not found: {audio_path}")

    provider = _provider()

    def on_progress(fraction: float, stage: str, message: Optional[str] = None) -> None:
        ctx.emit_progress(
            "stt.progress",
            {
                "fraction": max(0.0, min(1.0, float(fraction))),
                "stage": stage,
                "message": message,
            },
        )

    class _Ctx(TranscribeContext):
        def cancelled(self) -> bool:
            return ctx.cancelled()

        def on_progress(self, fraction: float, stage: str, message: Optional[str] = None) -> None:  # noqa: D401
            on_progress(fraction, stage, message)

    on_progress(0.0, "queued", None)
    try:
        language, segments = provider.transcribe(audio_path, options, _Ctx())
    except ProviderCancelled as e:
        raise RpcError(RpcErrorCode.CANCELLED, "transcription cancelled by user") from e
    except ProviderError as e:
        raise RpcError(e.code, e.message, data={"recoverable": e.recoverable}) from e

    if ctx.cancelled():
        raise RpcError(RpcErrorCode.CANCELLED, "transcription cancelled by user")

    validate_segments(segments)
    on_progress(1.0, "completed", None)

    cache_key = build_cache_key(_require_str(params, "audioHash"), options)
    return {
        "language": language,
        "segments": [_segment_to_wire(s) for s in segments],
        "cacheKey": cache_key,
        "model": options.model,
        "device": options.device,
        "computeType": options.compute_type,
        "wordTimestamps": options.word_timestamps,
        "options": _options_to_wire(options),
    }


def stt_download_model(params: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
    name = _require_str(params, "name")
    if not registry.is_known(name):
        raise RpcError(RpcErrorCode.STT_UNKNOWN_MODEL, f"unknown whisper model: {name}")

    root = _models_root()
    if registry.is_installed(root, name):
        ctx.emit_progress("stt.download_progress", {"stage": "already_installed", "fraction": 1.0})
        return {
            "ok": True,
            "name": name,
            "path": str(registry.model_dir(root, name)),
            "sizeBytes": registry.snapshot_size(registry.model_dir(root, name)),
            "alreadyInstalled": True,
        }

    def _progress(fraction: float, stage: str) -> None:
        ctx.emit_progress(
            "stt.download_progress",
            {"fraction": max(0.0, min(1.0, float(fraction))), "stage": stage},
        )

    try:
        _progress(0.0, "starting")
        registry.ensure_downloaded(
            root, name, hf_downloader=_hf_download, progress=_check_cancel(ctx, _progress),
        )
    except ProviderCancelled as e:
        registry.remove_model(root, name)
        raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user") from e
    except RpcError:
        raise
    except Exception as e:
        registry.remove_model(root, name)
        raise RpcError(
            RpcErrorCode.STT_DOWNLOAD_FAILED,
            f"failed to download {name!r}: {e}",
        ) from e

    dest = registry.model_dir(root, name)
    return {
        "ok": True,
        "name": name,
        "path": str(dest),
        "sizeBytes": registry.snapshot_size(dest),
        "alreadyInstalled": False,
    }


def stt_unload(_params: dict[str, Any]) -> dict[str, Any]:
    """Phase 11 — release the resident Whisper model so long idle
    periods don't hold multi-GB of tensors in RAM. Called by the
    Rust "unload all" surface and by the per-stage auto-unloader
    after a transcribe job settles.
    """
    provider = _PROVIDER
    released = False
    if provider is not None and hasattr(provider, "unload"):
        try:
            released = bool(provider.unload())
        except Exception as e:  # pragma: no cover - unload must not throw
            log.warn("stt provider unload failed", error=str(e))
    return {"ok": True, "released": released}


# ----------------------------------------------------------------- registration


def install(dispatcher) -> None:
    dispatcher.register("stt.env", stt_env)
    dispatcher.register("stt.list_models", stt_list_models)
    dispatcher.register("stt.remove_model", stt_remove_model)
    dispatcher.register("stt.unload", stt_unload)
    dispatcher.register_async("stt.transcribe", stt_transcribe)
    dispatcher.register_async("stt.download_model", stt_download_model)


# ------------------------------------------------------------------ helpers


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"missing required string param: {key}")
    return value


def _options_from_params(params: dict[str, Any]) -> TranscribeOptions:
    opts = params.get("options") or {}
    if not isinstance(opts, dict):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "options must be an object")
    language = opts.get("language")
    if language is not None and not isinstance(language, str):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "options.language must be a string or null")
    lang_norm: Optional[str]
    if language is None or language.lower() in ("", "auto"):
        lang_norm = None
    else:
        lang_norm = language

    devices = detect_devices()
    device = opts.get("device") or default_device(devices)
    valid_devices = {d.kind for d in devices if d.supported}
    if device not in valid_devices:
        # Silently downgrade to a supported device rather than fail: the
        # UI may hold a stale value if the host moved between machines.
        log.warn("stt device downgraded", requested=device, using="cpu")
        device = "cpu"
    compute_type = opts.get("computeType") or ("float16" if device == "cuda" else "int8")

    try:
        return TranscribeOptions(
            model=str(opts.get("model") or "small"),
            language=lang_norm,
            device=str(device),
            compute_type=str(compute_type),
            beam_size=int(opts.get("beamSize", 5)),
            word_timestamps=bool(opts.get("wordTimestamps", False)),
            vad_filter=bool(opts.get("vadFilter", False)),
            initial_prompt=(opts.get("initialPrompt") or None) or None,
            temperature=float(opts.get("temperature", 0.0)),
        )
    except (TypeError, ValueError) as e:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"invalid options: {e}") from e


def _options_to_wire(options: TranscribeOptions) -> dict[str, Any]:
    return {
        "model": options.model,
        "language": options.language,
        "device": options.device,
        "computeType": options.compute_type,
        "beamSize": options.beam_size,
        "wordTimestamps": options.word_timestamps,
        "vadFilter": options.vad_filter,
        "initialPrompt": options.initial_prompt,
        "temperature": options.temperature,
    }


def _segment_to_wire(s: Segment) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": s.id,
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "text": s.text,
    }
    if s.avg_logprob is not None:
        out["avgLogprob"] = s.avg_logprob
    if s.no_speech_prob is not None:
        out["noSpeechProb"] = s.no_speech_prob
    if s.words:
        out["words"] = [
            {
                "word": w.word,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                **({"probability": w.probability} if w.probability is not None else {}),
            }
            for w in s.words
        ]
    return out


def _whisper_installed() -> bool:
    try:
        import importlib
        importlib.util.find_spec("faster_whisper")  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _check_cancel(ctx: HandlerContext, cb) -> ProgressCallback:
    """Wraps a progress callback so we bail out promptly on cancel."""

    def _inner(fraction: float, stage: str) -> None:
        if ctx.cancelled():
            raise ProviderCancelled()
        cb(fraction, stage)

    return _inner  # type: ignore[return-value]


def _hf_download(repo_id: str, local_dir: Path, progress) -> None:  # noqa: ANN001
    """Real HuggingFace snapshot downloader. Isolated so tests can
    inject a stub without pulling in ``huggingface_hub``.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise RpcError(
            RpcErrorCode.STT_WHISPER_NOT_INSTALLED,
            "huggingface_hub is not installed; add faster-whisper to the worker environment",
        ) from e
    if progress is not None:
        try:
            progress(0.05, "downloading")
        except ProviderCancelled:
            raise
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=["*.json", "*.bin", "*.txt", "vocab*", "tokenizer*"],
    )
    if progress is not None:
        try:
            progress(1.0, "downloaded")
        except ProviderCancelled:
            raise
