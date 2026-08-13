"""RPC handlers for the translation subsystem.

Registered by :func:`install` from :mod:`..handlers`.

Methods:

  * ``translate.env`` — whether ``llama-cpp-python`` is importable and
    where the local model directory lives (sync).
  * ``translate.list_models`` — enumerate installed GGUF files (sync).
  * ``translate.list_prompt_versions`` — known prompt versions (sync).
  * ``translate.translate`` — run the LLM (**async**, cancellable,
    emits ``translate.progress`` and ``translate.chunk_completed``
    notifications). Returns metadata; the actual translations are
    streamed via chunk_completed events so the host can persist
    incrementally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcError, RpcErrorCode
from ..rpc import HandlerContext
from . import prompts, registry
from .llama_cpp_provider import LlamaCppTranslationProvider
from .models import (
    TranslateOptions,
    TranslatedSegment,
    build_cache_key,
    chunk_segments,
    missing_ids,
)
from .provider import (
    ProviderCancelled,
    ProviderError,
    TranslateContext,
    TranslationProvider,
)

_MODELS_ROOT: Optional[Path] = None
_PROVIDER: Optional[TranslationProvider] = None


def configure(
    *,
    models_root: Path,
    provider: Optional[TranslationProvider] = None,
) -> None:
    """Wire up per-process state.

    ``provider`` is optional: tests inject a fake, production leaves it
    ``None`` and the real :class:`LlamaCppTranslationProvider` is created
    lazily on first use.
    """
    global _MODELS_ROOT, _PROVIDER
    _MODELS_ROOT = Path(models_root)
    _PROVIDER = provider
    registry.ensure_models_dir(_MODELS_ROOT)


def _models_root() -> Path:
    if _MODELS_ROOT is None:
        raise RpcError(RpcErrorCode.INTERNAL, "translation handlers not configured")
    return _MODELS_ROOT


def _provider() -> TranslationProvider:
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    _PROVIDER = LlamaCppTranslationProvider(_models_root())
    return _PROVIDER


# --------------------------------------------------------------------- sync


def translate_env(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "llamaInstalled": registry.llama_cpp_installed(),
        "modelsRoot": str(_models_root()),
        "translationRoot": str(registry.models_dir(_models_root())),
        "defaultModel": registry.default_model(_models_root()),
        "promptVersions": prompts.prompt_versions(),
    }


def translate_list_models(_params: dict[str, Any]) -> dict[str, Any]:
    models = registry.list_models(_models_root())
    return {"models": [m.to_dict() for m in models]}


def translate_list_prompt_versions(_params: dict[str, Any]) -> dict[str, Any]:
    return {"versions": prompts.prompt_versions()}


def translate_list_recommended(_params: dict[str, Any]) -> dict[str, Any]:
    """Phase 12 — what the app can auto-download for the user when
    they don't have a translation model yet. Wire shape mirrors
    ``stt.list_models`` — see :func:`registry.recommended_presets`.
    """
    return {"presets": registry.recommended_presets()}


# ------------------------------------------------------------------- async


def translate_download_model(params: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
    """Phase 12 — pull a curated GGUF from HuggingFace into the
    translation models directory.

    Parity with ``stt.download_model``: emits ``translate.download_progress``
    notifications keyed on ``requestId`` so the Rust host can drive a
    progress bar. Refuses unknown presets and cleans up partial
    downloads on cancellation/failure so the next attempt starts
    fresh instead of picking up a broken half-file.
    """
    preset = _require_str(params, "preset")
    meta = registry.recommended_meta(preset)
    if meta is None:
        raise RpcError(
            RpcErrorCode.TRANSLATE_UNKNOWN_PRESET,
            f"unknown translation preset: {preset}",
        )
    repo = str(meta["repo"])
    filename = str(meta["filename"])
    root = _models_root()
    registry.ensure_models_dir(root)

    dest = registry.models_dir(root) / filename
    if dest.is_file() and dest.stat().st_size > 0:
        # Idempotent — already installed. Emit a completed progress
        # tick so the UI closes its download panel cleanly.
        ctx.emit_progress(
            "translate.download_progress",
            {"stage": "already_installed", "fraction": 1.0},
        )
        return {
            "ok": True,
            "preset": preset,
            "name": filename,
            "path": str(dest),
            "sizeBytes": dest.stat().st_size,
            "alreadyInstalled": True,
        }

    def _progress(fraction: float, stage: str) -> None:
        ctx.emit_progress(
            "translate.download_progress",
            {"fraction": max(0.0, min(1.0, float(fraction))), "stage": stage},
        )

    _progress(0.0, "starting")

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise RpcError(
            RpcErrorCode.TRANSLATE_LLAMA_NOT_INSTALLED,
            "huggingface_hub is not installed; add the [translation] extra to the worker environment",
        ) from e

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        # `hf_hub_download` downloads into its own cache and returns
        # a path; we then move it into place atomically so a crashed
        # download can't leave a corrupt `.gguf` that would confuse
        # `list_models`. We use a `.part` sidecar to keep the
        # intermediate file visible but out of the loader's scan.
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user")
        _progress(0.05, "downloading")
        cached_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(registry.models_dir(root)),
        )
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "download cancelled by user")
        # `hf_hub_download` with `local_dir` writes directly to that
        # directory. If it's already at `dest`, we're done; otherwise
        # move into place.
        src = Path(cached_path)
        if src != dest:
            if dest.exists():
                dest.unlink()
            src.rename(dest)
        _progress(1.0, "downloaded")
    except RpcError:
        _cleanup(dest, tmp_dest)
        raise
    except Exception as e:
        _cleanup(dest, tmp_dest)
        raise RpcError(
            RpcErrorCode.TRANSLATE_DOWNLOAD_FAILED,
            f"failed to download {filename!r} from {repo!r}: {e}",
        ) from e

    size = dest.stat().st_size if dest.is_file() else 0
    return {
        "ok": True,
        "preset": preset,
        "name": filename,
        "path": str(dest),
        "sizeBytes": size,
        "alreadyInstalled": False,
    }


def _cleanup(dest: Path, tmp: Path) -> None:
    for p in (tmp, dest):
        try:
            if p.exists() and p.stat().st_size == 0:
                p.unlink()
        except OSError:
            pass


def translate_translate(params: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
    options = _options_from_params(params)
    transcript_cache_key = _require_str(params, "transcriptCacheKey")
    audio_hash = _require_str(params, "audioHash")

    if not prompts.is_known_version(options.prompt_version):
        raise RpcError(
            RpcErrorCode.TRANSLATE_UNKNOWN_PROMPT,
            f"unknown prompt version: {options.prompt_version}",
        )
    if not registry.is_installed(_models_root(), options.model):
        raise RpcError(
            RpcErrorCode.TRANSLATE_MODEL_NOT_INSTALLED,
            f"translation model {options.model!r} is not installed",
            data={"model": options.model},
        )

    raw_segments = params.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "segments must be a non-empty list")
    segments_by_id, ordered_ids = _load_segments(raw_segments)

    existing_map = _existing_translations(params.get("existingTranslations"))
    # Seed the base list with any existing translations so we can compute
    # what's still missing.
    seeded_segments = _apply_existing(segments_by_id, ordered_ids, existing_map)
    todo_ids = missing_ids(seeded_segments)

    ctx.emit_progress(
        "translate.progress",
        {
            "stage": "planning",
            "fraction": 0.0,
            "completedSegments": len(ordered_ids) - len(todo_ids),
            "totalSegments": len(ordered_ids),
        },
    )

    cache_key = build_cache_key(
        transcript_cache_key=transcript_cache_key,
        audio_hash=audio_hash,
        options=options,
    )

    if not todo_ids:
        # Everything already translated — nothing for the LLM to do.
        ctx.emit_progress(
            "translate.progress",
            {
                "stage": "completed",
                "fraction": 1.0,
                "completedSegments": len(ordered_ids),
                "totalSegments": len(ordered_ids),
            },
        )
        return {
            "ok": True,
            "cacheKey": cache_key,
            "totalSegments": len(ordered_ids),
            "translatedSegments": 0,
            "chunks": 0,
            "model": options.model,
            "sourceLanguage": options.source_language,
            "targetLanguage": options.target_language,
            "promptVersion": options.prompt_version,
        }

    # Chunk *only* the ids that still need work — completed segments
    # don't need re-translation and shouldn't waste context.
    chunks = chunk_segments(
        todo_ids,
        chunk_size=options.chunk_size,
        context_before=options.context_before,
        context_after=options.context_after,
    )

    total_todo = len(todo_ids)
    completed = 0
    completed_baseline = len(ordered_ids) - total_todo

    provider = _provider()

    class _Ctx(TranslateContext):
        def cancelled(self) -> bool:
            return ctx.cancelled()

        def on_progress(self, fraction: float, stage: str, message: Optional[str] = None) -> None:
            ctx.emit_progress(
                "translate.progress",
                {
                    "stage": stage,
                    "fraction": max(0.0, min(1.0, float(fraction))),
                    "message": message,
                    "completedSegments": completed_baseline + completed,
                    "totalSegments": len(ordered_ids),
                },
            )

        def on_chunk_completed(self, chunk_index: int, translations: dict[int, str]) -> None:
            nonlocal completed
            completed += len(translations)
            ctx.emit_progress(
                "translate.chunk_completed",
                {
                    "chunkIndex": chunk_index,
                    "translations": [
                        {"id": sid, "translation": text}
                        for sid, text in translations.items()
                    ],
                    "completedSegments": completed_baseline + completed,
                    "totalSegments": len(ordered_ids),
                    "fraction": (completed_baseline + completed) / max(1, len(ordered_ids)),
                },
            )

    try:
        translations = provider.translate_chunks(
            chunks=chunks,
            segments_by_id={s.id: s for s in seeded_segments},
            options=options,
            ctx=_Ctx(),
        )
    except ProviderCancelled as e:
        raise RpcError(RpcErrorCode.CANCELLED, "translation cancelled by user") from e
    except ProviderError as e:
        raise RpcError(e.code, e.message, data={"recoverable": e.recoverable}) from e

    if ctx.cancelled():
        raise RpcError(RpcErrorCode.CANCELLED, "translation cancelled by user")

    return {
        "ok": True,
        "cacheKey": cache_key,
        "totalSegments": len(ordered_ids),
        "translatedSegments": len(translations),
        "chunks": len(chunks),
        "model": options.model,
        "sourceLanguage": options.source_language,
        "targetLanguage": options.target_language,
        "promptVersion": options.prompt_version,
    }


def translate_unload(_params: dict[str, Any]) -> dict[str, Any]:
    """Phase 11 — release the resident GGUF context. Multi-GB of
    resident RAM returns to the OS. Safe no-op if nothing was loaded.
    """
    provider = _PROVIDER
    released = False
    if provider is not None and hasattr(provider, "unload"):
        try:
            released = bool(provider.unload())
        except Exception as e:  # pragma: no cover - unload must not throw
            log.warn("translate provider unload failed", error=str(e))
    return {"ok": True, "released": released}


# ----------------------------------------------------------------- registration


def install(dispatcher) -> None:
    dispatcher.register("translate.env", translate_env)
    dispatcher.register("translate.list_models", translate_list_models)
    dispatcher.register("translate.list_prompt_versions", translate_list_prompt_versions)
    dispatcher.register("translate.list_recommended", translate_list_recommended)
    dispatcher.register("translate.unload", translate_unload)
    dispatcher.register_async("translate.translate", translate_translate)
    dispatcher.register_async("translate.download_model", translate_download_model)


# ------------------------------------------------------------------ helpers


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"missing required string param: {key}")
    return value


def _options_from_params(params: dict[str, Any]) -> TranslateOptions:
    opts = params.get("options") or {}
    if not isinstance(opts, dict):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "options must be an object")
    model = opts.get("model")
    if not isinstance(model, str) or not model:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "options.model is required")
    try:
        return TranslateOptions(
            model=model,
            source_language=str(opts.get("sourceLanguage") or "en"),
            target_language=str(opts.get("targetLanguage") or "vi"),
            prompt_version=str(opts.get("promptVersion") or "translation_prompt_v2"),
            chunk_size=int(opts.get("chunkSize", 10)),
            context_before=int(opts.get("contextBefore", 2)),
            context_after=int(opts.get("contextAfter", 2)),
            temperature=float(opts.get("temperature", 0.2)),
            top_p=float(opts.get("topP", 0.95)),
            max_tokens=int(opts.get("maxTokens", 2048)),
        ).normalised()
    except (TypeError, ValueError) as e:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"invalid translate options: {e}") from e


def _load_segments(
    raw: list[Any],
) -> tuple[dict[int, TranslatedSegment], list[int]]:
    by_id: dict[int, TranslatedSegment] = {}
    order: list[int] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RpcError(RpcErrorCode.INVALID_PARAMS, "each segment must be an object")
        try:
            sid = int(item["id"])
            text = str(item.get("text", ""))
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise RpcError(RpcErrorCode.INVALID_PARAMS, f"invalid segment: {e}") from e
        if sid in by_id:
            raise RpcError(RpcErrorCode.INVALID_PARAMS, f"duplicate segment id: {sid}")
        by_id[sid] = TranslatedSegment(
            id=sid,
            source_text=text,
            translation="",
            start=start,
            end=end,
        )
        order.append(sid)
    return by_id, order


def _existing_translations(payload: Any) -> dict[int, tuple[str, bool]]:
    """Parse the caller-supplied ``existingTranslations`` map.

    Value shape: ``[{ "id": int, "translation": str, "edited": bool? }]``.
    """
    if payload is None:
        return {}
    if not isinstance(payload, list):
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            "existingTranslations must be a list",
        )
    out: dict[int, tuple[str, bool]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        text = item.get("translation")
        if not isinstance(text, str):
            continue
        edited = bool(item.get("edited", False))
        out[sid] = (text, edited)
    return out


def _apply_existing(
    segments_by_id: dict[int, TranslatedSegment],
    order: list[int],
    existing: dict[int, tuple[str, bool]],
) -> list[TranslatedSegment]:
    out: list[TranslatedSegment] = []
    for sid in order:
        base = segments_by_id[sid]
        if sid in existing:
            text, edited = existing[sid]
            out.append(
                TranslatedSegment(
                    id=sid,
                    source_text=base.source_text,
                    translation=text,
                    start=base.start,
                    end=base.end,
                    edited=edited,
                )
            )
        else:
            out.append(base)
    return out
