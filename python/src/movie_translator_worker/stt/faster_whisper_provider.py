"""faster-whisper backed provider.

Nothing in this module is imported at package import time — faster
whisper is expensive to load and may not be installed in every
environment. The heavy imports happen inside :meth:`load` so unit
tests that don't touch the provider don't pay the cost.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcErrorCode
from .device import default_device
from .models import Segment, TranscribeOptions, Word
from .provider import (
    ProviderCancelled,
    ProviderError,
    SpeechToTextProvider,
    TranscribeContext,
)
from .registry import is_installed, model_dir


class FasterWhisperProvider(SpeechToTextProvider):
    """Runs Whisper inference via ``faster_whisper.WhisperModel``.

    The model is loaded lazily and cached across calls: a project may
    transcribe many audio files with the same model config and we want
    to pay the ~2s load penalty only once.
    """

    name = "faster-whisper"

    def __init__(self, models_root: Path) -> None:
        self._models_root = Path(models_root)
        self._loaded_model: Any = None
        self._loaded_key: Optional[tuple[str, str, str]] = None

    # ---------------------------------------------------- Phase 11 unload

    def unload(self) -> bool:
        """Drop the resident WhisperModel so its RAM (multi-GB for
        large-v3) is returned to the OS between long idle periods.
        Returns ``True`` if a model was actually released.

        The next transcribe call will re-instantiate on demand.
        """
        if self._loaded_model is None:
            return False
        # faster-whisper's WhisperModel does not expose an explicit
        # close on every version; GC + reference drop is enough to
        # release CTranslate2's native tensors.
        self._loaded_model = None
        self._loaded_key = None
        gc.collect()
        return True

    # ---------------------------------------------------------- SpeechToTextProvider

    def transcribe(
        self,
        audio_path: str,
        options: TranscribeOptions,
        ctx: TranscribeContext,
    ) -> tuple[str, list[Segment]]:
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            raise ProviderError(
                RpcErrorCode.STT_INVALID_AUDIO,
                f"audio file not found: {audio_path}",
            )
        if not is_installed(self._models_root, options.model):
            raise ProviderError(
                RpcErrorCode.STT_MODEL_NOT_INSTALLED,
                f"whisper model {options.model!r} is not installed; download it first",
            )

        ctx.on_progress(0.0, "loading_model", f"loading {options.model}")
        model = self._ensure_model(options)

        if ctx.cancelled():
            raise ProviderCancelled()

        ctx.on_progress(0.05, "transcribing", None)
        segments_iter, info = self._run_transcribe(model, str(audio_file), options, ctx)

        detected_language = getattr(info, "language", None) or options.normalised_language()
        total = float(getattr(info, "duration", 0.0)) or 0.0

        out: list[Segment] = []
        last_progress = -1.0
        for i, raw in enumerate(segments_iter):
            if ctx.cancelled():
                raise ProviderCancelled()
            seg = self._map_segment(i, raw, want_words=options.word_timestamps)
            out.append(seg)
            if total > 0.0:
                fraction = min(0.99, 0.05 + 0.9 * (seg.end / total))
            else:
                # Fall back to a modest tick when duration is unknown.
                fraction = min(0.99, 0.05 + 0.01 * len(out))
            if fraction - last_progress > 0.01:
                ctx.on_progress(fraction, "transcribing", seg.text[:80] or None)
                last_progress = fraction

        ctx.on_progress(0.99, "finalizing", None)
        return detected_language, out

    # ---------------------------------------------------------------- internals

    def _ensure_model(self, options: TranscribeOptions) -> Any:
        key = (options.model, options.device, options.compute_type)
        if self._loaded_model is not None and self._loaded_key == key:
            return self._loaded_model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ProviderError(
                RpcErrorCode.STT_WHISPER_NOT_INSTALLED,
                "faster-whisper is not installed in the worker environment",
                recoverable=True,
            ) from e

        device = options.device or default_device()
        compute_type = options.compute_type or _default_compute(device)
        path = model_dir(self._models_root, options.model)
        # Phase 11 — respect the user's advanced perf knobs. The
        # `perf` snapshot lives on `handlers._PERF`; we tolerate the
        # import failing (unit tests that stub the worker) by
        # defaulting to "let the engine pick".
        cpu_threads: Optional[int] = None
        try:
            from .. import handlers as _root_handlers  # type: ignore[import-not-found]

            perf = _root_handlers.get_perf()
            if isinstance(perf, dict):
                v = perf.get("cpu_threads")
                if isinstance(v, int) and v > 0:
                    cpu_threads = v
                gpu = perf.get("gpu_acceleration")
                if gpu is False and device != "cpu":
                    # User asked for CPU-only; downgrade automatic
                    # device selection but leave explicit overrides.
                    if options.device is None:
                        device = "cpu"
                        compute_type = _default_compute("cpu")
        except Exception:  # pragma: no cover - keep provider robust
            pass
        try:
            kwargs = {
                "device": device,
                "compute_type": compute_type,
                "local_files_only": True,
            }
            if cpu_threads is not None:
                kwargs["cpu_threads"] = cpu_threads
            model = WhisperModel(str(path), **kwargs)
        except MemoryError as e:  # pragma: no cover - depends on host
            raise ProviderError(
                RpcErrorCode.STT_OUT_OF_MEMORY,
                f"not enough memory to load {options.model!r} with compute_type={compute_type}",
                recoverable=True,
            ) from e
        except Exception as e:
            log.warn(
                "whisper model load failed",
                model=options.model,
                device=device,
                compute_type=compute_type,
                error=str(e),
                error_type=type(e).__name__,
                tb=traceback.format_exc(limit=25),
            )
            raise ProviderError(
                RpcErrorCode.STT_MODEL_LOAD_FAILED,
                f"failed to load whisper model {options.model!r}: {e}",
            ) from e
        self._loaded_model = model
        self._loaded_key = key
        return model

    def _run_transcribe(
        self,
        model: Any,
        audio_path: str,
        options: TranscribeOptions,
        ctx: TranscribeContext,
    ):
        kwargs: dict[str, Any] = {
            "beam_size": options.beam_size,
            "temperature": options.temperature,
            "word_timestamps": options.word_timestamps,
            "vad_filter": options.vad_filter,
            "initial_prompt": options.initial_prompt,
            # Movie audio has overlapping speakers and music; carrying
            # the previous window as a prompt causes Whisper to invent
            # repeated lines. Keep each window independent.
            "condition_on_previous_text": False,
        }
        if options.vad_filter:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 450,
                "speech_pad_ms": 220,
            }
        if options.language:
            kwargs["language"] = options.language
            kwargs["task"] = "transcribe"
        try:
            return model.transcribe(audio_path, **kwargs)
        except MemoryError as e:  # pragma: no cover
            raise ProviderError(
                RpcErrorCode.STT_OUT_OF_MEMORY, "whisper ran out of memory", recoverable=True,
            ) from e
        except Exception as e:
            if ctx.cancelled():
                raise ProviderCancelled() from e
            raise ProviderError(
                RpcErrorCode.STT_INVALID_AUDIO,
                f"whisper could not read the audio file: {e}",
            ) from e

    @staticmethod
    def _map_segment(index: int, raw: Any, *, want_words: bool) -> Segment:
        words: Optional[list[Word]] = None
        if want_words:
            raw_words = getattr(raw, "words", None) or []
            words = [
                Word(
                    word=str(getattr(w, "word", "")),
                    start=float(getattr(w, "start", 0.0)),
                    end=float(getattr(w, "end", 0.0)),
                    probability=_maybe_float(getattr(w, "probability", None)),
                )
                for w in raw_words
            ]
        return Segment(
            id=int(getattr(raw, "id", index)),
            start=float(getattr(raw, "start", 0.0)),
            end=float(getattr(raw, "end", 0.0)),
            text=str(getattr(raw, "text", "")).strip(),
            avg_logprob=_maybe_float(getattr(raw, "avg_logprob", None)),
            no_speech_prob=_maybe_float(getattr(raw, "no_speech_prob", None)),
            words=words,
        )


def _default_compute(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
