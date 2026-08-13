"""llama-cpp-python backed provider.

Nothing in this module is imported at package import time — llama.cpp
is a heavy native dependency and may not be installed. The heavy
imports happen inside :meth:`_ensure_llm` so unit tests that don't
touch the provider don't pay the cost.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .. import logging as log
from ..errors import RpcErrorCode
from .models import TranslateOptions, TranslationChunk, TranslatedSegment
from .prompts import PromptMessage, render_chunk_messages
from .provider import (
    ProviderCancelled,
    ProviderError,
    TranslateContext,
    TranslationProvider,
)
from .registry import resolve_model_path

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Rounds of repair after a chunk comes back with segments missing. The
# final round degrades to one request per segment, so this is also the
# point at which we stop trying and surface the failure.
_MAX_REPAIR_ATTEMPTS = 3


class LlamaCppTranslationProvider(TranslationProvider):
    """Runs local GGUF inference via ``llama_cpp.Llama``.

    The model is loaded lazily and cached across chunks in a single
    translate call: we pay the (potentially multi-second) load penalty
    once per invocation. Between calls the model is released so we
    never keep two large models in memory simultaneously.
    """

    name = "llama.cpp"

    def __init__(self, models_root: Path, *, n_ctx: int = 8192, n_threads: int | None = None) -> None:
        self._models_root = Path(models_root)
        self._n_ctx = int(n_ctx)
        self._n_threads = n_threads
        self._llm: Any = None
        self._loaded_path: Optional[str] = None

    # ------------------------------------------------------ TranslationProvider

    def translate_chunks(
        self,
        chunks: list[TranslationChunk],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        ctx: TranslateContext,
    ) -> dict[int, str]:
        if options.prompt_version not in (
            "translation_prompt_v1",
            "translation_prompt_v2",
        ):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_UNKNOWN_PROMPT,
                f"unknown prompt version: {options.prompt_version}",
            )
        model_path = resolve_model_path(self._models_root, options.model)
        if model_path is None:
            raise ProviderError(
                RpcErrorCode.TRANSLATE_MODEL_NOT_INSTALLED,
                f"translation model {options.model!r} not found; drop the GGUF file into <models>/translation/",
            )

        ctx.on_progress(0.0, "loading_model", f"loading {options.model}")
        llm = self._ensure_llm(model_path)

        if ctx.cancelled():
            raise ProviderCancelled()

        total_chunks = max(1, len(chunks))
        translations: dict[int, str] = {}
        for i, chunk in enumerate(chunks):
            if ctx.cancelled():
                raise ProviderCancelled()

            label = f"chunk {i + 1}/{total_chunks}"
            ctx.on_progress(
                i / total_chunks,
                "translating",
                f"{label} ({len(chunk.segment_ids)} segments)",
            )

            # Repairing a chunk can take longer than translating it did
            # — the last resort is one request per segment. Give the
            # repair its own slice of this chunk's share of the bar so
            # the UI keeps moving instead of sitting frozen.
            base = i / total_chunks
            span = 1.0 / total_chunks

            def report(within: float, message: str, _b=base, _s=span) -> None:
                ctx.on_progress(_b + _s * within, "translating", message)

            chunk_map = self._translate_one_chunk(
                llm=llm,
                chunk=chunk,
                segments_by_id=segments_by_id,
                options=options,
                ctx=ctx,
                label=label,
                report=report,
            )
            translations.update(chunk_map)
            ctx.on_chunk_completed(chunk.chunk_index, chunk_map)

        ctx.on_progress(1.0, "finalizing", None)
        return translations

    # ---------------------------------------------------------------- internals

    def _ensure_llm(self, model_path: Path) -> Any:
        if self._llm is not None and self._loaded_path == str(model_path):
            return self._llm
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as e:
            raise ProviderError(
                RpcErrorCode.TRANSLATE_LLAMA_NOT_INSTALLED,
                "llama-cpp-python is not installed in the worker environment",
                recoverable=True,
            ) from e

        # Release any previously loaded model before loading a new one so
        # we never hold two big models in RAM at once.
        self._release()
        # Phase 11 — pick up host-supplied perf knobs so the user can
        # tune threads / Metal without a restart.
        cpu_override: Optional[int] = None
        gpu_layers: Optional[int] = None
        try:
            from .. import handlers as _root_handlers  # type: ignore[import-not-found]

            perf = _root_handlers.get_perf()
            if isinstance(perf, dict):
                v = perf.get("cpu_threads")
                if isinstance(v, int) and v > 0:
                    cpu_override = v
                # llama.cpp's `n_gpu_layers` gates Metal / CUDA. -1 =
                # "offload everything the engine can". Users on
                # low-VRAM setups can turn the flag off via settings.
                gpu_layers = -1 if bool(perf.get("gpu_acceleration", True)) else 0
        except Exception:  # pragma: no cover - keep provider robust
            pass
        try:
            kwargs: dict[str, Any] = {
                "model_path": str(model_path),
                "n_ctx": self._n_ctx,
                "verbose": False,
            }
            resolved_threads = cpu_override if cpu_override is not None else self._n_threads
            if resolved_threads is not None:
                kwargs["n_threads"] = int(resolved_threads)
            if gpu_layers is not None:
                kwargs["n_gpu_layers"] = gpu_layers
            llm = Llama(**kwargs)
        except MemoryError as e:  # pragma: no cover
            raise ProviderError(
                RpcErrorCode.TRANSLATE_OUT_OF_MEMORY,
                f"not enough memory to load {model_path.name}",
                recoverable=True,
            ) from e
        except Exception as e:
            log.warn("llama model load failed", model=str(model_path), error=str(e))
            raise ProviderError(
                RpcErrorCode.TRANSLATE_MODEL_LOAD_FAILED,
                f"failed to load GGUF model {model_path.name}: {e}",
            ) from e
        self._llm = llm
        self._loaded_path = str(model_path)
        return llm

    def _release(self) -> None:
        # llama.cpp holds native resources; explicit close is a best
        # effort — GC will eventually reclaim it either way.
        try:
            if hasattr(self._llm, "close"):
                self._llm.close()
        except Exception:  # pragma: no cover
            pass
        self._llm = None
        self._loaded_path = None

    # ---------------------------------------------------- Phase 11 unload

    def unload(self) -> bool:
        """Release the resident GGUF context so its RAM (typically
        4–10 GB for the recommended quantised models) returns to the
        OS. Called by the Rust "unload" surface and by the per-stage
        auto-unloader. Returns ``True`` iff a model was actually held.
        """
        if self._llm is None:
            return False
        self._release()
        return True

    def _complete(
        self,
        *,
        llm: Any,
        messages: list[PromptMessage],
        options: TranslateOptions,
        temperature: float,
    ) -> str:
        try:
            response = llm.create_chat_completion(
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                top_p=options.top_p,
                max_tokens=options.max_tokens,
                response_format={"type": "json_object"},
            )
        except MemoryError as e:  # pragma: no cover
            raise ProviderError(
                RpcErrorCode.TRANSLATE_OUT_OF_MEMORY,
                "translation ran out of memory",
                recoverable=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                RpcErrorCode.TRANSLATE_LLM_FAILURE,
                f"llama.cpp inference failed: {e}",
            ) from e
        return _extract_content(response)

    def _attempt_ids(
        self,
        *,
        llm: Any,
        chunk: TranslationChunk,
        segment_ids: list[int],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        temperature: float,
    ) -> dict[int, str]:
        """Ask for `segment_ids` only, keeping the chunk's context window.

        Returns whatever parsed cleanly; callers decide what to do about
        ids the model skipped. Malformed JSON yields nothing rather than
        raising, so a retry at a different temperature still gets a shot.
        """
        sub = TranslationChunk(
            chunk_index=chunk.chunk_index,
            segment_ids=list(segment_ids),
            context_before_ids=chunk.context_before_ids,
            context_after_ids=chunk.context_after_ids,
        )
        messages = self._build_messages(
            chunk=sub,
            segments_by_id=segments_by_id,
            options=options,
        )
        raw = self._complete(
            llm=llm,
            messages=messages,
            options=options,
            temperature=temperature,
        )
        try:
            return self._parse_response(raw, expected_ids=segment_ids, strict=False)
        except ProviderError:
            # Unparseable output — treat as "nothing came back" so the
            # escalation below can retry instead of killing the job.
            return {}

    def _translate_one_chunk(
        self,
        *,
        llm: Any,
        chunk: TranslationChunk,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        ctx: TranslateContext,
        label: str = "chunk",
        report: Optional[Callable[[float, str], None]] = None,
    ) -> dict[int, str]:
        """Translate a chunk, recovering from partial LLM responses.

        Quantised models occasionally drop a line or two out of a
        20-segment batch. Failing the whole job over that used to throw
        away every chunk translated so far, so instead we escalate:
        re-ask for just the stragglers (a smaller batch is markedly more
        reliable), then one-by-one, nudging temperature up each round to
        break the model out of whatever groove produced the bad output.
        Only a segment that survives all of that is fatal.
        """
        translations = self._attempt_ids(
            llm=llm,
            chunk=chunk,
            segment_ids=chunk.segment_ids,
            segments_by_id=segments_by_id,
            options=options,
            temperature=options.temperature,
        )
        missing = [i for i in chunk.segment_ids if i not in translations]
        if not missing:
            return translations

        for attempt in range(_MAX_REPAIR_ATTEMPTS):
            if ctx.cancelled():
                raise ProviderCancelled()
            # Nudge off the deterministic path that just failed; cap so
            # we never wander into incoherent territory.
            temperature = min(1.0, options.temperature + 0.1 * (attempt + 1))
            log.warn(
                "translation chunk incomplete; retrying missing segments",
                chunk=chunk.chunk_index,
                missing=len(missing),
                attempt=attempt + 1,
            )
            # Last resort: one request per segment. Slow, but a single
            # line is about as easy as the task gets.
            per_segment = attempt == _MAX_REPAIR_ATTEMPTS - 1
            batches = [[i] for i in missing] if per_segment else [missing]
            # Repair occupies the back half of this chunk's slice, so
            # the bar advances even when every segment needs its own
            # request. Each attempt covers a third of that half.
            attempt_base = 0.5 + 0.5 * (attempt / _MAX_REPAIR_ATTEMPTS)
            attempt_span = 0.5 / _MAX_REPAIR_ATTEMPTS

            for k, batch in enumerate(batches):
                if ctx.cancelled():
                    raise ProviderCancelled()
                if report is not None:
                    within = attempt_base + attempt_span * (k / max(1, len(batches)))
                    detail = (
                        f"segment {batch[0]} ({k + 1}/{len(batches)})"
                        if per_segment
                        else f"{len(batch)} missing segments"
                    )
                    report(
                        min(0.99, within),
                        f"{label}: retrying {detail} "
                        f"(attempt {attempt + 1}/{_MAX_REPAIR_ATTEMPTS})",
                    )
                translations.update(
                    self._attempt_ids(
                        llm=llm,
                        chunk=chunk,
                        segment_ids=batch,
                        segments_by_id=segments_by_id,
                        options=options,
                        temperature=temperature,
                    )
                )
            missing = [i for i in chunk.segment_ids if i not in translations]
            if not missing:
                return translations

        raise ProviderError(
            RpcErrorCode.TRANSLATE_INCOMPLETE_RESPONSE,
            f"LLM kept omitting segment ids after {_MAX_REPAIR_ATTEMPTS} retries: "
            f"{missing[:10]}{'…' if len(missing) > 10 else ''}",
        )

    def _build_messages(
        self,
        *,
        chunk: TranslationChunk,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
    ) -> list[PromptMessage]:
        def _row(seg_id: int) -> dict:
            seg = segments_by_id.get(seg_id)
            if seg is None:
                return {"id": seg_id, "text": ""}
            return {"id": seg_id, "text": seg.source_text}

        return render_chunk_messages(
            prompt_version=options.prompt_version,
            source_lang=options.source_language,
            target_lang=options.target_language,
            context_before=[_row(i) for i in chunk.context_before_ids],
            chunk=[_row(i) for i in chunk.segment_ids],
            context_after=[_row(i) for i in chunk.context_after_ids],
        )

    @staticmethod
    def _parse_response(
        text: str, *, expected_ids: list[int], strict: bool = True
    ) -> dict[int, str]:
        """Pull `{id: translation}` out of the model's JSON.

        With ``strict=False`` a response that omits some requested ids is
        returned as-is instead of raising, letting the caller retry just
        the stragglers.
        """
        payload = _coerce_json(text)
        if not isinstance(payload, dict):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM response is not a JSON object",
            )
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM response is missing a `segments` array",
            )
        expected = set(expected_ids)
        translations: dict[int, str] = {}
        for entry in segments:
            if not isinstance(entry, dict):
                continue
            try:
                sid = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if sid not in expected:
                # Model returned a segment we didn't ask for — silently drop
                # it. We refuse only if the requested ids are missing.
                continue
            translation = entry.get("translation")
            if translation is None:
                continue
            translations[sid] = str(translation).strip()
        missing = [i for i in expected_ids if i not in translations]
        if missing and strict:
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INCOMPLETE_RESPONSE,
                f"LLM response missed segment ids: {missing[:10]}{'…' if len(missing) > 10 else ''}",
            )
        return translations


def _extract_content(response: Any) -> str:
    """Pull the message content out of a llama.cpp chat completion.

    Kept isolated so tests can pass a plain dict without importing
    ``llama_cpp``.
    """
    try:
        choices = response["choices"]
        if not choices:
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM returned no choices",
            )
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(msg, dict):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM response has no message",
            )
        content = msg.get("content")
        if not isinstance(content, str):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM response content is not a string",
            )
        return content
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(
            RpcErrorCode.TRANSLATE_INVALID_JSON,
            f"unexpected LLM response shape: {e}",
        ) from e


def _coerce_json(text: str) -> Any:
    """Best-effort recovery of a JSON object from the model's reply.

    Models sometimes wrap JSON in prose or a code fence despite the
    system prompt asking them not to. We first try direct parse; if
    that fails, we look for the first ``{...}`` span.
    """
    text = text.strip()
    if not text:
        raise ProviderError(
            RpcErrorCode.TRANSLATE_INCOMPLETE_RESPONSE,
            "LLM returned an empty response",
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ProviderError(
            RpcErrorCode.TRANSLATE_INVALID_JSON,
            "LLM response is not valid JSON and has no {...} span",
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ProviderError(
            RpcErrorCode.TRANSLATE_INVALID_JSON,
            f"LLM response could not be parsed as JSON: {e}",
        ) from e
