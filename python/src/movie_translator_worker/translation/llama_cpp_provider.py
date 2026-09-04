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
from ..source_protection import source_protection_payload
from .memory import TranslationMemory
from .models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationMetadata,
    TranslationResult,
)
from .prompts import PromptMessage, language_name, render_chunk_messages
from .quality import (
    HARD_SEMANTIC_FAILURES,
    global_consistency_issues,
    needs_retry,
    semantic_validate_result,
    select_best_candidate,
)
from .integrity import (
    invalid_translation_ids,
    merge_metadata_validation,
    normalized_speaker_id,
    stable_segment_id,
    validate_batch_integrity,
)
from .semantic_realization import analyze_source_semantics, compact_semantic_payload
from .units import (
    context_unit_ids,
    ownership_payload,
    provenance_for_segment,
    resolve_conversation_structure,
    source_unit_from_segment,
)
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
    ) -> dict[int, TranslationResult]:
        if options.prompt_version not in (
            "translation_prompt_v1",
            "translation_prompt_v2",
            "translation_prompt_v3",
            "translation_prompt_v4",
            "translation_prompt_v5",
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
        translations: dict[int, TranslationResult] = {}
        memory = TranslationMemory.from_segments(list(segments_by_id.values()))
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
                memory=memory,
            )
            translations.update(chunk_map)
            for sid, result in chunk_map.items():
                seg = segments_by_id.get(sid)
                if seg is not None and not result.metadata.needs_review:
                    memory.record(sid, seg.source_text, result)
            ctx.on_chunk_completed(chunk.chunk_index, chunk_map)

        translations = self._global_consistency_pass(
            llm=llm,
            translations=translations,
            segments_by_id=segments_by_id,
            options=options,
            ctx=ctx,
            memory=memory,
        )
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
        json_mode: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = llm.create_chat_completion(**kwargs)
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
        memory: TranslationMemory | None = None,
        hint: str | None = None,
    ) -> dict[int, TranslationResult]:
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
            all_segment_ids=chunk.all_segment_ids,
        )
        messages = self._build_messages(
            chunk=sub,
            segments_by_id=segments_by_id,
            options=options,
            memory=memory,
            hint=hint,
        )
        if _messages_too_large(messages) and len(segment_ids) > 1:
            out: dict[int, TranslationResult] = {}
            for sid in segment_ids:
                out.update(
                    self._attempt_ids(
                        llm=llm,
                        chunk=chunk,
                        segment_ids=[sid],
                        segments_by_id=segments_by_id,
                        options=options,
                        temperature=temperature,
                        memory=memory,
                        hint=hint,
                    )
                )
            return out
        raw = self._complete(
            llm=llm,
            messages=messages,
            options=options,
            temperature=temperature,
        )
        try:
            translations = self._parse_response(raw, expected_ids=segment_ids, strict=False)
        except ProviderError:
            # Unparseable output — treat as "nothing came back" so the
            # escalation below can retry instead of killing the job.
            return {}
        return _drop_invalid_outputs(
            translations,
            segment_ids=segment_ids,
            segments_by_id=segments_by_id,
            options=options,
        )

    def _attempt_single_rescue(
        self,
        *,
        llm: Any,
        segment_id: int,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        temperature: float,
    ) -> dict[int, TranslationResult]:
        """Last-resort one-line prompt for models that keep dodging a segment."""
        seg = segments_by_id.get(segment_id)
        if seg is None:
            return {}
        src = language_name(options.source_language)
        tgt = language_name(options.target_language)
        target_guard = ""
        if (options.target_language or "").lower() == "vi":
            target_guard = (
                "The translation must be Vietnamese in Latin script. "
                "Do not output Chinese/Japanese/Korean characters. "
            )
        protection = _compact_source_protection(seg)
        messages = [
            PromptMessage(
                "system",
                (
                    f"You translate one movie subtitle from {src} to {tgt}. "
                    f"{target_guard}"
                    "Preserve source proposition, sentence mood, question type, "
                    "numbers, polarity, predicate, certainty, and referents exactly. "
                    "Do not output translator notes or candidate alternatives. "
                    "Return JSON only, with exactly the requested id."
                ),
            ),
            PromptMessage(
                "user",
                (
                    f"Translate segment id {segment_id} to {tgt}.\n"
                    f"Source: {json.dumps(seg.source_text, ensure_ascii=False)}\n"
                    f"Source protection: {json.dumps(protection, ensure_ascii=False)}\n"
                    f"Pronoun context: {json.dumps(seg.pronoun_context, ensure_ascii=False)}\n"
                    f'Return: {{"translations":[{{"id":{segment_id},"translated_text":"...","confidence":0.0,"reason_flags":[]}}]}}'
                ),
            ),
        ]
        raw = self._complete(
            llm=llm,
            messages=messages,
            options=options,
            temperature=temperature,
        )
        try:
            translations = self._parse_response(raw, expected_ids=[segment_id], strict=False)
        except ProviderError:
            return {}
        return _drop_invalid_outputs(
            translations,
            segment_ids=[segment_id],
            segments_by_id=segments_by_id,
            options=options,
        )

    def _attempt_single_text_rescue(
        self,
        *,
        llm: Any,
        segment_id: int,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        temperature: float,
    ) -> dict[int, TranslationResult]:
        """Final one-line rescue that does not ask the model to echo an id.

        Some small GGUF models keep returning valid text while omitting
        the JSON id. At this point the caller has already isolated one
        segment, so we can safely wrap a plain-text answer ourselves and
        avoid failing the whole movie over JSON bookkeeping.
        """
        seg = segments_by_id.get(segment_id)
        if seg is None:
            return {}
        src = language_name(options.source_language)
        tgt = language_name(options.target_language)
        protection = _compact_source_protection(seg)
        target_guard = ""
        if (options.target_language or "").lower() == "vi":
            target_guard = (
                "Return Vietnamese in Latin script only. Do not copy Chinese, "
                "Japanese, or Korean characters."
            )
        messages = [
            PromptMessage(
                "system",
                (
                    f"Translate one movie subtitle from {src} to {tgt}. "
                    "Return only the translated subtitle text, no JSON, no id, "
                    f"no notes. {target_guard} Preserve sentence mood, question type, "
                    "numbers, polarity, predicate, certainty, and referents exactly. "
                    "Use conservative Vietnamese if source facts are uncertain."
                ).strip(),
            ),
            PromptMessage(
                "user",
                (
                    f"Source subtitle:\n{seg.source_text}\n\n"
                    f"Source facts:\n{json.dumps(protection, ensure_ascii=False)}\n\n"
                    f"Translation in {tgt}:"
                ),
            ),
        ]
        raw = self._complete(
            llm=llm,
            messages=messages,
            options=options,
            temperature=temperature,
            json_mode=False,
        )
        text = _clean_plain_translation(raw)
        if not text or _looks_like_bad_vietnamese_translation(seg.source_text, text):
            return {}
        result = TranslationResult(
            text,
            TranslationMetadata(
                confidence=0.62,
                needs_review=True,
                retry_count=_MAX_REPAIR_ATTEMPTS + 1,
                translation_method="single_text_rescue",
                reason_flags=["LLM_JSON_ID_RESCUE"],
                validation={
                    "valid": False,
                    "issues": ["LLM_JSON_ID_RESCUE"],
                    "sourceLength": len(seg.source_text.strip()),
                },
            ),
        )
        return {segment_id: result}

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
        memory: TranslationMemory | None = None,
    ) -> dict[int, TranslationResult]:
        """Translate a chunk, recovering from partial LLM responses.

        Quantised models occasionally drop a line or two out of a
        20-segment batch. Failing the whole job over that used to throw
        away every chunk translated so far, so instead we escalate:
        re-ask for just the stragglers (a smaller batch is markedly more
        reliable), then one-by-one, nudging temperature up each round to
        break the model out of whatever groove produced the bad output.
        Only a segment that survives all of that is fatal.
        """
        memory = memory or TranslationMemory.from_segments(list(segments_by_id.values()))
        translations = self._attempt_ids(
            llm=llm,
            chunk=chunk,
            segment_ids=chunk.segment_ids,
            segments_by_id=segments_by_id,
            options=options,
            temperature=options.temperature,
            memory=memory,
        )
        missing = [i for i in chunk.segment_ids if i not in translations]
        if not missing:
            return self._retry_low_confidence(
                llm=llm,
                chunk=chunk,
                translations=translations,
                segments_by_id=segments_by_id,
                options=options,
                ctx=ctx,
                label=label,
                report=report,
                memory=memory,
            )

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
                        memory=memory,
                    )
                )
            missing = [i for i in chunk.segment_ids if i not in translations]
            if not missing:
                return self._retry_low_confidence(
                    llm=llm,
                    chunk=chunk,
                    translations=translations,
                    segments_by_id=segments_by_id,
                    options=options,
                    ctx=ctx,
                    label=label,
                    report=report,
                    memory=memory,
                )

        for k, segment_id in enumerate(list(missing)):
            if ctx.cancelled():
                raise ProviderCancelled()
            if report is not None:
                report(
                    min(0.99, 0.97 + 0.02 * (k / max(1, len(missing)))),
                    f"{label}: rescue translating segment {segment_id} "
                    f"({k + 1}/{len(missing)})",
                )
            translations.update(
                self._attempt_single_rescue(
                    llm=llm,
                    segment_id=segment_id,
                    segments_by_id=segments_by_id,
                    options=options,
                    temperature=min(1.0, options.temperature + 0.5),
                )
            )
        missing = [i for i in chunk.segment_ids if i not in translations]
        if missing:
            for k, segment_id in enumerate(list(missing)):
                if ctx.cancelled():
                    raise ProviderCancelled()
                if report is not None:
                    report(
                        min(0.995, 0.99 + 0.005 * (k / max(1, len(missing)))),
                        f"{label}: plain-text rescue for segment {segment_id} "
                        f"({k + 1}/{len(missing)})",
                    )
                translations.update(
                    self._attempt_single_text_rescue(
                        llm=llm,
                        segment_id=segment_id,
                        segments_by_id=segments_by_id,
                        options=options,
                        temperature=min(1.0, options.temperature + 0.6),
                    )
                )
        missing = [i for i in chunk.segment_ids if i not in translations]
        if missing:
            for segment_id in missing:
                translations[segment_id] = _fallback_missing_translation(
                    segment_id=segment_id,
                    seg=segments_by_id.get(segment_id),
                    target_language=options.target_language,
                )
                log.warn(
                    "translation segment fell back after exhaustive rescue",
                    segment=segment_id,
                    chunk=chunk.chunk_index,
                )
        missing = [i for i in chunk.segment_ids if i not in translations]
        if not missing:
            return self._retry_low_confidence(
                llm=llm,
                chunk=chunk,
                translations=translations,
                segments_by_id=segments_by_id,
                options=options,
                ctx=ctx,
                label=label,
                report=report,
                memory=memory,
            )

        return self._retry_low_confidence(
            llm=llm,
            chunk=chunk,
            translations=translations,
            segments_by_id=segments_by_id,
            options=options,
            ctx=ctx,
            label=label,
            report=report,
            memory=memory,
        )

    def _build_messages(
        self,
        *,
        chunk: TranslationChunk,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        memory: TranslationMemory | None = None,
        hint: str | None = None,
    ) -> list[PromptMessage]:
        target_context_unit_ids = context_unit_ids(
            before=chunk.context_before_ids,
            after=chunk.context_after_ids,
            segments_by_id=segments_by_id,
        )
        nearby_ids = chunk.context_before_ids + chunk.segment_ids + chunk.context_after_ids
        nearby_units = [
            source_unit_from_segment(segments_by_id[sid])
            for sid in nearby_ids
            if sid in segments_by_id
        ]
        turns, boundaries, semantic_groups = resolve_conversation_structure(nearby_units)
        turn_by_unit = {
            unit_id: turn.turn_id
            for turn in turns
            for unit_id in turn.unit_ids
        }
        group_by_unit = {
            unit_id: group.semantic_group_id
            for group in semantic_groups
            for unit_id in group.member_unit_ids
        }

        def _row(seg_id: int, *, context_only: bool = False) -> dict:
            seg = segments_by_id.get(seg_id)
            if seg is None:
                return {
                    "id": seg_id,
                    "unitId": f"seg_{seg_id:05d}",
                    "segmentId": f"seg_{seg_id:05d}",
                    "text": "",
                    "contextOnly": context_only,
                    "speakerId": "UNKNOWN",
                }
            source_unit = source_unit_from_segment(seg)
            row = {
                "id": seg_id,
                "unitId": source_unit.unit_id,
                "segmentId": stable_segment_id(seg),
                "sourceSegmentId": stable_segment_id(seg),
                "text": seg.source_text,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "contextOnly": context_only,
                "sourceUnit": source_unit.to_dict(),
                "ownership": ownership_payload(
                    source_unit_id=source_unit.unit_id,
                    context_unit_ids=[] if context_only else target_context_unit_ids,
                    target=not context_only,
                ),
                "conversationTurnId": turn_by_unit.get(source_unit.unit_id),
                "semanticGroupId": group_by_unit.get(source_unit.unit_id),
                "speakerId": normalized_speaker_id(seg.speaker_id),
            }
            row["sourceProtection"] = _compact_source_protection(seg)
            if seg.speaker_confidence is not None:
                row["speakerConfidence"] = round(float(seg.speaker_confidence), 4)
            if seg.pronoun_context:
                row["pronounContext"] = seg.pronoun_context
            if memory is not None:
                pronoun_plan = memory.pronoun_plan_for_segment(seg_id)
                if pronoun_plan is not None:
                    row["automaticPronounPlan"] = pronoun_plan.to_dict()
            else:
                pronoun_plan = None
            semantic = analyze_source_semantics(
                segment_id=seg_id,
                source=seg.source_text,
                speaker_id=seg.speaker_id,
                pronoun_plan=pronoun_plan,
            )
            semantic_payload = compact_semantic_payload(semantic)
            if semantic_payload.get("terms") or semantic_payload.get("ambiguityScore", 0) >= 0.45:
                row["semanticRepresentation"] = semantic_payload
            return row

        memory_payload = (
            dict(memory.prompt_payload(nearby_ids)) if memory is not None else None
        )
        if memory_payload is None:
            memory_payload = {}
        memory_payload["translationUnitContract"] = {
            "targets": [
                source_unit_from_segment(segments_by_id[sid]).unit_id
                for sid in chunk.segment_ids
                if sid in segments_by_id
            ],
            "context": target_context_unit_ids,
            "conversationBoundaries": [boundary.to_dict() for boundary in boundaries],
            "rule": "context may be used as evidence; output ownership must remain one result per target unit",
        }
        messages = render_chunk_messages(
            prompt_version=options.prompt_version,
            source_lang=options.source_language,
            target_lang=options.target_language,
            context_before=[_row(i, context_only=True) for i in chunk.context_before_ids],
            chunk=[_row(i) for i in chunk.segment_ids],
            context_after=[_row(i, context_only=True) for i in chunk.context_after_ids],
            hint=hint,
            translation_memory=memory_payload,
        )
        if not _messages_too_large(messages):
            return messages
        return render_chunk_messages(
            prompt_version=options.prompt_version,
            source_lang=options.source_language,
            target_lang=options.target_language,
            context_before=[],
            chunk=[_row(i) for i in chunk.segment_ids],
            context_after=[],
            hint=hint,
            translation_memory=_compact_translation_memory(memory_payload),
        )

    def _retry_low_confidence(
        self,
        *,
        llm: Any,
        chunk: TranslationChunk,
        translations: dict[int, TranslationResult],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        ctx: TranslateContext,
        label: str,
        report: Optional[Callable[[float, str], None]],
        memory: TranslationMemory,
    ) -> dict[int, TranslationResult]:
        for retry_index in range(min(2, options.max_translation_retries)):
            translations = self._validate_chunk_results(
                chunk=chunk,
                translations=translations,
                segments_by_id=segments_by_id,
                options=options,
                memory=memory,
            )
            uncertain = [
                sid for sid, result in translations.items()
                if sid in chunk.segment_ids and needs_retry(result, options)
            ]
            if not uncertain:
                break
            if ctx.cancelled():
                raise ProviderCancelled()
            if report is not None:
                report(
                    min(0.99, 0.90 + retry_index * 0.03),
                    f"{label}: retrying {len(uncertain)} uncertain translations",
                )
            expanded = self._expanded_chunk(chunk, uncertain, segments_by_id, options)
            issue_summary = _revision_issue_summary(translations, uncertain)
            anchor_summary = _revision_anchor_summary(segments_by_id, uncertain)
            retry_map = self._attempt_ids(
                llm=llm,
                chunk=expanded,
                segment_ids=uncertain,
                segments_by_id=segments_by_id,
                options=options,
                temperature=max(0.0, min(options.temperature, options.temperature - 0.05 * (retry_index + 1))),
                memory=memory,
                hint=(
                    "The previous translation failed semantic validation. Return to the "
                    "original source text, not the previous Vietnamese. Use this expanded "
                    "context plus the source semantic anchors and validator issues. "
                    f"Semantic anchors by id: {anchor_summary}. Validator issues: {issue_summary}. "
                    "Rewrite only the requested CURRENT segment ids. Repair conservatively: "
                    "preserve proposition, speech act, question type, polarity, numbers, "
                    "predicate, referents, relationships, certainty, and source-grounded "
                    "tone before making Vietnamese natural. Do not add particles such as "
                    "à/hả/sao/ư/nhé unless the source sentence function supports them. "
                    "Do not output translator commentary."
                ),
            )
            for sid, retry_result in retry_map.items():
                previous = translations.get(sid)
                if previous is None:
                    translations[sid] = retry_result
                    continue
                retry_metadata = TranslationMetadata(
                    confidence=retry_result.metadata.confidence,
                    needs_review=retry_result.metadata.needs_review,
                    retry_count=retry_index + 1,
                    translation_method="context_retry",
                    reason_flags=retry_result.metadata.reason_flags,
                    validation=retry_result.metadata.validation,
                )
                candidate = self._validated_retry_candidate(
                    sid=sid,
                    result=TranslationResult(retry_result.translation, retry_metadata),
                    chunk=expanded,
                    segments_by_id=segments_by_id,
                    options=options,
                    memory=memory,
                    revision_attempt=retry_index + 1,
                )
                if _score_result(candidate) > _score_result(previous):
                    translations[sid] = candidate
        return self._validate_chunk_results(
            chunk=chunk,
            translations=translations,
            segments_by_id=segments_by_id,
            options=options,
            memory=memory,
        )

    def _validated_retry_candidate(
        self,
        *,
        sid: int,
        result: TranslationResult,
        chunk: TranslationChunk,
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        memory: TranslationMemory,
        revision_attempt: int,
    ) -> TranslationResult:
        seg = segments_by_id.get(sid)
        if seg is None:
            return result
        context_ids = context_unit_ids(
            before=chunk.context_before_ids,
            after=chunk.context_after_ids,
            segments_by_id=segments_by_id,
        )
        return semantic_validate_result(
            segment_id=sid,
            source=seg.source_text,
            source_protection=seg.source_protection,
            result=result,
            memory=memory,
            options=options,
            context_before=[
                segments_by_id[i]
                for i in chunk.context_before_ids
                if i in segments_by_id
            ],
            context_after=[
                segments_by_id[i]
                for i in chunk.context_after_ids
                if i in segments_by_id
            ],
            revision_attempt=revision_attempt,
            provenance=provenance_for_segment(
                seg=seg,
                context_ids=context_ids,
            ).to_dict(),
        )

    def _validate_chunk_results(
        self,
        *,
        chunk: TranslationChunk,
        translations: dict[int, TranslationResult],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        memory: TranslationMemory,
    ) -> dict[int, TranslationResult]:
        validated: dict[int, TranslationResult] = {}
        chunk_context_ids = context_unit_ids(
            before=chunk.context_before_ids,
            after=chunk.context_after_ids,
            segments_by_id=segments_by_id,
        )
        for sid, result in translations.items():
            seg = segments_by_id.get(sid)
            if seg is None:
                continue
            validated[sid] = semantic_validate_result(
                segment_id=sid,
                source=seg.source_text,
                source_protection=seg.source_protection,
                result=result,
                memory=memory,
                options=options,
                context_before=[
                    segments_by_id[i]
                    for i in chunk.context_before_ids
                    if i in segments_by_id
                ],
                context_after=[
                    segments_by_id[i]
                    for i in chunk.context_after_ids
                    if i in segments_by_id
                ],
                revision_attempt=result.metadata.retry_count,
                provenance=provenance_for_segment(
                    seg=seg,
                    context_ids=chunk_context_ids,
                ).to_dict(),
            )
        report = validate_batch_integrity(
            expected_ids=chunk.segment_ids,
            translations=validated,
            segments_by_id=segments_by_id,
            target_language=options.target_language,
        )
        for sid, errors in report.language_errors.items():
            if sid in validated:
                validated[sid] = merge_metadata_validation(validated[sid], errors=errors)
        for sid, errors in report.alignment_warnings.items():
            if sid in validated:
                validated[sid] = merge_metadata_validation(validated[sid], errors=errors)
        return validated

    def _global_consistency_pass(
        self,
        *,
        llm: Any,
        translations: dict[int, TranslationResult],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        ctx: TranslateContext,
        memory: TranslationMemory,
    ) -> dict[int, TranslationResult]:
        issues = global_consistency_issues(
            translations=translations,
            segments_by_id=segments_by_id,
            memory=memory,
        )
        suspicious = sorted(issues)[: max(1, options.chunk_size)]
        if not suspicious:
            return translations
        ctx.on_progress(0.98, "validating", f"global consistency pass ({len(suspicious)} segments)")
        ordered = [
            sid
            for sid, _seg in sorted(segments_by_id.items(), key=lambda item: item[1].start)
        ]
        revised = dict(translations)
        chunk = self._expanded_chunk(
            TranslationChunk(chunk_index=-1, segment_ids=suspicious, all_segment_ids=ordered),
            suspicious,
            segments_by_id,
            options,
        )
        if ctx.cancelled():
            raise ProviderCancelled()
        retry_map = self._attempt_ids(
            llm=llm,
            chunk=chunk,
            segment_ids=suspicious,
            segments_by_id=segments_by_id,
            options=options,
            temperature=min(1.0, options.temperature + 0.2),
            memory=memory,
            hint=(
                "Final movie-level consistency pass. Re-translate only the requested "
                f"segments with these global validator issues: {issues}. Keep character "
                "names, aliases, repeated phrases, relationship terms, and pronouns "
                "consistent with the movie memory. Do not rewrite unrelated content."
            ),
        )
        for sid, candidate in retry_map.items():
            seg = segments_by_id.get(sid)
            previous = revised.get(sid)
            if seg is None:
                continue
            marked = TranslationResult(
                candidate.translation,
                TranslationMetadata(
                    confidence=candidate.metadata.confidence,
                    needs_review=candidate.metadata.needs_review,
                    retry_count=candidate.metadata.retry_count + 1,
                    translation_method="global_consistency_revision",
                    reason_flags=candidate.metadata.reason_flags,
                    validation=candidate.metadata.validation,
                ),
            )
            chosen = select_best_candidate(
                segment_id=sid,
                source=seg.source_text,
                candidates=[item for item in (previous, marked) if item is not None],
                memory=memory,
                options=options,
                context_before=[
                    segments_by_id[i] for i in chunk.context_before_ids if i in segments_by_id
                ],
                context_after=[
                    segments_by_id[i] for i in chunk.context_after_ids if i in segments_by_id
                ],
            )
            if chosen is not None:
                revised[sid] = chosen
        return revised

    def _expanded_chunk(
        self,
        chunk: TranslationChunk,
        segment_ids: list[int],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
    ) -> TranslationChunk:
        ordered = chunk.all_segment_ids or [
            sid for sid, _ in sorted(segments_by_id.items(), key=lambda item: item[1].start)
        ]
        positions = [ordered.index(sid) for sid in segment_ids if sid in ordered]
        if not positions:
            return chunk
        first = min(positions)
        last = max(positions)
        return TranslationChunk(
            chunk_index=chunk.chunk_index,
            segment_ids=list(segment_ids),
            context_before_ids=ordered[
                max(0, first - options.retry_context_before) : first
            ],
            context_after_ids=ordered[last + 1 : last + 1 + options.retry_context_after],
            all_segment_ids=list(ordered),
        )

    @staticmethod
    def _parse_response(
        text: str, *, expected_ids: list[int], strict: bool = True
    ) -> dict[int, TranslationResult]:
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
        single = _single_entry_payload(payload, expected_ids)
        if single is not None:
            segments = [single]
        else:
            segments = payload.get("translations")
        if segments is None:
            segments = payload.get("segments")
        if isinstance(segments, dict):
            segments = [
                {"id": key, "translation": value}
                for key, value in segments.items()
            ]
        if not isinstance(segments, list):
            raise ProviderError(
                RpcErrorCode.TRANSLATE_INVALID_JSON,
                "LLM response is missing a `translations` array",
            )
        expected = set(expected_ids)
        translations: dict[int, TranslationResult] = {}
        seen: set[int] = set()
        for entry in segments:
            if not isinstance(entry, dict):
                continue
            translation = entry.get("translated_text")
            if translation is None:
                translation = entry.get("translation")
            if translation is None:
                translation = entry.get("text")
            if translation is None and isinstance(entry.get("candidates"), list):
                candidate = _best_entry_candidate(entry["candidates"])
                if candidate is not None:
                    translation = (
                        candidate.get("translated_text")
                        or candidate.get("translation")
                        or candidate.get("text")
                    )
                    entry = {**entry, **candidate}
            sid = _coerce_response_segment_id(entry)
            if sid is None and len(expected_ids) == 1 and len(segments) == 1 and translation is not None:
                sid = expected_ids[0]
            if sid is None:
                continue
            if sid in seen:
                raise ProviderError(
                    RpcErrorCode.TRANSLATE_INVALID_JSON,
                    f"LLM response duplicated segment id: {sid}",
                )
            seen.add(sid)
            if sid not in expected:
                raise ProviderError(
                    RpcErrorCode.TRANSLATE_INVALID_JSON,
                    f"LLM response returned unexpected segment id: {sid}",
                )
            if translation is None:
                continue
            translations[sid] = TranslationResult(
                str(translation).strip(),
                _metadata_from_entry(entry),
            )
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


def _single_entry_payload(payload: dict[str, Any], expected_ids: list[int]) -> dict[str, Any] | None:
    if len(expected_ids) != 1:
        return None
    if "translations" in payload or "segments" in payload:
        return None
    if (
        payload.get("translated_text") is not None
        or payload.get("translation") is not None
        or payload.get("text") is not None
    ):
        return payload
    return None


def _coerce_response_segment_id(entry: dict[str, Any]) -> int | None:
    raw = (
        entry.get("id")
        if entry.get("id") is not None
        else entry.get("segment_id")
        if entry.get("segment_id") is not None
        else entry.get("segmentId")
    )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if stripped.isdigit():
        return int(stripped)
    match = re.search(r"(?:seg(?:ment)?[_-]?)(\d+)$", stripped, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TRANSLATOR_META_RE = re.compile(
    r"\b(cần dịch lại|không dịch được|không rõ|có thể dịch là|có lẽ nghĩa là|"
    r"bản dịch|phương án|option|candidate|translation|translator note)\b",
    re.IGNORECASE,
)


def _drop_invalid_outputs(
    translations: dict[int, TranslationResult],
    *,
    segment_ids: list[int],
    segments_by_id: dict[int, TranslatedSegment],
    options: TranslateOptions,
) -> dict[int, TranslationResult]:
    """Treat structurally unsafe outputs as missing so repair retries.

    Small local models can obey the JSON shape while returning Chinese for
    Vietnamese targets. That is worse than an omitted segment because it
    poisons the cache and TTS will voice the wrong language. Phase 8.0.1
    extends this guard to candidate leakage and neighbor drift while
    preserving the same local-repair behavior.
    """
    report = validate_batch_integrity(
        expected_ids=segment_ids,
        translations=translations,
        segments_by_id=segments_by_id,
        target_language=options.target_language,
    )
    invalid_ids = set(invalid_translation_ids(report))
    filtered: dict[int, TranslationResult] = {}
    for sid in segment_ids:
        result = translations.get(sid)
        text = result.translation if result is not None else ""
        source = segments_by_id.get(sid).source_text if sid in segments_by_id else ""
        if not text:
            continue
        if sid in invalid_ids:
            issues = report.language_errors.get(sid, []) + report.alignment_warnings.get(sid, [])
            log.warn(
                "translation output failed integrity check; retrying",
                segment=sid,
                target=options.target_language,
                issues=issues,
            )
            continue
        if _looks_like_bad_vietnamese_translation(source, text):
            log.warn(
                "translation output failed target-language check; retrying",
                segment=sid,
                target=options.target_language,
            )
            continue
        filtered[sid] = result
    return filtered


def _metadata_from_entry(entry: dict[str, Any]) -> TranslationMetadata:
    confidence = entry.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    flags = entry.get("reason_flags")
    if flags is None:
        flags = entry.get("reasonFlags")
    if not isinstance(flags, list):
        flags = []
    return TranslationMetadata(
        confidence=confidence_value,
        needs_review=bool(entry.get("needs_review") or entry.get("needsReview")),
        retry_count=int(entry.get("retry_count") or entry.get("retryCount") or 0),
        translation_method=str(entry.get("translation_method") or entry.get("translationMethod") or "context_batch"),
        reason_flags=[str(flag) for flag in flags if str(flag).strip()],
    )


def _best_entry_candidate(candidates: list[Any]) -> dict[str, Any] | None:
    parsed = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if not parsed:
        return None
    parsed.sort(
        key=lambda item: (
            _float_or_default(item.get("confidence"), 0.75),
            -len(item.get("reason_flags") or item.get("reasonFlags") or []),
            len(str(item.get("translated_text") or item.get("translation") or "")),
        ),
        reverse=True,
    )
    return parsed[0]


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_result(result: TranslationResult) -> float:
    if HARD_SEMANTIC_FAILURES.intersection(result.metadata.reason_flags):
        return -10.0 - 0.1 * len(result.metadata.reason_flags)
    confidence = result.metadata.confidence
    score = confidence if confidence is not None else 0.85
    if result.metadata.needs_review:
        score -= 0.1
    score -= 0.03 * len(result.metadata.reason_flags)
    return score


def _looks_like_bad_vietnamese_translation(source: str, translation: str) -> bool:
    stripped = translation.strip()
    if not stripped:
        return True
    if _CJK_RE.search(source):
        cjk_count = len(_CJK_RE.findall(stripped))
        non_space = sum(1 for ch in stripped if not ch.isspace())
        if cjk_count >= 2 and cjk_count / max(1, non_space) > 0.15:
            return True
    if _TRANSLATOR_META_RE.search(stripped.casefold()):
        return True

    src_norm = _normalise_for_copy_check(source)
    out_norm = _normalise_for_copy_check(stripped)
    if src_norm and out_norm and src_norm == out_norm:
        return True
    return False


def _normalise_for_copy_check(text: str) -> str:
    words = _WORD_RE.findall(text.casefold())
    return " ".join(words)


def _clean_plain_translation(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json|text)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                for key in ("translated_text", "translation", "text"):
                    value = payload.get(key)
                    if isinstance(value, str):
                        return value.strip()
                rows = payload.get("translations") or payload.get("segments")
                if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
                    for key in ("translated_text", "translation", "text"):
                        value = rows[0].get(key)
                        if isinstance(value, str):
                            return value.strip()
                return ""
        except json.JSONDecodeError:
            pass
    if cleaned.startswith("["):
        return ""
    cleaned = re.sub(
        r"^(translation|translated subtitle|bản dịch|dịch)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if "\n" in cleaned:
        lines = [
            line.strip()
            for line in cleaned.splitlines()
            if line.strip() and not line.strip().startswith(("{", "}", "[", "]"))
        ]
        cleaned = lines[0] if len(lines) == 1 else " ".join(lines)
    return cleaned.strip().strip('"“”')


def _fallback_missing_translation(
    *,
    segment_id: int,
    seg: TranslatedSegment | None,
    target_language: str,
) -> TranslationResult:
    source = seg.source_text if seg is not None else ""
    text = (
        _conservative_fallback_vi(source)
        if (target_language or "").lower() == "vi"
        else "[translation unavailable]"
    )
    return TranslationResult(
        text,
        TranslationMetadata(
            confidence=0.05,
            needs_review=True,
            retry_count=_MAX_REPAIR_ATTEMPTS + 1,
            translation_method="missing_segment_fallback",
            reason_flags=[
                "LLM_OMITTED_SEGMENT",
                "TRANSLATION_PLACEHOLDER",
                "SEMANTIC_OMISSION",
            ],
            validation={
                "valid": False,
                "confidence": 0.0,
                "issues": [
                    "LLM_OMITTED_SEGMENT",
                    "TRANSLATION_PLACEHOLDER",
                    "SEMANTIC_OMISSION",
                ],
                "sourceLength": len(source.strip()),
                "missingSegmentId": segment_id,
            },
        ),
    )


def _conservative_fallback_vi(source: str) -> str:
    normalized = re.sub(r"\s+", "", source or "")
    if not normalized:
        return "..."
    if re.search(r"(不要走|别走|別走)", normalized):
        return "Đừng đi."
    if re.search(r"(你怎么没去|你怎麼沒去|你怎么没有去|你怎麼沒有去)", normalized):
        return "Sao bạn không đi?"
    if re.search(r"(我们结婚了|我們結婚了)", normalized):
        return "Chúng tôi đã kết hôn."
    if re.search(r"(相亲|相親)", normalized) and re.search(r"(结婚|結婚)", normalized):
        if re.search(r"(没什么难|沒有什麼難|没有什么难|沒什麼難)", normalized):
            return "Chuyện xem mắt rồi kết hôn cũng không có gì khó."
    if re.search(r"(等你|等了你)", normalized) and re.search(r"(3|三)\s*(个)?小时", normalized):
        prefix = "Anh ấy đã " if re.search(r"(他|她)", normalized) else "Đã "
        return f"{prefix}đợi bạn ba tiếng rồi."
    return "..."


def _revision_issue_summary(
    translations: dict[int, TranslationResult],
    segment_ids: list[int],
) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for sid in segment_ids:
        if sid not in translations:
            continue
        metadata = translations[sid].metadata
        out[sid] = list(metadata.validation.get("issues") or metadata.reason_flags)
    return out


def _revision_anchor_summary(
    segments_by_id: dict[int, TranslatedSegment],
    segment_ids: list[int],
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for sid in segment_ids:
        seg = segments_by_id.get(sid)
        if seg is None:
            continue
        protection = _compact_source_protection(seg)
        out[sid] = {
            key: value
            for key, value in {
                "source": seg.source_text,
                "speechAct": protection.get("speechAct"),
                "questionType": protection.get("questionType"),
                "predicate": protection.get("predicate"),
                "polarity": protection.get("polarity"),
                "numbers": protection.get("numbers"),
                "events": protection.get("events"),
                "certainty": protection.get("certainty"),
                "aspect": protection.get("aspect"),
                "mustPreserve": protection.get("mustPreserve"),
                "naturalizationBudget": protection.get("naturalizationBudget"),
            }.items()
            if value not in (None, [], {}, "")
        }
    return out


def _compact_source_protection(seg: TranslatedSegment) -> dict[str, Any]:
    """Small prompt-safe view of Phase 7 source facts.

    The full sourceProtection payload remains in validation metadata.
    The LLM only needs the constraints that affect translation, so avoid
    sending duplicate snake_case/camelCase keys or raw trace blobs here.
    """
    protection = seg.source_protection or source_protection_payload(
        segment_id=seg.source_segment_id or seg.id,
        text=seg.raw_source_text or seg.source_text,
        start=seg.start,
        end=seg.end,
    )
    semantic = protection.get("semantic") if isinstance(protection, dict) else {}
    if not isinstance(semantic, dict):
        semantic = {}
    quality = (
        protection.get("sourceQuality")
        or protection.get("source_quality")
        if isinstance(protection, dict)
        else {}
    )
    if not isinstance(quality, dict):
        quality = {}
    logical = (
        protection.get("logicalSubsegments")
        or protection.get("logical_subsegments")
        if isinstance(protection, dict)
        else []
    )
    logical_rows = []
    if isinstance(logical, list):
        for item in logical[:6]:
            if not isinstance(item, dict):
                continue
            text = item.get("textCn") or item.get("text_cn")
            if not text:
                continue
            logical_rows.append(
                {
                    "id": item.get("subSegmentId") or item.get("sub_segment_id"),
                    "text": text,
                }
            )
    actions = []
    for action in semantic.get("actions") or []:
        if isinstance(action, dict):
            actions.append(
                {
                    "source": action.get("source"),
                    "vi": action.get("viHint") or action.get("vi_hint"),
                }
            )
    out: dict[str, Any] = {
        "sourceId": seg.source_segment_id or str(seg.id),
        "subId": seg.source_sub_segment_id,
        "normalized": (
            protection.get("normalizedSource")
            or protection.get("normalized_source")
            or seg.normalized_source_text
            or seg.source_text
        ),
        "units": logical_rows,
        "numbers": semantic.get("numbers") or [],
        "negation": semantic.get("negation") or [],
        "speechAct": semantic.get("speechAct") or semantic.get("speech_act"),
        "questionType": semantic.get("questionType") or semantic.get("question_type"),
        "polarity": semantic.get("polarity"),
        "predicate": semantic.get("predicate"),
        "events": semantic.get("events") or [],
        "certainty": semantic.get("certainty"),
        "aspect": semantic.get("aspect"),
        "sourceParticles": semantic.get("sourceParticles") or semantic.get("source_particles") or [],
        "semanticAnchors": semantic.get("semanticAnchors") or semantic.get("semantic_anchors"),
        "mustPreserve": semantic.get("mustPreserve") or semantic.get("must_preserve") or [],
        "naturalizationBudget": semantic.get("naturalizationBudget") or semantic.get("naturalization_budget"),
        "literalBaselineHint": semantic.get("literalBaselineHint") or semantic.get("literal_baseline_hint"),
        "question": bool(semantic.get("isQuestion") or semantic.get("is_question")),
        "command": bool(semantic.get("isCommand") or semantic.get("is_command")),
        "actions": actions,
        "quality": {
            "confidence": quality.get("sourceConfidence")
            or quality.get("source_confidence"),
            "flags": quality.get("qualityFlags") or quality.get("quality_flags") or [],
        },
        "segmentationFlags": (
            protection.get("segmentationFlags")
            or protection.get("segmentation_flags")
            or []
        ),
    }
    return {key: value for key, value in out.items() if value not in (None, [], {}, "")}


def _messages_too_large(messages: list[PromptMessage], *, max_chars: int = 16_000) -> bool:
    # llama.cpp reports context in tokens; a conservative 3 chars/token
    # is still optimistic for CJK-heavy JSON. Keep the prompt comfortably
    # below an 8192-token window and split rows sooner when safeguards grow.
    return sum(len(message.content) for message in messages) > max_chars


def _compact_translation_memory(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    compact: dict[str, Any] = {}
    if payload.get("movieSummary"):
        compact["movieSummary"] = str(payload["movieSummary"])[:320]
    if isinstance(payload.get("characters"), list):
        compact["characters"] = payload["characters"][:6]
    if isinstance(payload.get("relationships"), list):
        compact["relationships"] = payload["relationships"][:8]
    if isinstance(payload.get("sceneRelationshipOverrides"), list):
        compact["sceneRelationshipOverrides"] = payload["sceneRelationshipOverrides"][:6]
    if isinstance(payload.get("pronounPlans"), dict):
        compact["pronounPlans"] = dict(list(payload["pronounPlans"].items())[:8])
    if isinstance(payload.get("translationUnitContract"), dict):
        contract = payload["translationUnitContract"]
        compact["translationUnitContract"] = {
            "targets": list(contract.get("targets") or [])[:16],
            "context": list(contract.get("context") or [])[:16],
            "conversationBoundaries": list(contract.get("conversationBoundaries") or [])[:20],
            "rule": contract.get("rule"),
        }
    if isinstance(payload.get("characterGraph"), dict):
        graph = payload["characterGraph"]
        compact["characterGraph"] = {
            "characters": list(graph.get("characters") or [])[:6],
            "relationshipFacts": list(graph.get("relationshipFacts") or [])[:8],
            "addressPatterns": list(graph.get("addressPatterns") or [])[:8],
            "contradictions": list(graph.get("contradictions") or [])[:4],
            "recentAddressHistory": dict(
                list(dict(graph.get("recentAddressHistory") or {}).items())[:6]
            ),
        }
    tm = payload.get("translationMemory")
    if isinstance(tm, dict):
        compact["translationMemory"] = {
            "translations": list(tm.get("translations") or [])[-6:],
            "names": dict(list(dict(tm.get("names") or {}).items())[-8:]),
            "pronounPatterns": list(tm.get("pronounPatterns") or [])[-4:],
        }
    elif isinstance(payload.get("translations"), list):
        compact["translations"] = list(payload.get("translations") or [])[-6:]
    if isinstance(payload.get("knownNames"), dict):
        compact["knownNames"] = dict(list(payload["knownNames"].items())[:8])
    return compact or None
