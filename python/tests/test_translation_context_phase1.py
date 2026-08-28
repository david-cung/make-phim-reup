from __future__ import annotations

import json

import pytest

from movie_translator_worker.errors import RpcErrorCode
from movie_translator_worker.translation.llama_cpp_provider import LlamaCppTranslationProvider
from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationResult,
    chunk_segments_with_context,
)
from movie_translator_worker.translation.provider import ProviderError
from movie_translator_worker.translation.quality import validate_result


def _segments(ids: list[int]) -> dict[int, TranslatedSegment]:
    return {
        sid: TranslatedSegment(
            id=sid,
            source_text=f"line {sid}",
            translation="",
            start=float(pos),
            end=float(pos + 1),
        )
        for pos, sid in enumerate(ids)
    }


def test_context_window_uses_full_movie_order_when_resuming() -> None:
    chunks = chunk_segments_with_context(
        ordered_ids=[1, 2, 3, 4, 5, 6],
        todo_ids=[4, 5],
        chunk_size=15,
        context_before=2,
        context_after=2,
    )

    assert len(chunks) == 1
    assert chunks[0].segment_ids == [4, 5]
    assert chunks[0].context_before_ids == [2, 3]
    assert chunks[0].context_after_ids == [6]
    assert chunks[0].all_segment_ids == [1, 2, 3, 4, 5, 6]


def test_context_window_handles_first_and_last_subtitle() -> None:
    first = chunk_segments_with_context(
        [10, 11, 12],
        [10],
        chunk_size=1,
        context_before=5,
        context_after=5,
    )[0]
    last = chunk_segments_with_context(
        [10, 11, 12],
        [12],
        chunk_size=1,
        context_before=5,
        context_after=5,
    )[0]

    assert first.context_before_ids == []
    assert first.context_after_ids == [11, 12]
    assert last.context_before_ids == [10, 11]
    assert last.context_after_ids == []


def test_strict_response_parser_accepts_structured_metadata(tmp_path) -> None:
    payload = {
        "translations": [
            {
                "id": 7,
                "translated_text": "Em đã nói với anh rồi.",
                "confidence": 0.92,
                "reason_flags": [],
            }
        ]
    }
    out = LlamaCppTranslationProvider(tmp_path)._parse_response(
        json.dumps(payload), expected_ids=[7]
    )

    assert out[7] == "Em đã nói với anh rồi."
    assert out[7].metadata.confidence == pytest.approx(0.92)


@pytest.mark.parametrize(
    "payload",
    [
        {"translations": [{"id": 1, "translated_text": "a"}]},
        {
            "translations": [
                {"id": 2, "translated_text": "a"},
                {"id": 2, "translated_text": "b"},
            ]
        },
        {"translations": [{"id": 9, "translated_text": "unexpected"}]},
    ],
)
def test_strict_response_parser_rejects_bad_id_mapping(tmp_path, payload) -> None:
    with pytest.raises(ProviderError) as exc:
        LlamaCppTranslationProvider(tmp_path)._parse_response(
            json.dumps(payload), expected_ids=[2]
        )
    assert exc.value.code in {
        RpcErrorCode.TRANSLATE_INCOMPLETE_RESPONSE,
        RpcErrorCode.TRANSLATE_INVALID_JSON,
    }


def test_translation_memory_is_job_scoped() -> None:
    a = TranslationMemory()
    b = TranslationMemory()
    a.record(1, "陈浩来了", "Trần Hạo đến rồi.")

    assert a.prompt_payload([1])["translations"]
    assert b.prompt_payload([1])["translations"] == []


class _Ctx:
    def cancelled(self) -> bool:
        return False

    def on_progress(self, *_args) -> None:
        pass

    def on_chunk_completed(self, *_args) -> None:
        pass


def _ids_in(rendered: str) -> list[int]:
    if "CURRENT (translate these):" in rendered:
        rendered = rendered.split("CURRENT (translate these):", 1)[1].split("\n\n", 1)[0]
    found: list[int] = []
    for token in rendered.replace(",", " ").replace(":", " ").split():
        stripped = token.strip('"')
        if stripped.isdigit():
            value = int(stripped)
            if value not in found:
                found.append(value)
    return found


class _LowConfidenceThenGood:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def create_chat_completion(self, *, messages, **_kw):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        ids = _ids_in(user)
        self.calls.append(ids)
        if len(self.calls) == 1:
            confidence = 0.5
            text = "Tôi đã nói với bạn rồi."
            flags = ["AMBIGUOUS_PRONOUN"]
        else:
            confidence = 0.93
            text = "Em đã nói với anh rồi."
            flags = []
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {
                                        "id": 3,
                                        "translated_text": text,
                                        "confidence": confidence,
                                        "reason_flags": flags,
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


def test_low_confidence_triggers_expanded_context_retry(tmp_path) -> None:
    llm = _LowConfidenceThenGood()
    segments = _segments([1, 2, 3, 4, 5])
    chunk = TranslationChunk(
        chunk_index=0,
        segment_ids=[3],
        context_before_ids=[2],
        context_after_ids=[4],
        all_segment_ids=[1, 2, 3, 4, 5],
    )

    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=chunk,
        segments_by_id=segments,
        options=TranslateOptions(
            model="m.gguf",
            low_confidence_threshold=0.8,
            retry_context_before=2,
            retry_context_after=2,
            max_translation_retries=2,
        ),
        ctx=_Ctx(),
    )

    assert out[3] == "Em đã nói với anh rồi."
    assert out[3].metadata.retry_count == 1
    assert len(llm.calls) == 2


def test_pronoun_checker_flags_without_rewriting() -> None:
    memory = TranslationMemory()
    memory.record(100, "我告诉过你了", "Em đã nói với anh rồi.")

    out = validate_result(
        segment_id=101,
        source="我告诉过你了",
        result=TranslationResult("Tôi đã nói với chị rồi."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
    )

    assert out.translation == "Tôi đã nói với chị rồi."
    assert "POSSIBLE_PRONOUN_INCONSISTENCY" in out.metadata.reason_flags
    assert out.metadata.needs_review is True


def test_translation_prompt_rows_include_speaker_id(tmp_path) -> None:
    provider = LlamaCppTranslationProvider(tmp_path)
    messages = provider._build_messages(
        chunk=TranslationChunk(
            chunk_index=0,
            segment_ids=[2],
            context_before_ids=[1],
            context_after_ids=[3],
        ),
        segments_by_id={
            1: TranslatedSegment(1, "previous", "", 0.0, 1.0, speaker_id="speaker_002"),
            2: TranslatedSegment(
                2,
                "我告诉过你了。",
                "",
                1.0,
                2.0,
                speaker_id="speaker_001",
                speaker_confidence=0.92,
            ),
            3: TranslatedSegment(3, "next", "", 2.0, 3.0, speaker_id="speaker_002"),
        },
        options=TranslateOptions(model="m.gguf"),
    )

    rendered = "\n".join(message.content for message in messages)
    assert "speaker_001" in rendered
    assert "speaker_002" in rendered
    assert "speakerConfidence" in rendered
