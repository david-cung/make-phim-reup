from __future__ import annotations

import json

from movie_translator_worker.translation.llama_cpp_provider import LlamaCppTranslationProvider
from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationMetadata,
    TranslationResult,
)
from movie_translator_worker.translation.quality import (
    global_consistency_issues,
    select_best_candidate,
    semantic_validate_result,
)


class _Ctx:
    def cancelled(self) -> bool:
        return False

    def on_progress(self, *_args) -> None:
        pass

    def on_chunk_completed(self, *_args) -> None:
        pass


class _ScriptedLlm:
    def __init__(self, replies: list[dict[int, str]]) -> None:
        self.replies = replies
        self.calls: list[list[int]] = []

    def create_chat_completion(self, *, messages, **_kw):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.calls.append(_ids_in(user))
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {
                                        "id": sid,
                                        "translated_text": text,
                                        "confidence": 0.92,
                                        "reason_flags": [],
                                    }
                                    for sid, text in self.replies[idx].items()
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


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


def _seg(segment_id: int, text: str, start: float = 0.0) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=start + 1.0,
    )


def test_semantic_failure_sets_structured_confidence() -> None:
    seg = _seg(1, "不要走。")
    out = semantic_validate_result(
        segment_id=1,
        source=seg.source_text,
        result=TranslationResult("Đi đi.", TranslationMetadata(confidence=0.94)),
        memory=TranslationMemory.from_segments([seg]),
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert "POSSIBLE_MEANING_CHANGE" in out.metadata.reason_flags
    assert out.metadata.needs_review is True
    assert out.metadata.validation["translationConfidence"] == 0.94
    assert out.metadata.validation["validationConfidence"] < 0.8
    assert out.metadata.validation["finalConfidence"] < 0.9


def test_semantic_failure_retries_and_repairs_translation(tmp_path) -> None:
    seg = _seg(1, "不要走。")
    llm = _ScriptedLlm([{1: "Đi đi."}, {1: "Đừng đi."}])

    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=TranslationChunk(chunk_index=0, segment_ids=[1], all_segment_ids=[1]),
        segments_by_id={1: seg},
        options=TranslateOptions(model="m.gguf"),
        ctx=_Ctx(),
    )

    assert out[1].translation == "Đừng đi."
    assert out[1].metadata.needs_review is False
    assert len(llm.calls) == 2


def test_revision_attempts_are_capped_at_two(tmp_path) -> None:
    seg = _seg(1, "不要走。")
    llm = _ScriptedLlm([{1: "Đi đi."}])

    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=TranslationChunk(chunk_index=0, segment_ids=[1], all_segment_ids=[1]),
        segments_by_id={1: seg},
        options=TranslateOptions(model="m.gguf", max_translation_retries=5),
        ctx=_Ctx(),
    )

    assert out[1].translation == "Đi đi."
    assert out[1].metadata.needs_review is True
    assert len(llm.calls) == 3


def test_candidate_selection_prefers_semantically_valid_candidate() -> None:
    seg = _seg(1, "不要走。")
    memory = TranslationMemory.from_segments([seg])

    best = select_best_candidate(
        segment_id=1,
        source=seg.source_text,
        candidates=[
            TranslationResult("Đi đi.", TranslationMetadata(confidence=0.96)),
            TranslationResult("Đừng đi.", TranslationMetadata(confidence=0.90)),
        ],
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
    )

    assert best is not None
    assert best.translation == "Đừng đi."
    assert best.metadata.validation["candidateCount"] == 2


def test_parser_supports_multiple_candidates(tmp_path) -> None:
    payload = {
        "translations": [
            {
                "id": 1,
                "candidates": [
                    {"translated_text": "Đi đi.", "confidence": 0.4},
                    {"translated_text": "Đừng đi.", "confidence": 0.9},
                ],
            }
        ]
    }

    out = LlamaCppTranslationProvider(tmp_path)._parse_response(
        json.dumps(payload, ensure_ascii=False),
        expected_ids=[1],
    )

    assert out[1].translation == "Đừng đi."
    assert out[1].metadata.confidence == 0.9


def test_global_inconsistency_detection_flags_name_variants() -> None:
    segments = {
        1: _seg(1, "陈浩来了。", 0.0),
        2: _seg(2, "陈浩别走。", 1.0),
    }
    memory = TranslationMemory.from_segments(list(segments.values()))
    translations = {
        1: TranslationResult("Trần Hạo đến rồi."),
        2: TranslationResult("Hạo, đừng đi."),
    }

    issues = global_consistency_issues(
        translations=translations,
        segments_by_id=segments,
        memory=memory,
    )

    assert issues == {
        2: [
            "POSSIBLE_GLOBAL_INCONSISTENCY",
            "POSSIBLE_CHARACTER_REFERENCE_CONFLICT",
        ]
    }


def test_partial_validation_retries_only_failed_segment(tmp_path) -> None:
    segments = {
        1: _seg(1, "不要走。", 0.0),
        2: _seg(2, "你好。", 1.0),
    }
    llm = _ScriptedLlm(
        [
            {1: "Đi đi.", 2: "Xin chào."},
            {1: "Đừng đi."},
        ]
    )

    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=TranslationChunk(
            chunk_index=0,
            segment_ids=[1, 2],
            all_segment_ids=[1, 2],
        ),
        segments_by_id=segments,
        options=TranslateOptions(model="m.gguf"),
        ctx=_Ctx(),
    )

    assert out[1].translation == "Đừng đi."
    assert out[2].translation == "Xin chào."
    assert llm.calls[1] == [1]
