from __future__ import annotations

import json

import pytest

from movie_translator_worker.errors import RpcErrorCode
from movie_translator_worker.translation.integrity import (
    BATCH_CARDINALITY_ERROR,
    CANDIDATE_LEAKAGE,
    SEGMENT_MERGE_ERROR,
    SOURCE_ALIGNMENT_DRIFT,
    SUSPICIOUS_TRANSLATION_DUPLICATION,
    UNTRANSLATED_SOURCE_FRAGMENT,
    duplicate_translation_issues,
    speaker_merge_allowed,
    validate_batch_integrity,
    validate_vietnamese_output,
)
from movie_translator_worker.translation.llama_cpp_provider import LlamaCppTranslationProvider
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationResult,
)
from movie_translator_worker.translation.provider import ProviderError


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
        self.calls.append(_ids_in_current(user))
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


def _ids_in_current(rendered: str) -> list[int]:
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


def _seg(segment_id: int, text: str, speaker: str = "A") -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=float(segment_id),
        end=float(segment_id) + 1.0,
        speaker_id=speaker,
        source_segment_id=f"seg_{segment_id:05d}",
    )


def test_cross_speaker_interruption_is_not_mergeable() -> None:
    first = _seg(1, "如果你再敢", "A")
    second = _seg(2, "怎么样", "B")

    assert speaker_merge_allowed(first, second) is False


def test_unknown_speakers_are_not_mergeable_for_translation() -> None:
    assert speaker_merge_allowed(_seg(1, "a", "UNKNOWN"), _seg(2, "b", "UNKNOWN")) is False


def test_candidate_leakage_is_rejected() -> None:
    validation = validate_vietnamese_output(
        "Chúng tôi kết hôn rồi. / Chúng ta kết hôn rồi. / Bọn tôi kết hôn rồi.",
        source="我们结婚了",
    )

    assert CANDIDATE_LEAKAGE in validation.errors


def test_language_residue_is_rejected() -> None:
    validation = validate_vietnamese_output("Sao bạn没去?", source="你怎么没去啊")

    assert UNTRANSLATED_SOURCE_FRAGMENT in validation.errors


def test_batch_cardinality_missing_id_is_reported() -> None:
    report = validate_batch_integrity(
        expected_ids=[1, 2, 3, 4, 5],
        translations={1: TranslationResult("a"), 2: TranslationResult("b"), 4: TranslationResult("d"), 5: TranslationResult("e")},
        segments_by_id={i: _seg(i, f"line {i}") for i in range(1, 6)},
        target_language="vi",
    )

    assert report.missing == [3]
    assert report.valid is False
    assert report.metrics()["missing_segment_count"] == 1
    assert BATCH_CARDINALITY_ERROR in {BATCH_CARDINALITY_ERROR}


def test_neighbor_drift_is_flagged() -> None:
    segments = {
        10: _seg(10, "谢谢你送我回家", "A"),
        11: _seg(11, "沈总您这一回国不回老宅", "B"),
    }
    report = validate_batch_integrity(
        expected_ids=[10],
        translations={10: TranslationResult("Tổng giám đốc Thẩm, lần này về nước mà không về nhà cũ.")},
        segments_by_id=segments,
        target_language="vi",
    )

    assert SOURCE_ALIGNMENT_DRIFT in report.alignment_warnings[10]


def test_one_output_with_neighbor_meaning_is_merge_error() -> None:
    segments = {
        1: _seg(1, "那他是谁？", "A"),
        2: _seg(2, "不知道。", "B"),
    }
    report = validate_batch_integrity(
        expected_ids=[1],
        translations={1: TranslationResult("Thế anh ấy là ai? Tôi không biết.")},
        segments_by_id=segments,
        target_language="vi",
    )

    assert SEGMENT_MERGE_ERROR in report.alignment_warnings[1]


def test_duplicate_different_sources_are_flagged() -> None:
    translations = {
        1: TranslationResult("Tổng giám đốc Thẩm, lần này về nước mà không về nhà cũ."),
        2: TranslationResult("Tổng giám đốc Thẩm, lần này về nước mà không về nhà cũ."),
    }
    issues = duplicate_translation_issues(
        translations,
        {1: _seg(1, "谢谢你送我回家"), 2: _seg(2, "沈总您这一回国不回老宅")},
    )

    assert SUSPICIOUS_TRANSLATION_DUPLICATION in issues[1]
    assert SUSPICIOUS_TRANSLATION_DUPLICATION in issues[2]


def test_provider_retries_candidate_leakage_locally(tmp_path) -> None:
    llm = _ScriptedLlm(
        [
            {1: "Chúng tôi kết hôn rồi. / Chúng ta kết hôn rồi. / Bọn tôi kết hôn rồi."},
            {1: "Bọn tôi kết hôn rồi."},
        ]
    )
    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=TranslationChunk(chunk_index=0, segment_ids=[1], all_segment_ids=[1]),
        segments_by_id={1: _seg(1, "我们结婚了")},
        options=TranslateOptions(model="m.gguf", source_language="zh", target_language="vi"),
        ctx=_Ctx(),
    )

    assert out[1] == "Bọn tôi kết hôn rồi."
    assert len(llm.calls) == 2


def test_strict_parser_rejects_missing_output_count(tmp_path) -> None:
    with pytest.raises(ProviderError) as exc:
        LlamaCppTranslationProvider(tmp_path)._parse_response(
            json.dumps({"translations": [{"id": 1, "translated_text": "a"}]}),
            expected_ids=[1, 2],
        )

    assert exc.value.code == RpcErrorCode.TRANSLATE_INCOMPLETE_RESPONSE
