from __future__ import annotations

from movie_translator_worker.source_protection import (
    semantic_analysis,
    source_protection_payload,
    split_logical_source_units,
    split_source_segment,
    validate_translation_against_source,
)
from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationResult,
)
from movie_translator_worker.translation.quality import semantic_validate_result


def _validate(source: str, translation: str) -> list[str]:
    return validate_translation_against_source(
        source=source,
        translation=translation,
        protection=source_protection_payload(segment_id=1, text=source),
    )


def test_duration_hours_cannot_become_years() -> None:
    issues = _validate("我给你介绍个相亲对象等你3个小时了", "Tôi giới thiệu cho bạn ba năm rồi.")

    assert "DURATION_MISMATCH" in issues


def test_chinese_year_duration_is_preserved() -> None:
    assert _validate("三年了", "Đã ba năm rồi.") == []
    assert "DURATION_MISMATCH" in _validate("三年了", "Đã ba giờ rồi.")


def test_person_quantity_is_preserved() -> None:
    assert _validate("三个人都来了", "Ba người đều đến rồi.") == []
    assert "QUANTITY_MISMATCH" in _validate("三个人都来了", "Ba năm đều đến rồi.")


def test_negation_must_survive_translation() -> None:
    assert "MISSING_NEGATION" in _validate("你怎么没去啊", "Sao bạn lại đi?")
    assert _validate("你怎么没去啊", "Sao bạn không đi?") == []


def test_dialogue_split_detects_merged_chinese_units() -> None:
    units = split_logical_source_units("你怎么没去啊我们结婚了那他是谁")

    assert units == ["你怎么没去啊", "我们结婚了", "那他是谁"]


def test_split_source_segment_keeps_traceability_and_timing() -> None:
    units = split_source_segment(
        segment_id=12,
        text="你怎么没去啊我们结婚了那他是谁",
        start=10.0,
        end=16.0,
    )

    assert [unit.sub_segment_id for unit in units] == ["12a", "12b", "12c"]
    assert all(unit.source_segment_id == "12" for unit in units)
    assert units[0].start == 10.0
    assert units[-1].end == 16.0


def test_interview_action_cannot_become_unrelated_sentence() -> None:
    assert "MISSING_ACTION" in _validate("我来面试", "Tôi đến chơi.")
    assert _validate("我来面试", "Tôi đến phỏng vấn.") == []


def test_untranslated_chinese_is_rejected() -> None:
    assert "UNTRANSLATED_CHINESE" in _validate("我们结婚了", "我们结婚了")


def test_semantic_validator_records_source_protection_metadata() -> None:
    seg = TranslatedSegment(
        id=1,
        source_text="等你3个小时了",
        translation="",
        start=0.0,
        end=2.0,
        source_protection=source_protection_payload(segment_id=1, text="等你3个小时了"),
    )

    result = semantic_validate_result(
        segment_id=1,
        source=seg.source_text,
        source_protection=seg.source_protection,
        result=TranslationResult("Đợi bạn ba năm rồi."),
        memory=TranslationMemory.from_segments([seg]),
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert result.metadata.needs_review is True
    assert "POSSIBLE_MEANING_CHANGE" in result.metadata.reason_flags
    assert result.metadata.validation["sourceProtection"]["semantic"]["numbers"][0]["value"] == 3


def test_semantic_analysis_extracts_numbers_and_question() -> None:
    analysis = semantic_analysis("你怎么没去啊等你3个小时了")

    assert analysis["isQuestion"] is True
    assert analysis["numbers"][0]["unit"] == "hour"
    assert analysis["numbers"][0]["value"] == 3
