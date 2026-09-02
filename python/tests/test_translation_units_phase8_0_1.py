from __future__ import annotations

from movie_translator_worker.translation.integrity import (
    EMPTY_TRANSLATION,
    validate_batch_integrity,
)
from movie_translator_worker.translation.llama_cpp_provider import LlamaCppTranslationProvider
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationResult,
)
from movie_translator_worker.translation.target_language import (
    TRANSLATOR_COMMENTARY_LEAK,
    UNTRANSLATED_SOURCE_FRAGMENT,
    validate_target_language,
)
from movie_translator_worker.translation.units import (
    MISSING_RESULT,
    SEMANTIC_OMISSION,
    UNKNOWN_RESULT_ID,
    AtomicSourceUnit,
    TimeRange,
    ownership_payload,
    resolve_conversation_structure,
    source_unit_from_segment,
    validate_translation_contract,
)


def _seg(
    segment_id: int,
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    speaker: str | None = "speaker_001",
) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=end,
        speaker_id=speaker,
        speaker_confidence=0.91 if speaker else None,
        source_segment_id=f"u_{segment_id:03d}",
    )


def test_atomic_source_unit_does_not_require_time_or_speaker() -> None:
    unit = AtomicSourceUnit(
        unit_id="u_text_001",
        source_text="manual line",
        time_range=TimeRange(),
    )

    assert unit.time_range.start is None
    assert unit.time_range.end is None
    assert unit.speaker.ref is None
    assert unit.to_dict()["sourceType"] == "transcript"


def test_source_unit_from_segment_preserves_traceable_identity() -> None:
    unit = source_unit_from_segment(_seg(7, "Hello", speaker=None))

    assert unit.unit_id == "u_007"
    assert unit.speaker.ref is None
    assert unit.speaker.confidence == 0.0
    assert unit.time_range.start == 0.0


def test_translation_contract_reports_missing_unknown_and_empty_results() -> None:
    report = validate_translation_contract(
        expected_unit_ids=["u_001", "u_002"],
        results={
            "u_001": TranslationResult(""),
            "u_003": TranslationResult("leaked"),
        },
    )
    codes = [issue.code for issue in report.issues]

    assert report.valid is False
    assert MISSING_RESULT in codes
    assert UNKNOWN_RESULT_ID in codes
    assert SEMANTIC_OMISSION in codes


def test_conversation_structure_keeps_uncertain_units_traceable() -> None:
    units = [
        source_unit_from_segment(_seg(1, "first", start=0.0, end=1.0, speaker="speaker_001")),
        source_unit_from_segment(_seg(2, "continued", start=1.1, end=2.0, speaker="speaker_001")),
        source_unit_from_segment(_seg(3, "unknown", start=2.1, end=3.0, speaker=None)),
    ]

    turns, boundaries, groups = resolve_conversation_structure(units)

    assert boundaries[0].relation == "same_speaker_continuation"
    assert boundaries[1].relation == "unknown_speaker_transition"
    assert turns[0].unit_ids == ["u_001", "u_002"]
    assert turns[1].unit_ids == ["u_003"]
    assert groups[0].member_unit_ids == ["u_001", "u_002"]


def test_ownership_payload_separates_context_from_owned_content() -> None:
    payload = ownership_payload(
        source_unit_id="u_102",
        context_unit_ids=["u_100", "u_101", "u_103"],
        target=True,
    )

    assert payload["sourceUnitIds"] == ["u_102"]
    assert payload["contextUnitIds"] == ["u_100", "u_101", "u_103"]
    assert payload["target"] is True


def test_target_language_adapter_is_called_by_batch_integrity() -> None:
    seg = _seg(1, "我们结婚了")
    report = validate_batch_integrity(
        expected_ids=[1],
        translations={1: TranslationResult("我们结婚了")},
        segments_by_id={1: seg},
        target_language="vi",
    )

    assert UNTRANSLATED_SOURCE_FRAGMENT in report.language_errors[1]


def test_non_vietnamese_target_uses_generic_adapter() -> None:
    seg = _seg(1, "我们结婚了")
    report = validate_batch_integrity(
        expected_ids=[1],
        translations={1: TranslationResult("我们结婚了")},
        segments_by_id={1: seg},
        target_language="zh",
    )

    assert report.language_errors == {}


def test_batch_integrity_exposes_generic_contract_issues() -> None:
    report = validate_batch_integrity(
        expected_ids=[1, 2],
        translations={1: TranslationResult("")},
        segments_by_id={1: _seg(1, "a"), 2: _seg(2, "b")},
        target_language="en",
    )
    codes = [str(issue["code"]) for issue in report.contract_issues]

    assert MISSING_RESULT in codes
    assert SEMANTIC_OMISSION in codes
    assert EMPTY_TRANSLATION in report.language_errors[1]


def test_vietnamese_adapter_rejects_translator_commentary() -> None:
    validation = validate_target_language("vi", "Đoạn này cần dịch lại.", source="你好")

    assert TRANSLATOR_COMMENTARY_LEAK in validation.errors


def test_prompt_contains_unit_contract_and_read_only_context(tmp_path) -> None:
    provider = LlamaCppTranslationProvider(tmp_path)
    segments = {
        1: _seg(1, "before", start=0.0, end=1.0, speaker="speaker_001"),
        2: _seg(2, "target", start=1.1, end=2.0, speaker="speaker_002"),
        3: _seg(3, "after", start=2.1, end=3.0, speaker="speaker_003"),
    }

    messages = provider._build_messages(
        chunk=TranslationChunk(
            chunk_index=0,
            segment_ids=[2],
            context_before_ids=[1],
            context_after_ids=[3],
            all_segment_ids=[1, 2, 3],
        ),
        segments_by_id=segments,
        options=TranslateOptions(model="m.gguf"),
    )
    rendered = "\n".join(message.content for message in messages)

    assert "translationUnitContract" in rendered
    assert '"targets":["u_002"]' in rendered
    assert '"context":["u_001","u_003"]' in rendered
    assert '"ownership"' in rendered
    assert "context is read-only evidence" in rendered
