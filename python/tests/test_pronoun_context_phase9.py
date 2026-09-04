from __future__ import annotations

from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import TranslatedSegment, TranslationResult
from movie_translator_worker.translation.quality import (
    PronounConsistencyValidator,
    global_consistency_issues,
)


def _seg(
    segment_id: int,
    text: str,
    speaker_id: str,
    *,
    start: float | None = None,
) -> TranslatedSegment:
    start_value = float(segment_id if start is None else start)
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start_value,
        end=start_value + 0.8,
        speaker_id=speaker_id,
        speaker_confidence=0.92,
    )


def test_case_a_female_to_male_pair_keeps_em_anh_mapping_stable() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "哥哥，你听我说。", "speaker_001"),
            _seg(2, "我在听。", "speaker_002"),
            _seg(3, "哥哥，别走。", "speaker_001"),
            _seg(4, "好。", "speaker_002"),
            _seg(5, "你为什么不告诉我？", "speaker_001"),
        ]
    )

    mapping = memory.pronoun_mapping_for_segment(5)
    assert mapping is not None
    assert mapping.self_pronoun == "em"
    assert mapping.listener_pronoun == "anh"
    assert mapping.confidence >= 0.80

    validator = PronounConsistencyValidator()
    assert validator.validate(
        segment_id=5,
        source="你为什么不告诉我？",
        translation="Sao anh không nói với em?",
        memory=memory,
    ) == []
    assert "ADDRESS_PAIR_INCONSISTENCY" in validator.validate(
        segment_id=5,
        source="你为什么不告诉我？",
        translation="Sao chị không nói với anh?",
        memory=memory,
    )


def test_case_b_male_to_female_pair_uses_pair_mapping_not_gender_rule() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "哥哥，你听我说。", "speaker_002"),
            _seg(2, "我知道。", "speaker_001"),
            _seg(3, "哥哥，等等。", "speaker_002"),
            _seg(4, "你别怕。", "speaker_001"),
        ]
    )

    mapping = memory.pronoun_mapping_for_segment(4)
    assert mapping is not None
    assert mapping.self_pronoun == "anh"
    assert mapping.listener_pronoun == "em"
    assert memory.address_debug_for_segment(4)["speaker_gender_hint"] == "male"

    assert PronounConsistencyValidator().validate(
        segment_id=4,
        source="你别怕。",
        translation="Em đừng sợ, có anh đây.",
        memory=memory,
    ) == []


def test_case_c_parent_child_context_allows_bo_con_not_anh_em() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "爸，你听我说。", "speaker_001"),
            _seg(2, "我在听。", "speaker_002"),
            _seg(3, "爸，等等。", "speaker_001"),
            _seg(4, "你说吧。", "speaker_002"),
        ]
    )

    mapping = memory.pronoun_mapping_for_segment(1)
    assert mapping is not None
    assert mapping.self_pronoun == "con"
    assert mapping.listener_pronoun == "bố"

    issues = PronounConsistencyValidator().validate(
        segment_id=1,
        source="爸，你听我说。",
        translation="Anh nghe em nói.",
        memory=memory,
    )
    assert "ADDRESS_PAIR_INCONSISTENCY" in issues


def test_case_d_unknown_relationship_stays_low_confidence_without_mapping() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你听我说。", "speaker_001"),
            _seg(2, "我在听。", "speaker_002"),
            _seg(3, "你知道吗？", "speaker_001"),
            _seg(4, "不知道。", "speaker_002"),
        ]
    )

    plan = memory.pronoun_plan_for_segment(1)
    assert plan is not None
    assert plan.confidence < 0.80
    assert memory.pronoun_mapping_for_segment(1) is None


def test_case_e_relationship_learned_later_updates_existing_low_confidence_turn() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你听我说。", "speaker_001"),
            _seg(2, "我在听。", "speaker_002"),
            _seg(3, "爸，我错了。", "speaker_001"),
            _seg(4, "没事。", "speaker_002"),
            _seg(5, "爸，等等。", "speaker_001"),
        ]
    )

    plan = memory.pronoun_plan_for_segment(1)
    assert plan is not None
    assert plan.relationship == "child_to_father"
    assert plan.confidence >= 0.80
    assert memory.pronoun_mapping_for_segment(1) is not None


def test_case_f_global_pass_flags_single_pronoun_outlier() -> None:
    segments = [
        _seg(1, "哥哥，你听我说。", "speaker_001"),
        _seg(2, "我在听。", "speaker_002"),
        _seg(3, "哥哥，别走。", "speaker_001"),
        _seg(4, "好。", "speaker_002"),
    ]
    for idx in range(5, 27):
        segments.append(_seg(idx, "你知道吗？", "speaker_001"))
    memory = TranslationMemory.from_segments(segments)
    translations = {
        sid: TranslationResult("Anh nghe em nói.")
        for sid in range(5, 26)
    }
    translations[26] = TranslationResult("Chị nghe anh nói.")
    by_id = {seg.id: seg for seg in segments}

    issues = global_consistency_issues(
        translations=translations,
        segments_by_id=by_id,
        memory=memory,
    )

    assert "PRONOUN_OUTLIER" in issues[26]


def test_case_g_multiple_speaker_pairs_do_not_leak_mappings() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "哥哥，你听我说。", "speaker_001", start=0),
            _seg(2, "我在听。", "speaker_002", start=1),
            _seg(3, "哥哥，别走。", "speaker_001", start=2),
            _seg(4, "好。", "speaker_002", start=3),
            _seg(10, "老板，我有件事想说。", "speaker_003", start=20),
            _seg(11, "你说。", "speaker_004", start=21),
            _seg(12, "老板，请等一下。", "speaker_003", start=22),
            _seg(13, "可以。", "speaker_004", start=23),
        ]
    )

    sibling = memory.pronoun_mapping_for_segment(3)
    workplace = memory.pronoun_mapping_for_segment(12)
    assert sibling is not None
    assert workplace is not None
    assert (sibling.self_pronoun, sibling.listener_pronoun) == ("em", "anh")
    assert (workplace.self_pronoun, workplace.listener_pronoun) == ("tôi", "sếp")

    validator = PronounConsistencyValidator()
    assert validator.validate(
        segment_id=12,
        source="老板，请等一下。",
        translation="Sếp, tôi có chuyện muốn nói.",
        memory=memory,
    ) == []
    assert "ADDRESS_PAIR_INCONSISTENCY" in validator.validate(
        segment_id=12,
        source="老板，请等一下。",
        translation="Anh, em có chuyện muốn nói.",
        memory=memory,
    )
