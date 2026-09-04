from __future__ import annotations

from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationResult,
)
from movie_translator_worker.translation.quality import semantic_validate_result
from movie_translator_worker.translation.semantic_realization import (
    ContextualVietnameseRealizer,
    analyze_source_semantics,
    compact_semantic_payload,
    realization_critic_issues,
)


def _seg(
    segment_id: int,
    text: str,
    start: float = 0.0,
    end: float = 1.0,
    *,
    speaker_id: str | None = "speaker_001",
) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=end,
        speaker_id=speaker_id,
        speaker_confidence=0.91 if speaker_id else None,
    )


def _issues(source: str, translation: str) -> list[str]:
    rep = analyze_source_semantics(segment_id=1, source=source)
    return realization_critic_issues(
        source=source,
        translation=translation,
        representation=rep,
        pronoun_plan=None,
    )


def test_unknown_listener_keeps_sister_statement_semantic_safe() -> None:
    source = "她是我的姐姐。"
    rep = analyze_source_semantics(segment_id=1, source=source)

    assert rep.discourse_role == "possessive_relationship"
    assert rep.propositions[0]["type"] == "relationship_assertion"
    assert "speaker_listener_pronoun_pair_unresolved" in rep.unresolved
    assert _issues(source, "Cô ấy là chị gái của tôi.") == []


def test_unknown_listener_rejects_invented_em_for_sister_statement() -> None:
    issues = _issues("她是我的姐姐。", "Cô ấy là chị của em.")

    assert "UNSUPPORTED_PRONOUN_INFERENCE" in issues
    assert "RELATIONSHIP_REALIZATION_ERROR" in issues


def test_same_sister_term_has_different_discourse_roles() -> None:
    direct = analyze_source_semantics(segment_id=1, source="姐姐，你过来一下。")
    possessive = analyze_source_semantics(segment_id=2, source="我姐姐已经回来了。")
    subject = analyze_source_semantics(segment_id=3, source="姐姐来了。")

    assert direct.discourse_role == "direct_address"
    assert possessive.discourse_role == "possessive_relationship"
    assert subject.discourse_role == "subject_or_object_reference"


def test_direct_address_rejects_third_person_realization() -> None:
    issues = _issues("姐姐，你过来一下。", "Chị ấy, lại đây một chút.")

    assert "DIRECT_ADDRESS_ERROR" in issues


def test_first_person_unknown_context_prefers_toi() -> None:
    assert _issues("我不知道。", "Tôi không biết.") == []
    assert "UNSUPPORTED_PRONOUN_INFERENCE" in _issues("我不知道。", "Em không biết.")


def test_mother_possessive_relationship_preserves_possession() -> None:
    good = _issues("她是我妈。", "Bà ấy là mẹ của tôi.")
    bad = _issues("她是我妈。", "Bà ấy là mẹ.")

    assert good == []
    assert "POSSESSION_ERROR" in bad


def test_direct_address_to_mother_allows_address_without_invented_self() -> None:
    issues = _issues("妈，你听我说。", "Mẹ, nghe con nói.")

    assert "DIRECT_ADDRESS_ERROR" not in issues
    assert "RELATIONSHIP_INFORMATION_LOSS" not in issues


def test_boss_title_is_not_family_relationship() -> None:
    rep = analyze_source_semantics(segment_id=1, source="他是我的老板。")
    payload = compact_semantic_payload(rep)

    assert payload["terms"][0]["category"] == "ORGANIZATIONAL_ROLE"
    assert _issues("他是我的老板。", "Ông ấy là sếp của tôi.") == []
    assert "TITLE_ROLE_ERROR" in _issues("他是我的老板。", "Ông ấy là bố của tôi.")


def test_boss_direct_address_is_distinct_from_third_person_reference() -> None:
    direct = analyze_source_semantics(segment_id=1, source="老板，我有件事想说。")
    third = analyze_source_semantics(segment_id=2, source="他是我的老板。")

    assert direct.discourse_role == "direct_address"
    assert third.discourse_role == "possessive_relationship"


def test_aunt_and_uncle_keep_social_address_ambiguity() -> None:
    aunt = analyze_source_semantics(segment_id=1, source="阿姨，你好。")
    uncle = analyze_source_semantics(segment_id=2, source="叔叔来了。")

    assert aunt.terms[0].category == "SOCIAL_KINSHIP_ADDRESS"
    assert "direct_address_listener_unresolved" in aunt.unresolved
    assert uncle.terms[0].category == "SOCIAL_KINSHIP_ADDRESS"
    assert "relationship_sense_unresolved" in uncle.unresolved


def test_semantic_validator_records_phase8_metadata_and_flags() -> None:
    seg = _seg(1, "她是我的姐姐。")
    result = semantic_validate_result(
        segment_id=1,
        source=seg.source_text,
        result=TranslationResult("Cô ấy là chị của em."),
        memory=TranslationMemory.from_segments([seg]),
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert result.metadata.needs_review is True
    assert "UNSUPPORTED_PRONOUN_INFERENCE" in result.metadata.reason_flags
    assert result.metadata.validation["semanticRepresentation"]["discourseRole"] == "possessive_relationship"
    assert result.metadata.validation["ambiguityScore"] > 0


def test_relationship_memory_stores_facts_and_evidence_not_only_vietnamese_words() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "哥哥，你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我在听。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, "哥哥，别走。", 2.1, 3.0, speaker_id="speaker_001"),
            _seg(4, "好。", 3.1, 4.0, speaker_id="speaker_002"),
        ]
    )

    assert memory.movie_memory is not None
    row = memory.movie_memory.scene_relationship_overrides["scene_001"][0].to_dict()
    assert row["relationshipFact"]["relation"] == "younger_to_older_brother"
    assert row["relationshipFact"]["relation_domain"] == "family"
    assert row["relationshipFact"]["evidence_source"] == "source_dialogue"
    assert row["surfaceRealizationSuggestion"]["from_target"] == "anh"


def test_generated_translation_cannot_create_relationship_memory_fact() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我知道。", 1.1, 2.0, speaker_id="speaker_002"),
        ]
    )
    memory.record(1, "你听我说。", "Anh nghe em nói.")

    assert memory.movie_memory is not None
    assert memory.movie_memory.relationships == []
    assert memory.movie_memory.scene_relationship_overrides == {}


def test_current_source_evidence_overrides_conflicting_translation_style_memory() -> None:
    memory = TranslationMemory.from_segments([_seg(1, "她是我的姐姐。")])
    memory.record(0, "你听我说。", "Anh nghe em nói.")
    result = semantic_validate_result(
        segment_id=1,
        source="她是我的姐姐。",
        result=TranslationResult("Cô ấy là chị của em."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert "UNSUPPORTED_PRONOUN_INFERENCE" in result.metadata.reason_flags


def test_contextual_realizer_separates_sister_statement_from_direct_address() -> None:
    realizer = ContextualVietnameseRealizer()

    statement = realizer.realize(
        representation=analyze_source_semantics(
            segment_id=1,
            source="她是我的姐姐。",
        )
    ).to_dict()
    direct = realizer.realize(
        representation=analyze_source_semantics(
            segment_id=2,
            source="姐姐，你过来一下。",
        )
    ).to_dict()

    assert statement["translation"] == "Cô ấy là chị gái của tôi."
    assert direct["translation"] == "Chị, lại đây một chút."
    assert statement["realization_notes"] != direct["realization_notes"]
    assert statement["confidence"] >= 0.9
    assert direct["confidence"] >= 0.9


def test_contextual_realizer_repairs_bad_seed_relationship_pronoun() -> None:
    realizer = ContextualVietnameseRealizer()
    result = realizer.realize(
        representation=analyze_source_semantics(
            segment_id=1,
            source="她是我的姐姐。",
        ),
        seed_translation="Chị ấy là chị em.",
    )

    assert result.translation == "Cô ấy là chị gái của tôi."
    assert "self_repair_attempts:1" in result.realization_notes
    assert result.confidence >= 0.9


def test_contextual_realizer_handles_required_phase8_relationship_examples() -> None:
    realizer = ContextualVietnameseRealizer()
    examples = {
        "我是她妹妹。": "Tôi là em gái của cô ấy.",
        "我哥已经回来了。": "Anh trai của tôi đã về rồi.",
        "她是我妈。": "Bà ấy là mẹ của tôi.",
        "妈，你听我说。": "Mẹ, nghe tôi nói.",
        "他是我的老板。": "Ông ấy là sếp của tôi.",
        "老板，我有件事想说。": "Sếp, tôi có chuyện muốn nói.",
    }

    for source, expected in examples.items():
        result = realizer.realize(
            representation=analyze_source_semantics(segment_id=1, source=source)
        )
        assert result.translation == expected
        assert "seed_translation" not in result.realization_notes
