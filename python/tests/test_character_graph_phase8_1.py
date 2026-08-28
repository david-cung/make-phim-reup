from __future__ import annotations

from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationResult,
)
from movie_translator_worker.translation.quality import semantic_validate_result


def _seg(
    segment_id: int,
    text: str,
    start: float,
    end: float,
    *,
    speaker_id: str,
) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=end,
        speaker_id=speaker_id,
        speaker_confidence=0.92,
    )


def _sibling_memory() -> TranslationMemory:
    return TranslationMemory.from_segments(
        [
            _seg(1, "哥，你去哪儿？", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我出去一下。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, "哥，什么时候回来？", 2.1, 3.0, speaker_id="speaker_001"),
            _seg(4, "很快。", 3.1, 4.0, speaker_id="speaker_002"),
            _seg(5, "那你早点回来。", 4.1, 5.0, speaker_id="speaker_001"),
        ]
    )


def test_character_graph_separates_relationship_fact_from_address_pattern() -> None:
    memory = _sibling_memory()

    assert memory.movie_memory is not None
    graph = memory.movie_memory.character_graph
    assert graph is not None
    fact = graph.relationship_facts[0].to_dict()
    pattern = graph.address_patterns[0].to_dict()

    assert fact["relationship"]["type"] == "younger_to_older_brother"
    assert fact["relationship"]["domain"] == "family"
    assert "preferred_realization" not in fact
    assert pattern["semantic_relationship"]["type"] == "younger_to_older_brother"
    assert pattern["preferred_realization"] == {"self": "em", "other": "anh"}
    assert pattern["source"] == "relationship_fact"


def test_relationship_graph_preserves_directionality() -> None:
    memory = _sibling_memory()

    assert memory.movie_memory is not None
    graph = memory.movie_memory.character_graph
    assert graph is not None
    rows = {
        (fact.from_character, fact.to_character): fact.relationship_type
        for fact in graph.relationship_facts
    }

    assert "younger_to_older_brother" in rows.values()
    assert "older_brother_to_younger" in rows.values()
    assert len(set(rows.values())) >= 2


def test_address_consistency_flags_unjustified_shift_for_verified_pair() -> None:
    memory = _sibling_memory()
    result = semantic_validate_result(
        segment_id=5,
        source="那你早点回来。",
        result=TranslationResult("Vậy chú về sớm nhé."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert "UNJUSTIFIED_ADDRESS_SHIFT" in result.metadata.reason_flags
    assert "ADDRESS_PAIR_INCONSISTENCY" in result.metadata.reason_flags
    assert result.metadata.validation["addressResolution"]["address_pair"] == ["em", "anh"]


def test_pronoun_omission_is_allowed_for_established_pair() -> None:
    memory = _sibling_memory()
    result = semantic_validate_result(
        segment_id=5,
        source="那你早点回来。",
        result=TranslationResult("Vậy về sớm nhé."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert "UNJUSTIFIED_ADDRESS_SHIFT" not in result.metadata.reason_flags
    assert "ADDRESS_PAIR_INCONSISTENCY" not in result.metadata.reason_flags


def test_intentional_address_shift_is_allowed_when_source_supports_it() -> None:
    memory = _sibling_memory()
    result = semantic_validate_result(
        segment_id=5,
        source="先生，你早点回来。",
        result=TranslationResult("Ông về sớm nhé."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert "UNJUSTIFIED_ADDRESS_SHIFT" not in result.metadata.reason_flags


def test_unknown_relationship_does_not_create_address_pattern_from_translation() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你去哪儿？", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我出去一下。", 1.1, 2.0, speaker_id="speaker_002"),
        ]
    )
    memory.record(1, "你去哪儿？", "Anh đi đâu vậy?")

    assert memory.movie_memory is not None
    graph = memory.movie_memory.character_graph
    assert graph is not None
    assert graph.relationship_facts == []
    assert graph.address_patterns == []


def test_translation_style_history_is_low_confidence_not_semantic_fact() -> None:
    memory = _sibling_memory()
    memory.record(5, "那你早点回来。", "Em bảo anh về sớm nhé.")

    assert memory.movie_memory is not None
    graph = memory.movie_memory.character_graph
    assert graph is not None
    key = next(iter(graph.recent_address_history))
    row = graph.recent_address_history[key][-1]
    assert row["source"] == "accepted_translation_style"
    assert row["confidence"] < 0.60
    assert all(fact.relationship_type != "accepted_translation_style" for fact in graph.relationship_facts)


def test_contradiction_state_records_conflicting_source_relationships() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "哥，你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我在听。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, "哥，别走。", 2.1, 3.0, speaker_id="speaker_001"),
            _seg(4, "好。", 3.1, 4.0, speaker_id="speaker_002"),
            _seg(5, "老板，请听我说。", 10.0, 11.0, speaker_id="speaker_001"),
            _seg(6, "你说。", 11.1, 12.0, speaker_id="speaker_002"),
            _seg(7, "老板，我会处理。", 12.1, 13.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.movie_memory is not None
    graph = memory.movie_memory.character_graph
    assert graph is not None
    assert graph.contradictions
    conflict = graph.contradictions[0].to_dict()
    assert conflict["conflict"] is True
    assert {
        conflict["existing_relation"],
        conflict["new_relation"],
    }.issubset({"younger_to_older_brother", "employee_to_boss"})


def test_relevant_payload_returns_only_graph_context_for_current_scene() -> None:
    memory = _sibling_memory()

    payload = memory.prompt_payload([1, 2, 3])
    graph_payload = payload["characterGraph"]

    assert graph_payload["relationshipFacts"]
    assert graph_payload["addressPatterns"]
    assert len(graph_payload["characters"]) <= 12
