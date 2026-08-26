from __future__ import annotations

from movie_translator_worker.translation.memory import (
    TranslationMemory,
    build_movie_memory,
    extract_name_mentions,
)
from movie_translator_worker.translation.models import TranslatedSegment


def _seg(
    segment_id: int,
    text: str,
    start: float,
    end: float,
    *,
    speaker_id: str | None = None,
    speaker_confidence: float | None = 0.9,
) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=end,
        speaker_id=speaker_id,
        speaker_confidence=speaker_confidence,
    )


def test_extracts_names_aliases_titles_and_address_terms() -> None:
    mentions = extract_name_mentions("陈浩，浩哥，陈总说哥会帮我们。")

    assert "陈浩" in mentions
    assert "浩哥" in mentions
    assert "陈总" in mentions
    assert "哥" in mentions


def test_known_names_deduplicate_and_count_aliases() -> None:
    memory = build_movie_memory(
        [
            _seg(1, "陈浩来了。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "陈浩，你听见了吗？", 1.2, 2.0, speaker_id="speaker_002"),
            _seg(3, "浩哥在外面。", 2.2, 3.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.known_names["陈浩"]["count"] == 2
    assert memory.known_names["浩哥"]["count"] == 1


def test_uncertain_single_speaker_without_names_is_not_mapped() -> None:
    memory = build_movie_memory(
        [_seg(1, "快走。", 0.0, 1.0, speaker_id="speaker_001")]
    )

    assert memory.characters == []
    assert memory.speaker_character_mapping == {}


def test_repeated_speaker_with_nearby_name_gets_stable_character() -> None:
    memory = build_movie_memory(
        [
            _seg(1, "陈浩，你回来。", 0.0, 1.0, speaker_id="speaker_002"),
            _seg(2, "我知道了。", 1.2, 2.0, speaker_id="speaker_001"),
            _seg(3, "浩哥，等等我。", 2.2, 3.0, speaker_id="speaker_002"),
            _seg(4, "别跟着我。", 3.2, 4.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.characters[0].id == "character_001"
    assert "speaker_002" in memory.speaker_character_mapping
    assert "陈浩" in memory.characters[0].source_names
    assert memory.characters[0].confidence >= 0.70


def test_scene_memory_uses_time_gaps_and_participants() -> None:
    memory = build_movie_memory(
        [
            _seg(1, "陈浩，你回来。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我马上来。", 1.2, 2.0, speaker_id="speaker_001"),
            _seg(3, "老板在等。", 10.0, 11.0, speaker_id="speaker_002"),
            _seg(4, "别告诉陈浩。", 11.2, 12.0, speaker_id="speaker_002"),
        ]
    )

    assert [scene.scene_id for scene in memory.scenes] == ["scene_001", "scene_002"]
    assert memory.scenes[0].segments == [1, 2]
    assert memory.scenes[1].segments == [3, 4]
    assert memory.scenes[0].participants


def test_movie_memory_is_isolated_per_translation_memory() -> None:
    first = TranslationMemory.from_segments(
        [
            _seg(1, "陈浩来了。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "陈浩别走。", 1.2, 2.0, speaker_id="speaker_001"),
        ]
    )
    second = TranslationMemory.from_segments(
        [
            _seg(1, "老板来了。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "老板别走。", 1.2, 2.0, speaker_id="speaker_001"),
        ]
    )

    assert first.movie_memory is not None
    assert second.movie_memory is not None
    assert "陈浩" in first.movie_memory.known_names
    assert "陈浩" not in second.movie_memory.known_names
    assert "老板" in second.movie_memory.known_names


def test_prompt_payload_returns_relevant_movie_memory_only() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "陈浩，你回来。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我马上来。", 1.2, 2.0, speaker_id="speaker_001"),
            _seg(3, "老板在等。", 10.0, 11.0, speaker_id="speaker_002"),
            _seg(4, "别告诉老板。", 11.2, 12.0, speaker_id="speaker_002"),
        ]
    )

    payload = memory.prompt_payload([1, 2])

    assert payload["movieSummary"]
    assert [scene["scene_id"] for scene in payload["currentScenes"]] == ["scene_001"]
    assert all("segment_3" not in scene["segments"] for scene in payload["currentScenes"])
    assert "characters" in payload
    assert "translationMemory" in payload
