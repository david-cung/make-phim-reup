from __future__ import annotations

import json

from movie_translator_worker.translation.llama_cpp_provider import LlamaCppTranslationProvider
from movie_translator_worker.translation.memory import (
    PRONOUN_PLAN_ENFORCE_THRESHOLD,
    TranslationMemory,
)
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationResult,
)
from movie_translator_worker.translation.quality import validate_result


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


def _memory_for(term: str) -> TranslationMemory:
    return TranslationMemory.from_segments(
        [
            _seg(1, f"{term}，你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我在听。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, f"{term}，别走。", 2.1, 3.0, speaker_id="speaker_001"),
            _seg(4, "好。", 3.1, 4.0, speaker_id="speaker_002"),
        ]
    )


def _plan_for_first_line(term: str):
    memory = _memory_for(term)
    assert memory.movie_memory is not None
    return memory.movie_memory.pronoun_plans[1]


def test_em_anh_plan_from_older_brother_address() -> None:
    plan = _plan_for_first_line("哥哥")

    assert plan.self_pronoun == "em"
    assert plan.target_pronoun == "anh"
    assert plan.confidence >= PRONOUN_PLAN_ENFORCE_THRESHOLD


def test_em_chi_plan_from_older_sister_address() -> None:
    plan = _plan_for_first_line("姐姐")

    assert plan.self_pronoun == "em"
    assert plan.target_pronoun == "chị"


def test_con_me_plan_from_mother_address() -> None:
    plan = _plan_for_first_line("妈妈")

    assert plan.self_pronoun == "con"
    assert plan.target_pronoun == "mẹ"


def test_con_bo_plan_from_father_address() -> None:
    plan = _plan_for_first_line("爸爸")

    assert plan.self_pronoun == "con"
    assert plan.target_pronoun == "bố"


def test_chau_ong_plan_from_grandfather_address() -> None:
    plan = _plan_for_first_line("爷爷")

    assert plan.self_pronoun == "cháu"
    assert plan.target_pronoun == "ông"


def test_chau_ba_plan_from_grandmother_address() -> None:
    plan = _plan_for_first_line("奶奶")

    assert plan.self_pronoun == "cháu"
    assert plan.target_pronoun == "bà"


def test_employee_boss_plan_from_title() -> None:
    plan = _plan_for_first_line("老板")

    assert plan.relationship == "employee_to_boss"
    assert plan.self_pronoun == "tôi"
    assert plan.target_pronoun == "sếp"


def test_student_teacher_plan_from_title() -> None:
    plan = _plan_for_first_line("老师")

    assert plan.relationship == "student_to_teacher"
    assert plan.self_pronoun == "em"
    assert plan.target_pronoun == "thầy/cô"


def test_low_confidence_unknown_listener_does_not_force_pronouns() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "我知道。", 1.1, 2.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.movie_memory is not None
    plan = memory.movie_memory.pronoun_plans[1]
    assert plan.listener is None
    assert plan.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD
    assert plan.to_dict()["context_request"] is True


def test_scene_level_relationship_override_can_change_pronouns() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "老板，请听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "你说。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, "老板，我会处理。", 2.1, 3.0, speaker_id="speaker_001"),
            _seg(4, "妈妈，我回来了。", 10.0, 11.0, speaker_id="speaker_001"),
            _seg(5, "回来了？", 11.1, 12.0, speaker_id="speaker_002"),
            _seg(6, "妈妈，我饿了。", 12.1, 13.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.movie_memory is not None
    office = memory.movie_memory.pronoun_plans[1]
    home = memory.movie_memory.pronoun_plans[4]
    assert office.target_pronoun == "sếp"
    assert home.self_pronoun == "con"
    assert home.target_pronoun == "mẹ"
    office_relationship = memory.movie_memory.scene_relationship_overrides["scene_001"][0]
    home_relationship = memory.movie_memory.scene_relationship_overrides["scene_002"][0]
    assert office_relationship.relationship == "employee_to_boss"
    assert home_relationship.relationship == "child_to_mother"


def test_multiple_people_in_one_conversation_stays_uncertain() -> None:
    memory = TranslationMemory.from_segments(
        [
            _seg(1, "你们都听我说。", 0.0, 1.0, speaker_id="speaker_001"),
            _seg(2, "嗯。", 1.1, 2.0, speaker_id="speaker_002"),
            _seg(3, "什么？", 2.1, 3.0, speaker_id="speaker_003"),
            _seg(4, "我再说一次。", 3.1, 4.0, speaker_id="speaker_001"),
        ]
    )

    assert memory.movie_memory is not None
    plan = memory.movie_memory.pronoun_plans[1]
    assert plan.listener is None
    assert plan.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD


def test_prompt_rows_include_automatic_pronoun_plan(tmp_path) -> None:
    memory = _memory_for("哥哥")
    segments = {
        1: _seg(1, "哥哥，你听我说。", 0.0, 1.0, speaker_id="speaker_001"),
        2: _seg(2, "我在听。", 1.1, 2.0, speaker_id="speaker_002"),
    }
    messages = LlamaCppTranslationProvider(tmp_path)._build_messages(
        chunk=TranslationChunk(chunk_index=0, segment_ids=[1], context_after_ids=[2]),
        segments_by_id=segments,
        options=TranslateOptions(model="m.gguf"),
        memory=memory,
    )

    rendered = "\n".join(message.content for message in messages)
    assert "automaticPronounPlan" in rendered
    assert '"self_pronoun":"em"' in rendered
    assert '"target_pronoun":"anh"' in rendered


def test_validator_flags_high_confidence_wrong_pronoun() -> None:
    memory = _memory_for("妈妈")
    out = validate_result(
        segment_id=1,
        source="妈妈，你听我说。",
        result=TranslationResult("Anh nghe em nói này."),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
    )

    assert "POSSIBLE_PRONOUN_INCONSISTENCY" in out.metadata.reason_flags
    assert out.metadata.needs_review is True


def test_relationship_memory_payload_is_relevant_only() -> None:
    memory = _memory_for("哥哥")

    payload = memory.prompt_payload([1, 2])
    json_payload = json.dumps(payload, ensure_ascii=False)

    assert "relationships" in payload
    assert "pronounPlans" in payload
    assert "younger_to_older_brother" in json_payload
