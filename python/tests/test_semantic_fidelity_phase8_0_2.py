from __future__ import annotations

import json

from movie_translator_worker.source_protection import (
    semantic_analysis,
    source_protection_payload,
    validate_translation_against_source,
)
from movie_translator_worker.translation.llama_cpp_provider import (
    LlamaCppTranslationProvider,
    _fallback_missing_translation,
)
from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationMetadata,
    TranslationResult,
)
from movie_translator_worker.translation.quality import (
    select_best_candidate,
    semantic_validate_result,
)
from movie_translator_worker.translation.semantic_realization import (
    analyze_source_semantics,
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
        self.temperatures: list[float] = []
        self.user_prompts: list[str] = []

    def create_chat_completion(self, *, messages, temperature, **_kw):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.calls.append(_ids_in_current(user))
        self.temperatures.append(float(temperature))
        self.user_prompts.append(user)
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


def _seg(segment_id: int, text: str) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=float(segment_id),
        end=float(segment_id) + 1.0,
        source_segment_id=f"seg_{segment_id:05d}",
        source_protection=source_protection_payload(segment_id=segment_id, text=text),
    )


def _issues(source: str, translation: str) -> list[str]:
    return validate_translation_against_source(
        source=source,
        translation=translation,
        protection=source_protection_payload(segment_id=1, text=source),
    )


def test_statement_semantic_representation_keeps_declarative_force() -> None:
    rep = analyze_source_semantics(segment_id=1, source="我们结婚了")

    assert rep.speech_act == "STATEMENT"
    assert rep.question_type is None
    assert rep.predicate == "marriage"
    assert "speech_act:STATEMENT" in rep.must_preserve


def test_statement_to_question_is_hard_rejected() -> None:
    issues = _issues("我们结婚了", "Chúng tôi đã kết hôn rồi à?")

    assert "STATEMENT_TO_QUESTION_ERROR" in issues
    assert "UNSUPPORTED_PARTICLE_INSERTION" in issues
    assert _issues("我们结婚了", "Chúng tôi đã kết hôn.") == []


def test_why_question_type_must_not_become_confirmation_question() -> None:
    assert semantic_analysis("你怎么没去啊")["questionType"] == "WHY"
    assert _issues("你怎么没去啊", "Sao em không đi?") == []

    issues = _issues("你怎么没去啊", "Em không đi à?")

    assert "QUESTION_TYPE_ERROR" in issues


def test_numeric_anchor_preserves_value_and_unit() -> None:
    assert _issues("他等你3个小时了", "Anh ấy đợi em ba tiếng rồi.") == []

    wrong = _issues("他等你3个小时了", "Anh ấy đợi em bảy tiếng rồi.")

    assert "NUMERIC_FIDELITY_ERROR" in wrong
    assert "NUMBER_MISMATCH" in wrong


def test_negated_predicate_cannot_become_unrelated_instruction() -> None:
    source = "相亲结婚也没什么难的吧"
    good = "Chuyện xem mắt rồi kết hôn cũng đâu có khó gì."
    bad = "Chịu khó lấy chồng đi."

    assert semantic_analysis(source)["speechAct"] == "STATEMENT"
    assert _issues(source, good) == []

    issues = _issues(source, bad)

    assert "POLARITY_ERROR" in issues
    assert "PREDICATE_ERROR" in issues
    assert "SEMANTIC_ADDITION" in issues


def test_translator_commentary_is_rejected() -> None:
    issues = _issues("我们结婚了", "Đoạn này cần dịch lại.")

    assert "TRANSLATOR_COMMENTARY_LEAK" in issues


def test_chinese_modal_particles_are_not_fixed_vietnamese_particles() -> None:
    source = "相亲结婚也没什么难的吧"
    rep = analyze_source_semantics(segment_id=1, source=source)

    assert rep.source_particles[0]["source"] == "吧"
    assert "UNSUPPORTED_PARTICLE_INSERTION" in _issues(
        source,
        "Xem mắt rồi kết hôn cũng không khó nhỉ?",
    )
    assert _issues(source, "Xem mắt rồi kết hôn cũng không có gì khó.") == []


def test_naturalizer_semantic_delta_rejects_added_question_particle() -> None:
    memory = TranslationMemory.from_segments([_seg(1, "我们结婚了")])
    result = semantic_validate_result(
        segment_id=1,
        source="我们结婚了",
        result=TranslationResult("Chúng tôi đã kết hôn rồi à?"),
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
        context_before=[],
        context_after=[],
    )

    assert result.metadata.needs_review is True
    assert "STATEMENT_TO_QUESTION_ERROR" in result.metadata.reason_flags
    assert "STATEMENT_TO_QUESTION_ERROR" in result.metadata.validation["semanticCritic"]["hardFailures"]


def test_candidate_selection_prefers_literal_faithful_over_fluent_wrong_mood() -> None:
    seg = _seg(1, "我们结婚了")
    memory = TranslationMemory.from_segments([seg])

    best = select_best_candidate(
        segment_id=1,
        source=seg.source_text,
        candidates=[
            TranslationResult("Chúng tôi đã kết hôn rồi à?", TranslationMetadata(confidence=0.98)),
            TranslationResult("Chúng tôi đã kết hôn.", TranslationMetadata(confidence=0.88)),
        ],
        memory=memory,
        options=TranslateOptions(model="m.gguf"),
    )

    assert best is not None
    assert best.translation == "Chúng tôi đã kết hôn."


def test_semantic_repair_uses_lower_temperature_and_source_anchors(tmp_path) -> None:
    seg = _seg(1, "我们结婚了")
    llm = _ScriptedLlm(
        [
            {1: "Chúng tôi đã kết hôn rồi à?"},
            {1: "Chúng tôi đã kết hôn."},
        ]
    )

    out = LlamaCppTranslationProvider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=TranslationChunk(chunk_index=0, segment_ids=[1], all_segment_ids=[1]),
        segments_by_id={1: seg},
        options=TranslateOptions(model="m.gguf", source_language="zh", target_language="vi", temperature=0.2),
        ctx=_Ctx(),
    )

    assert out[1].translation == "Chúng tôi đã kết hôn."
    assert [round(value, 2) for value in llm.temperatures] == [0.2, 0.15]
    assert "Semantic anchors by id" in llm.user_prompts[1]
    assert "speechAct" in llm.user_prompts[1]


def test_missing_segment_fallback_does_not_emit_translator_commentary() -> None:
    result = _fallback_missing_translation(
        segment_id=1,
        seg=_seg(1, "我们结婚了"),
        target_language="vi",
    )

    assert result.translation == "Chúng tôi đã kết hôn."
    assert "TRANSLATOR_COMMENTARY_LEAK" not in _issues("我们结婚了", result.translation)
