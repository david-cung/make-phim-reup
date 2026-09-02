"""Recovery from partial LLM responses during translation.

Quantised GGUF models sometimes drop a line out of a batch. That used to
abort the whole translate job, discarding every chunk already done, so
the provider now re-asks for the stragglers before giving up.
"""

from __future__ import annotations

import json

import pytest

from movie_translator_worker.translation.llama_cpp_provider import (
    LlamaCppTranslationProvider,
)
from movie_translator_worker.translation.models import (
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
)


class _FakeCtx:
    def __init__(self, cancel_after: int | None = None) -> None:
        self.progress: list[tuple[float, str]] = []
        self.chunks: list[tuple[int, dict[int, str]]] = []
        self._cancel_after = cancel_after
        self._checks = 0

    def cancelled(self) -> bool:
        self._checks += 1
        if self._cancel_after is None:
            return False
        return self._checks > self._cancel_after

    def on_progress(self, fraction, stage, message=None):
        self.progress.append((fraction, stage))

    def on_chunk_completed(self, chunk_index, translations):
        self.chunks.append((chunk_index, dict(translations)))


class _ScriptedLlm:
    """Returns a canned payload per call, recording requested ids."""

    def __init__(self, replies: list[dict[int, str]]) -> None:
        self._replies = replies
        self.calls: list[list[int]] = []
        self.temperatures: list[float] = []

    def create_chat_completion(self, *, messages, temperature, **_kw):
        # The user turn embeds the rows to translate; recover the ids so
        # assertions can check what was re-asked.
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.calls.append(_ids_in(user))
        self.temperatures.append(temperature)
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        payload = {
            "segments": [
                {"id": sid, "translation": text}
                for sid, text in self._replies[idx].items()
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _ids_in(rendered: str) -> list[int]:
    found: list[int] = []
    for token in rendered.replace(",", " ").replace(":", " ").split():
        if token.strip('"').isdigit():
            value = int(token.strip('"'))
            if value not in found:
                found.append(value)
    return found


def _segments(ids):
    return {
        i: TranslatedSegment(
            id=i, source_text=f"line {i}", translation="", start=i, end=i + 1
        )
        for i in ids
    }


def _chunk(ids):
    return TranslationChunk(chunk_index=0, segment_ids=list(ids))


def _provider(tmp_path):
    return LlamaCppTranslationProvider(tmp_path)


def test_missing_segment_is_repaired_on_retry(tmp_path):
    ids = [58, 59, 60]
    # First reply drops 60 — the exact failure reported from the field.
    llm = _ScriptedLlm(
        [
            {58: "a", 59: "b"},
            {60: "c"},
        ]
    )
    ctx = _FakeCtx()
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=TranslateOptions(model="m.gguf"),
        ctx=ctx,
    )

    assert out == {58: "a", 59: "b", 60: "c"}
    # The retry must ask for the straggler only, not the whole batch.
    assert llm.calls[1] == [60]


def test_retry_raises_temperature(tmp_path):
    ids = [1, 2]
    llm = _ScriptedLlm([{1: "a"}, {2: "b"}])
    opts = TranslateOptions(model="m.gguf", temperature=0.2)
    _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=opts,
        ctx=_FakeCtx(),
    )
    assert llm.temperatures[0] == pytest.approx(0.2)
    assert llm.temperatures[1] > llm.temperatures[0]


def test_unparseable_reply_does_not_abort_immediately(tmp_path):
    ids = [7]

    class _Garbage(_ScriptedLlm):
        def create_chat_completion(self, *, messages, temperature, **_kw):
            self.calls.append([])
            if len(self.calls) == 1:
                return {"choices": [{"message": {"content": "sorry, I cannot"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"segments": [{"id": 7, "translation": "ok"}]}
                            )
                        }
                    }
                ]
            }

    out = _provider(tmp_path)._translate_one_chunk(
        llm=_Garbage([]),
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=TranslateOptions(model="m.gguf"),
        ctx=_FakeCtx(),
    )
    assert out == {7: "ok"}


def test_exhausted_retries_fall_back_instead_of_aborting_chunk(tmp_path):
    ids = [3]
    # Never returns the requested id, no matter how often it is asked.
    llm = _ScriptedLlm([{}])
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=TranslateOptions(model="m.gguf"),
        ctx=_FakeCtx(),
    )
    assert out[3].translation
    assert out[3].metadata.needs_review is True
    assert "LLM_OMITTED_SEGMENT" in out[3].metadata.reason_flags
    # Original attempt plus the repair rounds, rather than a single shot.
    assert len(llm.calls) > 1


def test_repair_reports_progress(tmp_path):
    ids = [1, 2, 3]
    llm = _ScriptedLlm([{1: "a"}, {2: "b", 3: "c"}])
    seen: list[tuple[float, str]] = []
    _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=TranslateOptions(model="m.gguf"),
        ctx=_FakeCtx(),
        label="chunk 2/9",
        report=lambda frac, msg: seen.append((frac, msg)),
    )
    assert seen, "repair must report progress; the bar froze otherwise"
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert "chunk 2/9" in seen[0][1]


def test_per_segment_repair_reports_each_segment(tmp_path):
    ids = [4, 5, 6]
    # Never satisfies the batch, forcing the one-request-per-segment round.
    llm = _ScriptedLlm([{}])
    seen: list[str] = []
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=_segments(ids),
        options=TranslateOptions(model="m.gguf"),
        ctx=_FakeCtx(),
        report=lambda frac, msg: seen.append(msg),
    )
    # The slowest path must narrate per segment, not go quiet.
    assert any("segment 4" in m for m in seen)
    assert any("segment 6 (3/3)" in m for m in seen)
    assert set(out) == {4, 5, 6}
    assert all(result.metadata.needs_review for result in out.values())


def test_context_ids_are_not_treated_as_missing(tmp_path):
    chunk = TranslationChunk(
        chunk_index=2,
        segment_ids=[10, 11],
        context_before_ids=[8, 9],
        context_after_ids=[12],
    )
    llm = _ScriptedLlm([{10: "x", 11: "y"}])
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=chunk,
        segments_by_id=_segments([8, 9, 10, 11, 12]),
        options=TranslateOptions(model="m.gguf"),
        ctx=_FakeCtx(),
    )
    assert out == {10: "x", 11: "y"}
    assert len(llm.calls) == 1


def test_vietnamese_target_retries_cjk_copy(tmp_path):
    ids = [1]
    segments = {
        1: TranslatedSegment(
            id=1,
            source_text="相 亲 结 婚 也 没 什么 难 的",
            translation="",
            start=1,
            end=2,
        )
    }
    llm = _ScriptedLlm(
        [
            {1: "相亲结婚也挺简单的嘛。"},
            {1: "Kết hôn qua mai mối cũng đơn giản mà."},
        ]
    )
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=segments,
        options=TranslateOptions(
            model="m.gguf",
            source_language="zh",
            target_language="vi",
            prompt_version="translation_prompt_v3",
        ),
        ctx=_FakeCtx(),
    )
    assert out == {1: "Kết hôn qua mai mối cũng đơn giản mà."}
    assert len(llm.calls) == 2


def test_vietnamese_target_retries_verbatim_copy(tmp_path):
    ids = [2]
    segments = {
        2: TranslatedSegment(
            id=2,
            source_text="Hello there",
            translation="",
            start=2,
            end=3,
        )
    }
    llm = _ScriptedLlm([{2: "Hello there"}, {2: "Xin chào."}])
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=segments,
        options=TranslateOptions(
            model="m.gguf",
            source_language="en",
            target_language="vi",
            prompt_version="translation_prompt_v3",
        ),
        ctx=_FakeCtx(),
    )
    assert out == {2: "Xin chào."}
    assert len(llm.calls) == 2


def test_final_rescue_prompt_handles_stubborn_segment(tmp_path):
    ids = [6]
    segments = {
        6: TranslatedSegment(
            id=6,
            source_text="沈 总 您 这 一 回 国 不 回 老 宅",
            translation="",
            start=6,
            end=7,
        )
    }
    bad = {6: "沈总，这次回国不回老宅。"}
    llm = _ScriptedLlm(
        [
            bad,
            bad,
            bad,
            bad,
            {6: "Tổng giám đốc Thẩm, lần này về nước mà không về nhà cũ."},
        ]
    )
    out = _provider(tmp_path)._translate_one_chunk(
        llm=llm,
        chunk=_chunk(ids),
        segments_by_id=segments,
        options=TranslateOptions(
            model="m.gguf",
            source_language="zh",
            target_language="vi",
            prompt_version="translation_prompt_v3",
        ),
        ctx=_FakeCtx(),
    )
    assert out == {
        6: "Tổng giám đốc Thẩm, lần này về nước mà không về nhà cũ."
    }
    assert len(llm.calls) == 5


def test_parser_accepts_prefixed_segment_ids() -> None:
    payload = json.dumps(
        {"translations": [{"id": "seg_00006", "translated_text": "Xin chào."}]}
    )

    out = LlamaCppTranslationProvider._parse_response(
        payload,
        expected_ids=[6],
    )

    assert out == {6: "Xin chào."}


def test_parser_wraps_single_top_level_translation_for_one_expected_id() -> None:
    payload = json.dumps({"translated_text": "Xin chào."})

    out = LlamaCppTranslationProvider._parse_response(
        payload,
        expected_ids=[6],
        strict=False,
    )

    assert out == {6: "Xin chào."}


def test_plain_text_rescue_wraps_translation_when_json_id_is_missing(tmp_path) -> None:
    class _PlainTextAfterJsonFailures(_ScriptedLlm):
        def create_chat_completion(self, *, messages, temperature, **kw):
            self.calls.append(_ids_in(next(m["content"] for m in reversed(messages) if m["role"] == "user")))
            self.temperatures.append(temperature)
            if "response_format" not in kw:
                return {
                    "choices": [
                        {"message": {"content": "Tổng giám đốc Thẩm về nước lần này."}}
                    ]
                }
            return {"choices": [{"message": {"content": json.dumps({"segments": []})}}]}

    ids = [6]
    segments = {
        6: TranslatedSegment(
            id=6,
            source_text="沈 总 您 这 一 回 国",
            translation="",
            start=6,
            end=7,
        )
    }
    out = _provider(tmp_path)._translate_one_chunk(
        llm=_PlainTextAfterJsonFailures([]),
        chunk=_chunk(ids),
        segments_by_id=segments,
        options=TranslateOptions(
            model="m.gguf",
            source_language="zh",
            target_language="vi",
        ),
        ctx=_FakeCtx(),
    )

    assert out[6].translation == "Tổng giám đốc Thẩm về nước lần này."
    assert out[6].metadata.translation_method == "single_text_rescue"
