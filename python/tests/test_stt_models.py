"""Unit tests for the STT dataclasses + JSON schema + cache key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.stt.models import (  # noqa: E402
    TRANSCRIPT_SCHEMA_VERSION,
    Segment,
    TranscribeOptions,
    Transcript,
    Word,
    build_cache_key,
    transcript_from_dict,
    transcript_to_dict,
    validate_segments,
)


def _sample_transcript(word_ts: bool = False) -> Transcript:
    segments = [
        Segment(id=0, start=0.0, end=1.5, text="Hello."),
        Segment(id=1, start=1.5, end=3.0, text="World.",
                words=[Word("World", 1.5, 3.0, probability=0.9)] if word_ts else None),
    ]
    return Transcript(
        version=TRANSCRIPT_SCHEMA_VERSION,
        language="en",
        segments=segments,
        model="small",
        device="cpu",
        compute_type="int8",
        word_timestamps=word_ts,
        audio_hash="sha256:deadbeef",
        audio_path="audio/original.wav",
        duration_secs=3.0,
        cache_key="sha256:cafe",
        created_at="2026-08-11T09:00:00Z",
    )


def test_transcript_roundtrip_without_words() -> None:
    t = _sample_transcript(word_ts=False)
    payload = transcript_to_dict(t)
    assert payload["version"] == TRANSCRIPT_SCHEMA_VERSION
    assert payload["language"] == "en"
    assert payload["segments"][0] == {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello."}
    assert "words" not in payload["segments"][0]

    back = transcript_from_dict(payload)
    assert back.language == "en"
    assert len(back.segments) == 2
    assert back.segments[1].text == "World."
    assert back.segments[1].words is None


def test_transcript_roundtrip_with_words() -> None:
    t = _sample_transcript(word_ts=True)
    payload = transcript_to_dict(t)
    words = payload["segments"][1]["words"]
    assert words == [{"word": "World", "start": 1.5, "end": 3.0, "probability": 0.9}]

    back = transcript_from_dict(payload)
    assert back.segments[1].words is not None
    assert back.segments[1].words[0].word == "World"


def test_transcript_rejects_unknown_schema_version() -> None:
    payload = transcript_to_dict(_sample_transcript())
    payload["version"] = 999
    with pytest.raises(ValueError):
        transcript_from_dict(payload)


def test_validate_segments_rejects_negative_time() -> None:
    with pytest.raises(ValueError):
        validate_segments([Segment(id=0, start=-1.0, end=1.0, text="")])


def test_validate_segments_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        validate_segments([Segment(id=0, start=2.0, end=1.0, text="")])


def test_validate_segments_rejects_out_of_order() -> None:
    segs = [
        Segment(id=0, start=1.0, end=2.0, text=""),
        Segment(id=1, start=0.5, end=1.0, text=""),
    ]
    with pytest.raises(ValueError):
        validate_segments(segs)


def test_validate_segments_accepts_touching_segments() -> None:
    segs = [
        Segment(id=0, start=0.0, end=1.0, text=""),
        Segment(id=1, start=1.0, end=2.0, text=""),
    ]
    validate_segments(segs)


def test_validate_segments_word_end_before_start_rejected() -> None:
    segs = [
        Segment(
            id=0, start=0.0, end=2.0, text="hi",
            words=[Word("hi", 1.0, 0.5)],
        )
    ]
    with pytest.raises(ValueError):
        validate_segments(segs)


def test_cache_key_stable_and_deps_matter() -> None:
    opts_a = TranscribeOptions(model="small", language="en")
    opts_b = TranscribeOptions(model="small", language="en")
    opts_c = TranscribeOptions(model="small", language="ja")
    opts_d = TranscribeOptions(model="medium", language="en")

    k_a = build_cache_key("sha256:deadbeef", opts_a)
    k_b = build_cache_key("sha256:deadbeef", opts_b)
    k_c = build_cache_key("sha256:deadbeef", opts_c)
    k_d = build_cache_key("sha256:deadbeef", opts_d)
    k_e = build_cache_key("sha256:otherhash", opts_a)

    assert k_a == k_b
    assert k_a != k_c
    assert k_a != k_d
    assert k_a != k_e
    assert k_a.startswith("sha256:")


def test_cache_key_language_none_matches_auto_string() -> None:
    a = build_cache_key("h", TranscribeOptions(language=None))
    b = build_cache_key("h", TranscribeOptions(language="Auto"))
    assert a == b


def test_transcript_to_dict_rounds_to_3_decimals() -> None:
    t = _sample_transcript()
    t.segments[0] = Segment(id=0, start=0.123456, end=1.234567, text="hi")
    payload = transcript_to_dict(t)
    assert payload["segments"][0]["start"] == 0.123
    assert payload["segments"][0]["end"] == 1.235
