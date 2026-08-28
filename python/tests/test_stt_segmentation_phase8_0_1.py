from __future__ import annotations

from movie_translator_worker.stt.models import Segment
from movie_translator_worker.stt.segment import SegmenterSettings, resegment


def test_resegment_without_words_does_not_merge_across_speakers() -> None:
    segments = [
        Segment(1, 0.0, 0.4, "Nếu anh còn dám", speaker_id="speaker_001"),
        Segment(2, 0.45, 0.8, "thì sao?", speaker_id="speaker_002"),
    ]

    out = resegment(
        segments,
        SegmenterSettings(min_duration=0.2, max_duration=4.0, max_chars_per_line=42),
    )

    assert [seg.text for seg in out] == ["Nếu anh còn dám", "thì sao?"]
    assert [seg.speaker_id for seg in out] == ["speaker_001", "speaker_002"]


def test_resegment_without_words_keeps_same_speaker_merge_available() -> None:
    segments = [
        Segment(1, 0.0, 0.4, "Tôi chỉ muốn", speaker_id="speaker_001"),
        Segment(2, 0.45, 0.8, "nói với anh một chuyện.", speaker_id="speaker_001"),
    ]

    out = resegment(
        segments,
        SegmenterSettings(min_duration=0.2, max_duration=4.0, max_chars_per_line=42),
    )

    assert [seg.text for seg in out] == ["Tôi chỉ muốn nói với anh một chuyện."]
    assert out[0].speaker_id == "speaker_001"
