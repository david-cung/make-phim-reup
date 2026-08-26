from __future__ import annotations

import math
import wave
from pathlib import Path

from movie_translator_worker.stt.diarization import (
    UNKNOWN_SPEAKER,
    SpeakerTurn,
    build_speaker_memory,
    diarize_segments,
    map_turns_to_segments,
    merge_speaker_turns,
)
from movie_translator_worker.stt.models import Segment
from movie_translator_worker.stt.models import transcript_from_dict


def _seg(segment_id: int, start: float, end: float) -> Segment:
    return Segment(id=segment_id, start=start, end=end, text=f"line {segment_id}")


def test_maps_turns_to_segments_by_overlap() -> None:
    mapped = map_turns_to_segments(
        [_seg(1, 10.0, 12.0)],
        [SpeakerTurn("speaker_001", 9.0, 13.0, 0.95)],
    )

    assert mapped[0].speaker_id == "speaker_001"
    assert mapped[0].speaker_confidence and mapped[0].speaker_confidence > 0.9


def test_overlap_ambiguity_returns_unknown() -> None:
    mapped = map_turns_to_segments(
        [_seg(1, 10.0, 12.0)],
        [
            SpeakerTurn("speaker_001", 10.0, 11.0, 0.9),
            SpeakerTurn("speaker_002", 11.0, 12.0, 0.9),
        ],
    )

    assert mapped[0].speaker_id == UNKNOWN_SPEAKER
    assert mapped[0].speaker_confidence is not None
    assert mapped[0].speaker_confidence < 0.7


def test_unknown_when_no_turn_overlap() -> None:
    mapped = map_turns_to_segments(
        [_seg(1, 30.0, 31.0)],
        [SpeakerTurn("speaker_001", 10.0, 12.0, 0.95)],
    )

    assert mapped[0].speaker_id == UNKNOWN_SPEAKER
    assert mapped[0].speaker_confidence == 0.0


def test_merge_consecutive_same_speaker_turns() -> None:
    merged = merge_speaker_turns(
        [
            Segment(1, 0.0, 1.0, "a", speaker_id="speaker_001", speaker_confidence=0.9),
            Segment(2, 1.2, 2.0, "b", speaker_id="speaker_001", speaker_confidence=0.8),
            Segment(3, 3.0, 4.0, "c", speaker_id="speaker_002", speaker_confidence=0.9),
        ]
    )

    assert [(t.speaker_id, round(t.start, 1), round(t.end, 1)) for t in merged] == [
        ("speaker_001", 0.0, 2.0),
        ("speaker_002", 3.0, 4.0),
    ]


def test_speaker_memory_is_movie_scoped() -> None:
    memory = build_speaker_memory(
        [
            Segment(1, 2.0, 3.0, "a", speaker_id="speaker_001", speaker_confidence=0.9),
            Segment(5, 5.0, 6.0, "b", speaker_id="speaker_001", speaker_confidence=0.8),
        ]
    )

    assert memory["speaker_001"]["segments"] == ["segment_1", "segment_5"]
    assert memory["speaker_001"]["firstSeen"] == 2.0


def test_diarize_segments_assigns_stable_speaker_ids(tmp_path: Path) -> None:
    wav_path = tmp_path / "speakers.wav"
    _write_tone_wav(wav_path, [(220.0, 1.0), (880.0, 1.0), (220.0, 1.0)])

    result = diarize_segments(
        wav_path,
        [_seg(1, 0.0, 1.0), _seg(2, 1.0, 2.0), _seg(3, 2.0, 3.0)],
    )

    ids = [segment.speaker_id for segment in result.segments]
    assert ids[0] == ids[2]
    assert ids[0] != ids[1]
    assert ids[0] == "speaker_001"
    assert ids[1] == "speaker_002"
    assert "speaker_001" in result.speaker_memory


def test_too_short_audio_segment_is_unknown(tmp_path: Path) -> None:
    wav_path = tmp_path / "short.wav"
    _write_tone_wav(wav_path, [(220.0, 0.5)])

    result = diarize_segments(wav_path, [_seg(1, 0.0, 0.1)])

    assert result.segments[0].speaker_id == UNKNOWN_SPEAKER
    assert result.segments[0].speaker_confidence == 0.0


def test_old_transcript_without_speaker_fields_still_loads() -> None:
    transcript = transcript_from_dict(
        {
            "version": 1,
            "language": "en",
            "segments": [{"id": 1, "start": 0.0, "end": 1.0, "text": "hello"}],
            "model": "small",
            "device": "cpu",
            "computeType": "int8",
            "wordTimestamps": False,
            "audio": {"path": "audio/original.wav", "hash": "sha256:a"},
            "durationSecs": 1.0,
            "cacheKey": "sha256:b",
            "createdAt": "2026-01-01T00:00:00Z",
        }
    )

    assert transcript.segments[0].speaker_id is None
    assert transcript.segments[0].speaker_confidence is None
    assert transcript.speaker_memory == {}


def _write_tone_wav(path: Path, tones: list[tuple[float, float]]) -> None:
    rate = 16_000
    samples: list[int] = []
    for frequency, duration in tones:
        count = int(rate * duration)
        for n in range(count):
            value = int(12000 * math.sin(2 * math.pi * frequency * (n / rate)))
            samples.append(value)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
