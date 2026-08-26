"""Lightweight local speaker diarization.

This is deliberately dependency-free. It is not a replacement for a
neural diarization model, but it gives the local-first pipeline stable
movie-scoped speaker ids from audio timing and simple acoustic features,
with conservative UNKNOWN output when evidence is weak.
"""

from __future__ import annotations

import audioop
import math
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .models import Segment

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    first_seen: float
    centroid: tuple[float, ...]
    segments: tuple[int, ...]
    confidence_sum: float

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(1, len(self.segments))


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[Segment]
    turns: list[SpeakerTurn]
    speaker_memory: dict[str, dict]


def diarize_segments(audio_path: str | Path, segments: Iterable[Segment]) -> DiarizationResult:
    source_segments = list(segments)
    if not source_segments:
        return DiarizationResult([], [], {})
    try:
        audio = _read_wav(audio_path)
    except Exception:
        labelled = [
            replace(seg, speaker_id=UNKNOWN_SPEAKER, speaker_confidence=0.0)
            for seg in source_segments
        ]
        return DiarizationResult(labelled, [], {})

    raw_features = [_segment_features(audio, seg.start, seg.end) for seg in source_segments]
    normalised = _normalise_features(raw_features)
    profiles: list[SpeakerProfile] = []
    labelled: list[Segment] = []

    for seg, raw, feature in zip(source_segments, raw_features, normalised):
        duration = max(0.0, seg.end - seg.start)
        if duration < 0.25 or raw[0] < 80:
            labelled.append(
                replace(seg, speaker_id=UNKNOWN_SPEAKER, speaker_confidence=0.0)
            )
            continue
        speaker_id, confidence, profiles = _assign_profile(
            profiles=profiles,
            feature=feature,
            segment_id=seg.id,
            start=seg.start,
        )
        if confidence < 0.70:
            labelled.append(
                replace(seg, speaker_id=UNKNOWN_SPEAKER, speaker_confidence=confidence)
            )
        else:
            labelled.append(
                replace(seg, speaker_id=speaker_id, speaker_confidence=confidence)
            )

    turns = merge_speaker_turns(labelled)
    memory = build_speaker_memory(labelled)
    return DiarizationResult(labelled, turns, memory)


def map_turns_to_segments(
    segments: Iterable[Segment],
    turns: Iterable[SpeakerTurn],
    *,
    min_confidence: float = 0.70,
) -> list[Segment]:
    mapped: list[Segment] = []
    turns_list = list(turns)
    for seg in segments:
        scores: dict[str, float] = {}
        total_overlap = 0.0
        for turn in turns_list:
            overlap = _overlap(seg.start, seg.end, turn.start, turn.end)
            if overlap <= 0:
                continue
            total_overlap += overlap
            scores[turn.speaker_id] = scores.get(turn.speaker_id, 0.0) + overlap * turn.confidence
        if not scores:
            mapped.append(replace(seg, speaker_id=UNKNOWN_SPEAKER, speaker_confidence=0.0))
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        confidence = best_score / max(0.001, seg.end - seg.start)
        if len(ranked) > 1:
            confidence *= max(0.0, min(1.0, (best_score - ranked[1][1]) / best_score))
        if total_overlap / max(0.001, seg.end - seg.start) < 0.25:
            confidence *= 0.5
        if confidence < min_confidence:
            mapped.append(
                replace(seg, speaker_id=UNKNOWN_SPEAKER, speaker_confidence=confidence)
            )
        else:
            mapped.append(replace(seg, speaker_id=best_id, speaker_confidence=confidence))
    return mapped


def merge_speaker_turns(segments: Iterable[Segment], *, max_gap: float = 0.65) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        speaker = seg.speaker_id or UNKNOWN_SPEAKER
        if speaker == UNKNOWN_SPEAKER:
            continue
        confidence = float(seg.speaker_confidence or 0.0)
        if (
            turns
            and turns[-1].speaker_id == speaker
            and seg.start - turns[-1].end <= max_gap
        ):
            prev = turns[-1]
            span_prev = max(0.001, prev.end - prev.start)
            span_new = max(0.001, seg.end - seg.start)
            weighted_conf = (
                prev.confidence * span_prev + confidence * span_new
            ) / (span_prev + span_new)
            turns[-1] = SpeakerTurn(
                speaker_id=speaker,
                start=prev.start,
                end=max(prev.end, seg.end),
                confidence=weighted_conf,
            )
        else:
            turns.append(
                SpeakerTurn(
                    speaker_id=speaker,
                    start=seg.start,
                    end=seg.end,
                    confidence=confidence,
                )
            )
    return turns


def build_speaker_memory(segments: Iterable[Segment]) -> dict[str, dict]:
    memory: dict[str, dict] = {}
    for seg in segments:
        speaker = seg.speaker_id or UNKNOWN_SPEAKER
        if speaker == UNKNOWN_SPEAKER:
            continue
        entry = memory.setdefault(
            speaker,
            {
                "segments": [],
                "firstSeen": seg.start,
                "confidence": 0.0,
                "_confidenceSum": 0.0,
            },
        )
        entry["segments"].append(f"segment_{seg.id}")
        entry["firstSeen"] = min(float(entry["firstSeen"]), seg.start)
        entry["_confidenceSum"] += float(seg.speaker_confidence or 0.0)
        entry["confidence"] = entry["_confidenceSum"] / max(1, len(entry["segments"]))
    for entry in memory.values():
        entry.pop("_confidenceSum", None)
        entry["confidence"] = round(float(entry["confidence"]), 4)
    return memory


def _assign_profile(
    *,
    profiles: list[SpeakerProfile],
    feature: tuple[float, ...],
    segment_id: int,
    start: float,
) -> tuple[str, float, list[SpeakerProfile]]:
    if not profiles:
        profile = SpeakerProfile("speaker_001", start, feature, (segment_id,), 0.95)
        return profile.speaker_id, 0.95, [profile]

    distances = [(_distance(feature, profile.centroid), idx) for idx, profile in enumerate(profiles)]
    distances.sort(key=lambda item: item[0])
    best_dist, best_idx = distances[0]
    second_dist = distances[1][0] if len(distances) > 1 else best_dist + 1.0
    threshold = 1.35

    if best_dist > threshold and len(profiles) < 12:
        speaker_id = f"speaker_{len(profiles) + 1:03d}"
        profile = SpeakerProfile(speaker_id, start, feature, (segment_id,), 0.9)
        return speaker_id, 0.9, profiles + [profile]

    confidence = max(0.0, min(0.98, 1.0 - best_dist / (threshold * 1.25)))
    if len(distances) > 1 and second_dist > 0:
        confidence *= max(0.0, min(1.0, (second_dist - best_dist) / second_dist + 0.35))

    updated = list(profiles)
    old = updated[best_idx]
    n = len(old.segments)
    centroid = tuple((old.centroid[i] * n + feature[i]) / (n + 1) for i in range(len(feature)))
    updated[best_idx] = SpeakerProfile(
        old.speaker_id,
        old.first_seen,
        centroid,
        old.segments + (segment_id,),
        old.confidence_sum + confidence,
    )
    return old.speaker_id, confidence, updated


def _read_wav(path: str | Path) -> tuple[int, list[int]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels > 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
    samples = [
        int.from_bytes(frames[i : i + 2], "little", signed=True)
        for i in range(0, len(frames), 2)
    ]
    return rate, samples


def _segment_features(audio: tuple[int, list[int]], start: float, end: float) -> tuple[float, ...]:
    rate, samples = audio
    lo = max(0, int(start * rate))
    hi = min(len(samples), max(lo + 1, int(end * rate)))
    chunk = samples[lo:hi]
    if not chunk:
        return (0.0, 0.0, 0.0, 0.0)
    rms = math.sqrt(sum(float(s) * float(s) for s in chunk) / len(chunk))
    zcr = sum(1 for a, b in zip(chunk, chunk[1:]) if (a < 0 <= b) or (a >= 0 > b)) / max(1, len(chunk) - 1)
    diff = sum(abs(b - a) for a, b in zip(chunk, chunk[1:])) / max(1, len(chunk) - 1)
    peak = max(abs(s) for s in chunk)
    return (rms, zcr, diff / 32768.0, peak / 32768.0)


def _normalise_features(features: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    if not features:
        return []
    transformed = [
        (math.log1p(f[0]), f[1] * 30.0, f[2] * 12.0, f[3])
        for f in features
    ]
    dims = len(transformed[0])
    means = [sum(f[i] for f in transformed) / len(transformed) for i in range(dims)]
    stds = [
        math.sqrt(sum((f[i] - means[i]) ** 2 for f in transformed) / len(transformed)) or 1.0
        for i in range(dims)
    ]
    return [
        tuple((f[i] - means[i]) / stds[i] for i in range(dims))
        for f in transformed
    ]


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
