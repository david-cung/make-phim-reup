"""Transport-shaped dataclasses for STT output + input.

Kept intentionally free of any faster-whisper types so this module can
be imported without the heavy runtime dependency and reused verbatim
by tests that use a fake provider.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

TRANSCRIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Word:
    """A single word timestamp emitted by the provider."""

    word: str
    start: float
    end: float
    probability: Optional[float] = None


@dataclass(frozen=True)
class Segment:
    """A single Whisper segment. These are the canonical subtitle timings."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    words: Optional[list[Word]] = None
    speaker_id: Optional[str] = None
    speaker_confidence: Optional[float] = None


@dataclass
class TranscribeOptions:
    """Every parameter that materially affects the produced transcript.

    Anything that is part of the cache key must live here so cache
    invalidation is trivial (a hash of the options + audio fingerprint).
    """

    model: str = "medium"
    language: Optional[str] = None  # None => auto detect
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    initial_prompt: Optional[str] = None
    temperature: float = 0.0
    quality_profile: Optional[str] = None
    resegment: bool = True

    def normalised_language(self) -> str:
        return (self.language or "auto").lower()


@dataclass
class Transcript:
    """The persisted result for a completed transcription.

    Written to ``<project>/transcription/transcription.json``. The
    ``cache_key`` field lets us skip re-running the provider when the
    inputs have not changed.
    """

    version: int
    language: str
    segments: list[Segment]
    model: str
    device: str
    compute_type: str
    word_timestamps: bool
    audio_hash: str
    audio_path: str
    duration_secs: float
    cache_key: str
    created_at: str
    provider: str = "faster-whisper"
    options: dict[str, Any] = field(default_factory=dict)
    speaker_memory: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- codec


def transcript_to_dict(t: Transcript) -> dict[str, Any]:
    """Serialise :class:`Transcript` to a JSON-friendly ``dict`` using
    the on-disk camelCase schema."""

    return {
        "version": t.version,
        "language": t.language,
        "segments": [_segment_to_dict(s) for s in t.segments],
        "model": t.model,
        "device": t.device,
        "computeType": t.compute_type,
        "wordTimestamps": t.word_timestamps,
        "audio": {"path": t.audio_path, "hash": t.audio_hash},
        "durationSecs": t.duration_secs,
        "cacheKey": t.cache_key,
        "createdAt": t.created_at,
        "provider": t.provider,
        "options": t.options,
        "speakerMemory": t.speaker_memory,
    }


def transcript_from_dict(payload: dict[str, Any]) -> Transcript:
    """Inverse of :func:`transcript_to_dict`; raises ``ValueError`` on
    schema violations."""

    if not isinstance(payload, dict):
        raise ValueError("transcript must be a JSON object")
    version = payload.get("version")
    if version != TRANSCRIPT_SCHEMA_VERSION:
        raise ValueError(f"unsupported transcript schema version: {version}")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")
    segments = [_segment_from_dict(s) for s in raw_segments]
    validate_segments(segments)
    audio = payload.get("audio") or {}
    if not isinstance(audio, dict):
        raise ValueError("audio must be an object")
    return Transcript(
        version=version,
        language=str(payload.get("language") or ""),
        segments=segments,
        model=str(payload.get("model") or ""),
        device=str(payload.get("device") or ""),
        compute_type=str(payload.get("computeType") or ""),
        word_timestamps=bool(payload.get("wordTimestamps") or False),
        audio_hash=str(audio.get("hash") or ""),
        audio_path=str(audio.get("path") or ""),
        duration_secs=float(payload.get("durationSecs") or 0.0),
        cache_key=str(payload.get("cacheKey") or ""),
        created_at=str(payload.get("createdAt") or ""),
        provider=str(payload.get("provider") or "faster-whisper"),
        options=dict(payload.get("options") or {}),
        speaker_memory=dict(payload.get("speakerMemory") or {}),
    )


def validate_segments(segments: Iterable[Segment]) -> None:
    """Raise ``ValueError`` if segment timings are not monotonically
    non-decreasing or if any segment is malformed.

    We *do not* require gap-free segments — Whisper sometimes leaves a
    tiny silence between adjacent segments — but we do require:

      * ``start <= end`` for every segment.
      * ``segments[i].start >= segments[i-1].start``.
      * All ``id`` values non-negative.
    """
    last_start = -1.0
    for seg in segments:
        if not isinstance(seg, Segment):
            raise ValueError(f"expected Segment, got {type(seg).__name__}")
        if seg.id < 0:
            raise ValueError(f"segment id must be non-negative: {seg.id}")
        if seg.start < 0 or seg.end < 0:
            raise ValueError(f"segment {seg.id} has negative timestamps")
        if seg.end < seg.start:
            raise ValueError(
                f"segment {seg.id} has end < start ({seg.end} < {seg.start})"
            )
        if seg.start < last_start - 1e-3:
            raise ValueError(
                f"segment {seg.id} starts before the previous segment ({seg.start} < {last_start})"
            )
        last_start = seg.start
        if seg.words:
            for w in seg.words:
                if not isinstance(w, Word):
                    raise ValueError(f"expected Word, got {type(w).__name__}")
                if w.end < w.start:
                    raise ValueError(
                        f"word '{w.word}' has end < start ({w.end} < {w.start})"
                    )


def build_cache_key(audio_hash: str, options: TranscribeOptions) -> str:
    """Deterministic hash over the inputs that materially affect the
    output. Anything not in here is presumed not to invalidate the
    cache.
    """
    parts = [
        "v2",
        audio_hash,
        options.model,
        options.normalised_language(),
        options.device,
        options.compute_type,
        f"beam={options.beam_size}",
        f"words={1 if options.word_timestamps else 0}",
        f"vad={1 if options.vad_filter else 0}",
        f"temp={options.temperature:.4f}",
        f"prompt={options.initial_prompt or ''}",
        f"reseg={1 if options.resegment else 0}",
    ]
    payload = "\x1f".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


# --------------------------------------------------------------------- helpers


def _segment_to_dict(s: Segment) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": s.id,
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "text": s.text,
    }
    if s.avg_logprob is not None:
        d["avgLogprob"] = s.avg_logprob
    if s.no_speech_prob is not None:
        d["noSpeechProb"] = s.no_speech_prob
    if s.speaker_id:
        d["speakerId"] = s.speaker_id
    if s.speaker_confidence is not None:
        d["speakerConfidence"] = round(float(s.speaker_confidence), 4)
    if s.words:
        d["words"] = [_word_to_dict(w) for w in s.words]
    return d


def _segment_from_dict(payload: Any) -> Segment:
    if not isinstance(payload, dict):
        raise ValueError("segment must be an object")
    words_raw = payload.get("words")
    words = [_word_from_dict(w) for w in words_raw] if isinstance(words_raw, list) else None
    try:
        return Segment(
            id=int(payload["id"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=str(payload["text"]),
            avg_logprob=_opt_float(payload.get("avgLogprob")),
            no_speech_prob=_opt_float(payload.get("noSpeechProb")),
            words=words,
            speaker_id=(
                str(payload.get("speakerId"))
                if payload.get("speakerId") is not None
                else None
            ),
            speaker_confidence=_opt_float(payload.get("speakerConfidence")),
        )
    except KeyError as e:
        raise ValueError(f"segment missing required field: {e}") from e


def _word_to_dict(w: Word) -> dict[str, Any]:
    d: dict[str, Any] = {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
    if w.probability is not None:
        d["probability"] = w.probability
    return d


def _word_from_dict(payload: Any) -> Word:
    if not isinstance(payload, dict):
        raise ValueError("word must be an object")
    return Word(
        word=str(payload.get("word", "")),
        start=float(payload.get("start", 0.0)),
        end=float(payload.get("end", 0.0)),
        probability=_opt_float(payload.get("probability")),
    )


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------- test helper


def _debug_asdict(t: Transcript) -> dict[str, Any]:  # pragma: no cover - dev use
    return asdict(t)
