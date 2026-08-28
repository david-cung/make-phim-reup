"""Turn Whisper output into subtitle-sized cues.

Whisper segments are acoustic chunks, not readable subtitles. When
word timestamps exist we rebuild cues from words using:

* sentence / punctuation boundaries
* max characters per line and max lines
* min / max cue duration
* a reading-speed ceiling

We never invent times. Word (or original segment) timestamps are the
only clock.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .models import Segment, Word

_UNKNOWN_SPEAKER = "UNKNOWN"

_SENTENCE_END = re.compile(r"[.!?…。！？]+$")
_SOFT_BREAK = re.compile(r"[,;:，、]$")
_WHITESPACE = re.compile(r"\s+")


class SegmenterSettings:
    def __init__(
        self,
        *,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        max_cps: float = 20.0,
    ) -> None:
        self.max_chars_per_line = max(12, int(max_chars_per_line))
        self.max_lines = max(1, int(max_lines))
        self.min_duration = max(0.2, float(min_duration))
        self.max_duration = max(self.min_duration, float(max_duration))
        self.max_cps = max(8.0, float(max_cps))

    @property
    def max_chars(self) -> int:
        return self.max_chars_per_line * self.max_lines


def resegment(segments: Iterable[Segment], settings: Optional[SegmenterSettings] = None) -> list[Segment]:
    """Rebuild subtitle cues from Whisper segments.

    If word timestamps are present they are the source of truth. If
    not, we still merge/split using punctuation and proportional
    timing of the original cue — never a fixed wall-clock grid.
    """
    cfg = settings or SegmenterSettings()
    source_segments = list(segments)
    if not any(seg.words for seg in source_segments):
        return _resegment_without_words(source_segments, cfg)
    words = _flatten_words(source_segments)
    if not words:
        return _resegment_without_words(source_segments, cfg)
    return _cues_from_words(words, cfg)


def _flatten_words(segments: Iterable[Segment]) -> list[Word]:
    out: list[Word] = []
    for seg in segments:
        if not seg.words:
            text = (seg.text or "").strip()
            if text:
                out.append(Word(word=text, start=seg.start, end=max(seg.end, seg.start + 0.01)))
            continue
        for word in seg.words:
            token = (word.word or "").strip()
            if not token:
                continue
            start = float(word.start)
            end = float(word.end)
            if end <= start:
                end = start + 0.04
            out.append(
                Word(
                    word=token,
                    start=start,
                    end=end,
                    probability=word.probability,
                )
            )
    return out


def _cues_from_words(words: list[Word], cfg: SegmenterSettings) -> list[Segment]:
    cues: list[Segment] = []
    bucket: list[Word] = []
    next_id = 0

    def flush() -> None:
        nonlocal next_id, bucket
        if not bucket:
            return
        text = _join_words(bucket)
        start = bucket[0].start
        end = max(bucket[-1].end, start + 0.04)
        cues.append(
            Segment(
                id=next_id,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                words=list(bucket),
            )
        )
        next_id += 1
        bucket = []

    for word in words:
        if not bucket:
            bucket = [word]
            continue
        candidate = bucket + [word]
        text = _join_words(candidate)
        duration = candidate[-1].end - candidate[0].start
        chars = len(text)
        cps = chars / duration if duration > 0.05 else chars / 0.05
        prev_text = _join_words(bucket)
        at_sentence = bool(_SENTENCE_END.search(prev_text.rstrip()))
        at_soft = bool(_SOFT_BREAK.search(prev_text.rstrip()))
        too_long = (
            chars > cfg.max_chars
            or duration > cfg.max_duration
            or cps > cfg.max_cps
        )
        if too_long and len(bucket) >= 1:
            flush()
            bucket = [word]
            continue
        if at_sentence and duration >= cfg.min_duration:
            flush()
            bucket = [word]
            continue
        if at_soft and (chars >= cfg.max_chars_per_line or duration >= cfg.max_duration * 0.7):
            flush()
            bucket = [word]
            continue
        bucket.append(word)

    flush()
    return _merge_tiny(cues, cfg)


def _resegment_without_words(segments: list[Segment], cfg: SegmenterSettings) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        if not merged:
            merged.append(
                _copy_segment_with(seg, id=0, text=text, words=None)
            )
            continue
        prev = merged[-1]
        gap = seg.start - prev.end
        combined = f"{prev.text} {text}".strip()
        duration = seg.end - prev.start
        if (
            gap <= 0.35
            and _can_merge_segments(prev, seg)
            and not _SENTENCE_END.search(prev.text.rstrip())
            and len(combined) <= cfg.max_chars
            and duration <= cfg.max_duration
        ):
            merged[-1] = Segment(
                id=prev.id,
                start=prev.start,
                end=seg.end,
                text=combined,
                words=None,
                speaker_id=prev.speaker_id,
                speaker_confidence=prev.speaker_confidence,
                raw_text=_join_text(prev.raw_text or prev.text, seg.raw_text or seg.text),
                normalized_text=combined,
                source_segment_id=prev.source_segment_id,
                source_quality=prev.source_quality,
                semantic_facts=prev.semantic_facts,
            )
        else:
            merged.append(
                _copy_segment_with(seg, id=len(merged), text=text, words=None)
            )

    split: list[Segment] = []
    for seg in merged:
        pieces = _split_long_text(seg.text, cfg)
        if len(pieces) == 1:
            split.append(
                _copy_segment_with(seg, id=len(split), text=pieces[0], words=None)
            )
            continue
        total_chars = sum(max(1, len(p)) for p in pieces)
        cursor = seg.start
        span = max(seg.end - seg.start, 0.04)
        for i, piece in enumerate(pieces):
            share = span * (len(piece) / total_chars)
            end = seg.end if i == len(pieces) - 1 else cursor + share
            split.append(
                _copy_segment_with(
                    seg,
                    id=len(split),
                    start=round(cursor, 3),
                    end=round(end, 3),
                    text=piece,
                    words=None,
                )
            )
            cursor = end
    return _merge_tiny(split, cfg)


def _split_long_text(text: str, cfg: SegmenterSettings) -> list[str]:
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) <= cfg.max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?…。！？])\s+", text)
    if len(parts) == 1:
        parts = re.split(r"(?<=[,;:，、])\s+", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        trial = f"{buf} {part}".strip() if buf else part
        if len(trial) <= cfg.max_chars:
            buf = trial
            continue
        if buf:
            out.append(buf)
        if len(part) <= cfg.max_chars:
            buf = part
        else:
            out.extend(_hard_wrap(part, cfg.max_chars))
            buf = ""
    if buf:
        out.append(buf)
    return out or [text]


def _hard_wrap(text: str, limit: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for word in words:
        extra = len(word) + (1 if buf else 0)
        if buf and size + extra > limit:
            out.append(" ".join(buf))
            buf = [word]
            size = len(word)
        else:
            buf.append(word)
            size += extra
    if buf:
        out.append(" ".join(buf))
    return out


def _merge_tiny(cues: list[Segment], cfg: SegmenterSettings) -> list[Segment]:
    if not cues:
        return []
    out: list[Segment] = []
    for cue in cues:
        if (
            out
            and (cue.end - cue.start) < cfg.min_duration
            and _can_merge_segments(out[-1], cue)
            and len(_join_words(out[-1].words) if out[-1].words else out[-1].text)
            + 1
            + len(cue.text)
            <= cfg.max_chars
            and (cue.end - out[-1].start) <= cfg.max_duration
        ):
            prev = out[-1]
            words = None
            if prev.words is not None and cue.words is not None:
                words = list(prev.words) + list(cue.words)
            out[-1] = Segment(
                id=prev.id,
                start=prev.start,
                end=cue.end,
                text=_join_text(prev.text, cue.text),
                words=words,
                speaker_id=prev.speaker_id,
                speaker_confidence=prev.speaker_confidence,
                raw_text=_join_text(prev.raw_text or prev.text, cue.raw_text or cue.text),
                normalized_text=_join_text(prev.normalized_text or prev.text, cue.normalized_text or cue.text),
                source_segment_id=prev.source_segment_id,
                source_quality=prev.source_quality,
                semantic_facts=prev.semantic_facts,
            )
        else:
            out.append(cue)
    for i, cue in enumerate(out):
        out[i] = Segment(
            id=i,
            start=cue.start,
            end=cue.end,
            text=cue.text,
            avg_logprob=cue.avg_logprob,
            no_speech_prob=cue.no_speech_prob,
            words=cue.words,
            speaker_id=cue.speaker_id,
            speaker_confidence=cue.speaker_confidence,
            raw_text=cue.raw_text,
            normalized_text=cue.normalized_text,
            source_segment_id=cue.source_segment_id,
            source_sub_segment_id=cue.source_sub_segment_id,
            source_quality=cue.source_quality,
            semantic_facts=cue.semantic_facts,
        )
    return out


def _join_words(words: Iterable[Word]) -> str:
    parts: list[str] = []
    for word in words:
        token = (word.word or "").strip()
        if not token:
            continue
        if parts and re.match(r"^[.,!?;:…'\"”’)\]]", token):
            parts[-1] = parts[-1] + token
        elif token.startswith("'") and parts:
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return _WHITESPACE.sub(" ", " ".join(parts)).strip()


def _join_text(left: str, right: str) -> str:
    return _WHITESPACE.sub(" ", f"{left} {right}").strip()


def _copy_segment_with(
    seg: Segment,
    *,
    id: int,
    text: str,
    words: list[Word] | None,
    start: float | None = None,
    end: float | None = None,
) -> Segment:
    return Segment(
        id=id,
        start=seg.start if start is None else start,
        end=seg.end if end is None else end,
        text=text,
        avg_logprob=seg.avg_logprob,
        no_speech_prob=seg.no_speech_prob,
        words=words,
        speaker_id=seg.speaker_id,
        speaker_confidence=seg.speaker_confidence,
        raw_text=seg.raw_text,
        normalized_text=seg.normalized_text,
        source_segment_id=seg.source_segment_id,
        source_sub_segment_id=seg.source_sub_segment_id,
        source_quality=seg.source_quality,
        semantic_facts=seg.semantic_facts,
    )


def _speaker_key(seg: Segment) -> str:
    speaker = (seg.speaker_id or "").strip()
    return speaker if speaker else _UNKNOWN_SPEAKER


def _can_merge_segments(left: Segment, right: Segment) -> bool:
    if not (left.speaker_id or "").strip() and not (right.speaker_id or "").strip():
        return True
    left_speaker = _speaker_key(left)
    right_speaker = _speaker_key(right)
    if left_speaker == _UNKNOWN_SPEAKER and right_speaker == _UNKNOWN_SPEAKER:
        return False
    if left_speaker == _UNKNOWN_SPEAKER or right_speaker == _UNKNOWN_SPEAKER:
        return False
    return left_speaker == right_speaker
