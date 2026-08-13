"""Local TTS duration / pause helpers.

Used by the Piper provider so we can:

* keep punctuation pauses instead of gluing every sentence together
* shorten spoken text slightly when a line overruns its subtitle window
* never invent meaning — shortening only drops filler / extra words
"""

from __future__ import annotations

import re
from typing import Optional

_ELLIPSIS = re.compile(r"\.\.\.|…")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_VI_FILLERS = (
    "thì ",
    "mà ",
    "đang ",
    "cái ",
    "ạ,",
    "ạ ",
    "ừ ",
    "ờ ",
    "nhỉ",
    "nhé",
    "cơ mà",
    "thật sự ",
)

PAUSE_MS_ELLIPSIS = 280
PAUSE_MS_SENTENCE = 160


def split_spoken_units(text: str) -> list[tuple[str, int]]:
    """Return ``(utterance, trailing_pause_ms)`` units.

    Trailing pause is applied *after* the unit, never before the first.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    chunks = _ELLIPSIS.split(raw)
    units: list[tuple[str, int]] = []
    for i, chunk in enumerate(chunks):
        piece = chunk.strip()
        if not piece:
            if units:
                last_text, last_pause = units[-1]
                units[-1] = (last_text, max(last_pause, PAUSE_MS_ELLIPSIS))
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(piece) if s.strip()]
        if not sentences:
            sentences = [piece]
        for j, sentence in enumerate(sentences):
            pause = PAUSE_MS_SENTENCE if j < len(sentences) - 1 else 0
            units.append((sentence, pause))
        if i < len(chunks) - 1 and units:
            last_text, last_pause = units[-1]
            units[-1] = (last_text, max(last_pause, PAUSE_MS_ELLIPSIS))
    return units or [(raw, 0)]


def shorten_for_duration(text: str, ratio: float) -> Optional[str]:
    """Return a shorter spoken line when ``ratio`` (actual/target) > 1.

    ``None`` means we could not shorten without emptying the line.
    """
    original = (text or "").strip()
    if not original or ratio <= 1.05:
        return None
    candidate = original
    if ratio > 1.12:
        for filler in _VI_FILLERS:
            candidate = candidate.replace(filler, " ")
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,;.")
    if ratio > 1.25:
        candidate = re.sub(r"\s*\([^)]*\)", "", candidate)
        candidate = candidate.replace("...", ".").replace("…", ".")
        candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate or candidate == original:
        return None
    return candidate


def silence_pcm16(sample_rate: int, duration_ms: int) -> bytes:
    if duration_ms <= 0 or sample_rate <= 0:
        return b""
    samples = int(sample_rate * (duration_ms / 1000.0))
    return b"\x00\x00" * max(0, samples)
