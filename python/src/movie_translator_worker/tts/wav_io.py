"""Tiny WAV helpers.

We deliberately stay on top of the ``wave`` stdlib module and hand-
rolled byte manipulation so the TTS subsystem carries no numpy
dependency. All helpers assume 16-bit PCM mono, which is what Piper
emits and what downstream Phase 7 mixing will consume.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Tuple


def write_pcm16_mono(
    path: str | Path,
    samples: bytes,
    *,
    sample_rate: int,
) -> None:
    """Write raw little-endian PCM16 samples out as a canonical WAV file.

    ``samples`` must be a byte string whose length is a multiple of 2
    (one 16-bit sample per pair). We write via ``wave`` so the resulting
    file has a proper RIFF header FFmpeg / browsers understand.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(samples)


def probe_wav(path: str | Path) -> Tuple[float, int, int]:
    """Return ``(duration_secs, sample_rate, channels)`` for ``path``."""
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
    duration = frames / float(rate) if rate > 0 else 0.0
    return (duration, int(rate), int(channels))


def apply_volume_pcm16(samples: bytes, gain: float) -> bytes:
    """Apply a linear amplitude scale to raw PCM16 samples.

    Used when the caller passed a ``volume`` other than 1.0 and the
    underlying engine can't apply it natively (Piper). We clip to
    int16 range to avoid wraparound.
    """
    if abs(gain - 1.0) < 1e-4:
        return samples
    if len(samples) % 2 != 0:
        raise ValueError("PCM16 sample buffer length must be even")
    count = len(samples) // 2
    if count == 0:
        return samples
    unpacked = struct.unpack("<" + "h" * count, samples)
    scaled = []
    for s in unpacked:
        v = int(s * gain)
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        scaled.append(v)
    return struct.pack("<" + "h" * count, *scaled)
