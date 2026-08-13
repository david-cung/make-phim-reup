"""Tiny WAV helpers.

We deliberately stay on top of the ``wave`` stdlib module and hand-
rolled byte manipulation so the TTS subsystem carries no numpy
dependency. All helpers assume 16-bit PCM mono, which is what Piper
emits and what downstream Phase 7 mixing will consume.
"""

from __future__ import annotations

import struct
import wave
from array import array
from math import sqrt
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


def postprocess_pcm16_wav(
    path: str | Path,
    *,
    volume: float = 1.0,
    peak_limit: float = 0.95,
    fade_ms: int = 5,
    target_rms_dbfs: float = -20.0,
) -> None:
    """Apply conservative gain/peak limiting and tiny edge fades.

    This keeps dialogue levels consistent without loudness-maximising the
    performance or destroying intentional dynamics.
    """
    src = Path(path)
    with wave.open(str(src), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if sample_width != 2 or not frames:
        return

    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        return
    peak = max(abs(value) for value in samples)
    rms = sqrt(sum(float(value) * float(value) for value in samples) / len(samples))
    safe_peak = max(0.1, min(0.99, float(peak_limit))) * 32767.0
    target_rms = 32767.0 * (10.0 ** (float(target_rms_dbfs) / 20.0))
    normalisation_gain = (
        max(0.75, min(1.5, target_rms / rms)) if rms > 1.0 else 1.0
    )
    gain = max(0.0, float(volume)) * normalisation_gain
    if peak > 0:
        gain = min(gain, safe_peak / peak)

    fade_frames = min(
        int(rate * max(0, fade_ms) / 1000),
        len(samples) // max(1, channels) // 2,
    )
    total_frames = len(samples) // max(1, channels)
    for frame in range(total_frames):
        edge = 1.0
        if fade_frames > 0 and frame < fade_frames:
            edge = frame / fade_frames
        elif fade_frames > 0 and frame >= total_frames - fade_frames:
            edge = (total_frames - 1 - frame) / fade_frames
        scale = gain * max(0.0, edge)
        for channel in range(channels):
            index = frame * channels + channel
            samples[index] = int(max(-32768, min(32767, samples[index] * scale)))

    tmp = src.with_suffix(".postprocess.tmp.wav")
    with wave.open(str(tmp), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    tmp.replace(src)


def inspect_pcm16_wav(path: str | Path) -> dict[str, float | int]:
    """Return lightweight signal metrics used to reject broken output."""
    with wave.open(str(path), "rb") as reader:
        sample_width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if sample_width != 2:
        return {"peak": 0, "rms": 0.0, "clippedSamples": 0}
    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        return {"peak": 0, "rms": 0.0, "clippedSamples": 0}
    peak = max(abs(value) for value in samples)
    energy = sum(float(value) * float(value) for value in samples)
    clipped = sum(1 for value in samples if abs(value) >= 32760)
    return {
        "peak": peak,
        "rms": (energy / len(samples)) ** 0.5,
        "clippedSamples": clipped,
    }
