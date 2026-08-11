"""Transport-shaped dataclasses for the Phase 7 sync subsystem.

Kept free of any FFmpeg imports so this module is safe to load in
unit tests without a working FFmpeg install.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

SYNC_CACHE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------- settings


@dataclass
class SyncSettings:
    """Per-project sync knobs.

    * ``min_speed`` / ``max_speed`` bound how aggressively we allow
      ``atempo`` to stretch a line. The spec's default range is
      0.85 – 1.20 — anything beyond that gets flagged as *too_long*
      rather than silently distorted.
    * ``output_sample_rate`` is what we ask FFmpeg to resample to.
      ``None`` means "keep the source TTS WAV's rate", which is what
      Phase 7's default flow uses.
    """

    min_speed: float = 0.85
    max_speed: float = 1.20
    output_sample_rate: Optional[int] = None
    output_channels: int = 1

    def normalised(self) -> "SyncSettings":
        lo = _clamp(float(self.min_speed), 0.5, 1.0)
        hi = _clamp(float(self.max_speed), 1.0, 2.0)
        if hi < lo:
            hi = lo
        sr = int(self.output_sample_rate) if self.output_sample_rate else None
        if sr is not None and sr < 8000:
            sr = 8000
        return SyncSettings(
            min_speed=lo,
            max_speed=hi,
            output_sample_rate=sr,
            output_channels=int(self.output_channels) if self.output_channels else 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minSpeed": self.min_speed,
            "maxSpeed": self.max_speed,
            "outputSampleRate": self.output_sample_rate,
            "outputChannels": self.output_channels,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "SyncSettings":
        if not payload:
            return cls().normalised()
        try:
            sr_raw = payload.get("outputSampleRate")
            sr = int(sr_raw) if isinstance(sr_raw, (int, float)) and sr_raw else None
            return cls(
                min_speed=float(payload.get("minSpeed", 0.85)),
                max_speed=float(payload.get("maxSpeed", 1.20)),
                output_sample_rate=sr,
                output_channels=int(payload.get("outputChannels", 1)),
            ).normalised()
        except (TypeError, ValueError):
            return cls().normalised()


def _clamp(v: float, lo: float, hi: float) -> float:
    if v != v:  # NaN
        return lo
    return max(lo, min(hi, v))


# --------------------------------------------------------------------- statuses


# The per-segment classification the planner produces and the UI shows.
# Kept as a string constant set so we don't accidentally invent a new
# value on either side of the wire.
SYNC_STATUS_EMPTY = "empty"
SYNC_STATUS_FITS = "fits"
SYNC_STATUS_ADJUSTED = "adjusted"
SYNC_STATUS_TOO_LONG = "too_long"

_SYNC_STATUSES = {
    SYNC_STATUS_EMPTY,
    SYNC_STATUS_FITS,
    SYNC_STATUS_ADJUSTED,
    SYNC_STATUS_TOO_LONG,
}


def is_valid_status(status: str) -> bool:
    return status in _SYNC_STATUSES


# --------------------------------------------------------------------- planner


@dataclass(frozen=True)
class SyncPlan:
    """Pure result of :func:`planner.plan_segment` — no FFmpeg yet."""

    status: str
    target_duration_secs: float
    original_duration_secs: float
    final_duration_secs: float
    speed_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "targetDurationSecs": self.target_duration_secs,
            "originalDurationSecs": self.original_duration_secs,
            "finalDurationSecs": self.final_duration_secs,
            "speedFactor": self.speed_factor,
        }


# --------------------------------------------------------------------- segments


@dataclass
class BatchSegment:
    """One segment queued for `sync.apply_batch`.

    ``sourceFile`` is the *relative* path (under the project root) to
    the Phase 6 TTS WAV. ``outputFile`` is the relative output path
    the host wants us to write to.
    """

    id: int
    target_start: float
    target_end: float
    source_file: str
    output_file: str
    tts_cache_key: str
    source_sample_rate: Optional[int] = None

    @classmethod
    def from_wire(cls, payload: Any) -> "BatchSegment":
        if not isinstance(payload, dict):
            raise ValueError("segment must be an object")
        try:
            sid = int(payload["id"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid segment id: {e}") from e

        def _f(key: str) -> float:
            v = payload.get(key)
            if not isinstance(v, (int, float)):
                raise ValueError(f"segment {sid} field {key!r} must be a number")
            return float(v)

        src = payload.get("sourceFile")
        out = payload.get("outputFile")
        cache_key = payload.get("ttsCacheKey", "")
        if not isinstance(src, str) or not src:
            raise ValueError(f"segment {sid} is missing sourceFile")
        if not isinstance(out, str) or not out:
            raise ValueError(f"segment {sid} is missing outputFile")
        sr_raw = payload.get("sourceSampleRate")
        sr = int(sr_raw) if isinstance(sr_raw, (int, float)) and sr_raw else None
        return cls(
            id=sid,
            target_start=_f("targetStart"),
            target_end=_f("targetEnd"),
            source_file=src,
            output_file=out,
            tts_cache_key=str(cache_key) if isinstance(cache_key, str) else "",
            source_sample_rate=sr,
        )

    @property
    def target_duration_secs(self) -> float:
        return max(0.0, self.target_end - self.target_start)


@dataclass
class SyncSegmentEntry:
    """What we return per completed segment.

    The wire shape mirrors what the Rust host expects to fold into
    ``voices/synced/sync.json`` and to hand back to the frontend as an
    incremental completion event.
    """

    segment_id: int
    plan: SyncPlan
    tts_cache_key: str
    cache_key: str
    file: str
    sample_rate: int
    channels: int
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.plan.to_dict()
        payload.update(
            {
                "segmentId": self.segment_id,
                "ttsCacheKey": self.tts_cache_key,
                "cacheKey": self.cache_key,
                "file": self.file,
                "sampleRate": self.sample_rate,
                "channels": self.channels,
                "sizeBytes": self.size_bytes,
            }
        )
        return payload


# --------------------------------------------------------------------- cache


def build_sync_cache_key(
    *,
    tts_cache_key: str,
    target_duration_secs: float,
    settings: SyncSettings,
) -> str:
    """Deterministic hash over everything that changes the synced WAV.

    Identical bytes on Python and Rust so both sides agree on cache
    validity. Rounds durations to 3 decimals (ms precision) so tiny
    floating-point wobble doesn't cause spurious misses.
    """
    s = settings.normalised()
    sr_part = str(s.output_sample_rate) if s.output_sample_rate is not None else ""
    parts = [
        f"sync_v{SYNC_CACHE_SCHEMA_VERSION}",
        f"tts={tts_cache_key}",
        f"target={target_duration_secs:.3f}",
        f"min_speed={s.min_speed:.4f}",
        f"max_speed={s.max_speed:.4f}",
        f"out_sample_rate={sr_part}",
        f"out_channels={s.output_channels}",
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --------------------------------------------------------------------- validation


def validate_batch(segments: Iterable[BatchSegment]) -> None:
    seen: set[int] = set()
    for seg in segments:
        if seg.id in seen:
            raise ValueError(f"duplicate segment id in sync batch: {seg.id}")
        seen.add(seg.id)
        if seg.target_end <= seg.target_start:
            raise ValueError(
                f"segment {seg.id}: targetEnd ({seg.target_end}) must exceed targetStart ({seg.target_start})"
            )
