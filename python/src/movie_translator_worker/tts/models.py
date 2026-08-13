"""Transport-shaped dataclasses for the TTS subsystem.

Kept free of any engine-specific imports so this module can be pulled
in by unit tests without ``piper``, ``onnxruntime`` or any other
runtime dependency being installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

TTS_CACHE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------- voices


@dataclass(frozen=True)
class VoiceInfo:
    """Metadata for one installed TTS voice.

    The :attr:`supported_settings` list tells the UI which of
    ``speed``/``pitch``/``volume`` the underlying engine actually
    honours — the UI hides sliders for the rest so users don't set a
    value the engine will silently ignore.
    """

    id: str
    name: str
    language: str
    gender: str  # "male" | "female" | "neutral" | "unknown"
    engine: str  # "piper" | ...
    model_path: str  # absolute path to the primary model file
    config_path: Optional[str]  # e.g. Piper's ``model.onnx.json``
    sample_rate: int
    installed: bool
    quality: Optional[str] = None  # engine-specific quality/tier tag
    supported_settings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modelPath"] = payload.pop("model_path")
        payload["configPath"] = payload.pop("config_path")
        payload["sampleRate"] = payload.pop("sample_rate")
        payload["supportedSettings"] = payload.pop("supported_settings")
        return payload


# ------------------------------------------------------------------- settings


@dataclass
class TTSSettings:
    """Per-segment synthesis knobs.

    Everything is stored here even when the current engine doesn't
    honour it, because the cache key hashes all three fields — a user
    who switches to an engine that *does* respect e.g. ``pitch`` still
    gets a cache miss and a re-render at that point.
    """

    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0

    def normalised(self) -> "TTSSettings":
        return TTSSettings(
            speed=_clamp(float(self.speed), 0.25, 4.0),
            pitch=_clamp(float(self.pitch), -12.0, 12.0),
            volume=_clamp(float(self.volume), 0.0, 4.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"speed": self.speed, "pitch": self.pitch, "volume": self.volume}

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "TTSSettings":
        if not payload:
            return cls()
        try:
            return cls(
                speed=float(payload.get("speed", 1.0)),
                pitch=float(payload.get("pitch", 0.0)),
                volume=float(payload.get("volume", 1.0)),
            ).normalised()
        except (TypeError, ValueError):
            return cls()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------------------------------------------- results


@dataclass
class SynthesisResult:
    """Filesystem-oriented result returned by :meth:`TTSProvider.synthesize`."""

    file_path: str
    duration_secs: float
    sample_rate: int
    channels: int
    size_bytes: int


# ------------------------------------------------------------------- segments


@dataclass
class BatchSegment:
    """One subtitle segment queued for synthesis.

    Only ``id`` and ``text`` are strictly required; the rest override
    the batch defaults if present.
    """

    id: int
    text: str
    voice_id: Optional[str] = None
    settings: Optional[TTSSettings] = None
    target_duration_secs: Optional[float] = None

    @classmethod
    def from_wire(cls, payload: Any) -> "BatchSegment":
        if not isinstance(payload, dict):
            raise ValueError("segment must be an object")
        try:
            sid = int(payload["id"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid segment id: {e}") from e
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError(f"segment {sid} is missing text")
        vid_raw = payload.get("voiceId")
        vid: Optional[str] = str(vid_raw) if isinstance(vid_raw, str) and vid_raw else None
        settings_raw = payload.get("settings")
        settings = (
            TTSSettings.from_dict(settings_raw) if isinstance(settings_raw, dict) else None
        )
        target_raw = payload.get("targetDurationSecs")
        target: Optional[float] = None
        if isinstance(target_raw, (int, float)):
            target = float(target_raw)
        return cls(
            id=sid,
            text=text,
            voice_id=vid,
            settings=settings,
            target_duration_secs=target,
        )


# ------------------------------------------------------------------- cache


def build_segment_cache_key(
    *,
    engine: str,
    voice_id: str,
    model_name: str,
    text: str,
    settings: TTSSettings,
) -> str:
    """Deterministic hash over everything that changes the generated WAV.

    Kept identical between the host (Rust) and the worker (Python) so
    both sides agree on what "already generated" means.
    """
    s = settings.normalised()
    parts = [
        f"tts_v{TTS_CACHE_SCHEMA_VERSION}",
        engine or "",
        voice_id or "",
        model_name or "",
        f"speed={s.speed:.4f}",
        f"pitch={s.pitch:.4f}",
        f"volume={s.volume:.4f}",
        "text=" + text,
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- validation


def validate_batch(segments: Iterable[BatchSegment]) -> None:
    seen: set[int] = set()
    for seg in segments:
        if seg.id in seen:
            raise ValueError(f"duplicate segment id in batch: {seg.id}")
        seen.add(seg.id)
        if not seg.text.strip():
            raise ValueError(f"segment {seg.id} has empty text")
