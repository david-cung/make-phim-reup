"""Local speech-to-text (Phase 3).

The public surface is deliberately narrow so nothing else in the app
depends directly on faster-whisper. Consumers should only touch the
:class:`SpeechToTextProvider` protocol and the transport-shaped
dataclasses in :mod:`.models`.
"""

from __future__ import annotations

from . import registry
from .device import DeviceInfo, detect_devices
from .models import (
    TRANSCRIPT_SCHEMA_VERSION,
    Segment,
    TranscribeOptions,
    Transcript,
    Word,
    build_cache_key,
    transcript_from_dict,
    transcript_to_dict,
    validate_segments,
)
from .provider import (
    ProgressCallback,
    ProviderCancelled,
    ProviderError,
    SpeechToTextProvider,
    TranscribeContext,
)

__all__ = [
    "DeviceInfo",
    "ProgressCallback",
    "ProviderCancelled",
    "ProviderError",
    "Segment",
    "SpeechToTextProvider",
    "TRANSCRIPT_SCHEMA_VERSION",
    "TranscribeContext",
    "TranscribeOptions",
    "Transcript",
    "Word",
    "build_cache_key",
    "detect_devices",
    "registry",
    "transcript_from_dict",
    "transcript_to_dict",
    "validate_segments",
]
