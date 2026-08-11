"""The public ``SpeechToTextProvider`` interface.

The rest of the application should only ever depend on this module,
never on ``faster-whisper`` directly. That gives us a clean seam for
tests (which use a fake provider) and future providers (whisper.cpp,
ggml, sherpa, cloud, ...) without touching the orchestrator.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from .models import Segment, TranscribeOptions


# progress fraction (0..1), stage label, and an optional free-form message
ProgressCallback = Callable[[float, str, Optional[str]], None]


class ProviderError(Exception):
    """A structured error emitted by a provider.

    The ``code`` is expected to be one of the ``STT_*`` codes defined
    in :mod:`movie_translator_worker.errors`.
    """

    def __init__(self, code: str, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable


class ProviderCancelled(Exception):
    """Raised by providers when a running transcription is aborted."""


class TranscribeContext(Protocol):
    """Runtime hooks passed to a provider for a single transcription.

    Deliberately a Protocol so tests can pass a plain object without
    inheriting anything.
    """

    def cancelled(self) -> bool: ...
    def on_progress(self, fraction: float, stage: str, message: Optional[str] = None) -> None: ...


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Minimal STT surface consumed by :mod:`.service`."""

    name: str

    def transcribe(
        self,
        audio_path: str,
        options: TranscribeOptions,
        ctx: TranscribeContext,
    ) -> tuple[str, list[Segment]]:
        """Run inference and return ``(detected_language, segments)``.

        Implementations MUST:

          * poll ``ctx.cancelled()`` between segments and raise
            :class:`ProviderCancelled` promptly when true;
          * call ``ctx.on_progress`` at reasonable intervals; and
          * raise :class:`ProviderError` with a stable ``code`` for
            every foreseeable failure (missing model, out of memory,
            invalid audio, ...) so the host can react.
        """
