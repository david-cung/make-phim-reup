"""The public ``TTSProvider`` protocol.

The rest of the application depends only on this module — never on
``piper`` or any other concrete engine directly. That gives us a clean
seam for tests (which use a fake provider) and for future runtimes
(Coqui TTS, ESPnet, XTTS, MMS-TTS, cloud, ...) without touching the
orchestration layers.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from .models import SynthesisResult, TTSSettings, VoiceInfo


# (fraction 0..1, stage, optional message)
ProgressCallback = Callable[[float, str, Optional[str]], None]


class ProviderError(Exception):
    """Structured error raised by a TTS provider.

    ``code`` must be one of the ``TTS_*`` codes defined in
    :mod:`movie_translator_worker.errors`.
    """

    def __init__(
        self, code: str, message: str, *, recoverable: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable


class ProviderCancelled(Exception):
    """Raised by providers when a running synthesis is aborted."""


@runtime_checkable
class TTSProvider(Protocol):
    """Minimal TTS surface consumed by :mod:`.handlers`.

    A concrete provider MUST:

      * enumerate installed voices via :meth:`get_voices`;
      * synthesise a single utterance to a filesystem path via
        :meth:`synthesize`, honouring only settings it truly supports;
      * report the effective ``sample_rate``/``duration``/``channels``
        in the returned :class:`SynthesisResult` (Phase 7 will need
        them for lip-sync padding);
      * raise :class:`ProviderError` with a stable ``code`` for every
        foreseeable failure so the host can react;
      * release any RAM-heavy state through :meth:`unload` when asked
        to idle between generations.
    """

    name: str

    def get_voices(self) -> list[VoiceInfo]: ...

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        settings: TTSSettings,
    ) -> SynthesisResult: ...

    def unload(self) -> None: ...
