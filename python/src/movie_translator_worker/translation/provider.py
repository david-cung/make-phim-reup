"""The public ``TranslationProvider`` protocol.

The rest of the application depends only on this module — never on
``llama_cpp`` directly. That gives us a clean seam for tests (which use
a fake provider) and for future runtimes (Ollama, MLX, mistral.rs,
cloud, ...) without touching orchestration.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from .models import TranslateOptions, TranslationChunk, TranslatedSegment, TranslationResult


# (fraction 0..1, stage, optional message)
ProgressCallback = Callable[[float, str, Optional[str]], None]


class ProviderError(Exception):
    """Structured error raised by a translation provider.

    ``code`` should be one of the ``TRANSLATE_*`` codes defined in
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
    """Raised by providers when a running translation is aborted."""


class TranslateContext(Protocol):
    """Runtime hooks passed to a provider for a single translate call."""

    def cancelled(self) -> bool: ...

    def on_progress(
        self, fraction: float, stage: str, message: Optional[str] = None
    ) -> None: ...

    def on_chunk_completed(
        self,
        chunk_index: int,
        translations: dict[int, TranslationResult],
    ) -> None:
        """Callback invoked as each chunk finishes so the host can persist
        incrementally. ``translations`` maps ``segment_id -> result``.
        """
        ...


@runtime_checkable
class TranslationProvider(Protocol):
    """Minimal translation surface consumed by :mod:`.service`."""

    name: str

    def translate_chunks(
        self,
        chunks: list[TranslationChunk],
        segments_by_id: dict[int, TranslatedSegment],
        options: TranslateOptions,
        ctx: TranslateContext,
    ) -> dict[int, TranslationResult]:
        """Translate the given chunks.

        Returns a ``{segment_id: translation}`` map covering every id
        listed across ``chunks``. Implementations MUST:

          * check ``ctx.cancelled()`` at least between chunks and raise
            :class:`ProviderCancelled` promptly when true;
          * call ``ctx.on_progress`` and ``ctx.on_chunk_completed`` as
            each chunk finishes so the host can persist partial results;
          * raise :class:`ProviderError` with a stable ``code`` for every
            foreseeable failure so the host can react.
        """
