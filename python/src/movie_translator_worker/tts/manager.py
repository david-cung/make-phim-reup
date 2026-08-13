"""Local TTS routing.

The rest of the pipeline talks to :class:`TTSManager`, never to a
concrete engine. Today the Vietnamese voice is Piper; additional local
engines can register here later without rewriting transcription,
translation, sync, or mix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..errors import RpcError, RpcErrorCode
from .piper_provider import PiperTTSProvider
from .provider import TTSProvider


class TTSManager:
    """``TTSManager → TTSProvider`` seam used by the RPC handlers."""

    def __init__(
        self,
        models_root: Path,
        providers: Optional[dict[str, TTSProvider]] = None,
    ) -> None:
        self._models_root = Path(models_root)
        self._providers: dict[str, TTSProvider] = dict(providers or {})

    def provider(self, engine: str) -> TTSProvider:
        key = (engine or "piper").strip().lower()
        if key in ("vietnamese", "vi", "vi-vn"):
            key = "piper"
        if key in self._providers:
            return self._providers[key]
        if key == "piper":
            provider: TTSProvider = PiperTTSProvider(self._models_root)
            self._providers[key] = provider
            return provider
        raise RpcError(
            RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
            f"unknown local TTS engine: {engine!r}",
        )

    def items(self) -> list[tuple[str, TTSProvider]]:
        return list(self._providers.items())
