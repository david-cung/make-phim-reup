"""Local Text-to-Speech (Phase 6).

Public surface:

  * :class:`~.provider.TTSProvider` — the abstraction the rest of the
    application talks to; concrete engines live behind it.
  * :class:`~.manager.TTSManager` — routes engine ids to a provider.
  * :class:`~.piper_provider.PiperTTSProvider` — the Vietnamese-capable
    local engine (small, offline, permissive license).
  * :mod:`~.handlers` — RPC handlers, wired into the main dispatcher
    by :func:`.handlers.install`.

Nothing in this package imports the heavy ``piper`` runtime at module
import time; that only happens lazily inside the provider so tests and
non-TTS RPC methods pay no cost.
"""

from __future__ import annotations
