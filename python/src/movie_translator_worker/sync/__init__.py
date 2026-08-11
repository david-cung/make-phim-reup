"""Voice / subtitle timing synchronisation (Phase 7).

Turns the per-segment TTS WAVs produced in Phase 6 into
*timing-adjusted* WAVs that fit their subtitle window on the movie
timeline.

Nothing in this package imports ``ffmpeg`` at module import time; the
actual FFmpeg invocation lives in :mod:`.ffmpeg_apply` and is only
touched when :func:`.handlers.sync_apply_batch` needs it, so tests and
non-sync RPC methods pay no cost.

Public surface:

  * :mod:`.models` — dataclasses used on the wire.
  * :mod:`.planner` — pure math that classifies each segment as
    ``fits`` / ``adjusted`` / ``too_long`` and computes the exact
    ``speedFactor`` FFmpeg should apply.
  * :mod:`.ffmpeg_apply` — the FFmpeg subprocess wrapper.
  * :mod:`.handlers` — RPC handlers, wired into the top-level
    dispatcher by :func:`.handlers.install`.
"""

from __future__ import annotations
