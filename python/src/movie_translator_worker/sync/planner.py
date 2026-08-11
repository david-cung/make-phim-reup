"""Pure fit-check logic for Phase 7.

Given a target subtitle window and the raw TTS duration, produce a
:class:`SyncPlan` describing what FFmpeg should do (nothing / stretch /
pad) and how the UI should classify the row.

There are three outcomes per the spec:

* **fits** — the raw voice fits inside the window. We do NOT slow it
  down artificially; we just pad the tail with silence so downstream
  mixing can place it at ``target_start``.
* **adjusted** — the voice is longer than the window but within the
  allowed ``[min_speed, max_speed]`` stretch. FFmpeg's ``atempo``
  handles it without touching pitch.
* **too_long** — even stretching to ``max_speed`` leaves the voice
  longer than the window. We still produce a WAV (at max_speed) so
  the user can hear it, but we set a warning status so the UI can
  flag the line and prompt them to shorten the translation.
"""

from __future__ import annotations

from .models import (
    SYNC_STATUS_ADJUSTED,
    SYNC_STATUS_EMPTY,
    SYNC_STATUS_FITS,
    SYNC_STATUS_TOO_LONG,
    SyncPlan,
    SyncSettings,
)

# Anything shorter than this is treated as "no audio" — Piper occasionally
# emits a 20-sample warm-up burst if the text is only punctuation.
_EMPTY_THRESHOLD_SECS = 0.02

# Match tolerance when comparing durations. Below this, we treat
# `original ≈ target` as *fits*, no atempo needed.
_FIT_TOLERANCE_SECS = 0.01


def plan_segment(
    *,
    target_duration_secs: float,
    source_duration_secs: float,
    settings: SyncSettings,
) -> SyncPlan:
    """Classify one segment into a :class:`SyncPlan`.

    The plan is deterministic — same inputs, same output — so both
    the host (Rust) and the worker (Python) can reason about a
    segment's expected status without touching FFmpeg.
    """
    s = settings.normalised()
    target = max(0.0, float(target_duration_secs))
    source = max(0.0, float(source_duration_secs))

    if source <= _EMPTY_THRESHOLD_SECS:
        # No usable TTS — output will be pure silence of the requested
        # length. Downstream still gets a WAV so segment ids stay in
        # lockstep.
        return SyncPlan(
            status=SYNC_STATUS_EMPTY,
            target_duration_secs=target,
            original_duration_secs=source,
            final_duration_secs=target,
            speed_factor=1.0,
        )

    if source <= target + _FIT_TOLERANCE_SECS:
        # Voice already fits — pad tail with silence so total = target.
        return SyncPlan(
            status=SYNC_STATUS_FITS,
            target_duration_secs=target,
            original_duration_secs=source,
            final_duration_secs=target,
            speed_factor=1.0,
        )

    if target <= 0.0:
        # Degenerate case: subtitle has no window but we have audio.
        # Treat as too_long at the max cap so we still emit something.
        return SyncPlan(
            status=SYNC_STATUS_TOO_LONG,
            target_duration_secs=0.0,
            original_duration_secs=source,
            final_duration_secs=source / max(s.max_speed, 1e-6),
            speed_factor=s.max_speed,
        )

    required_speed = source / target
    if required_speed <= s.max_speed + 1e-6:
        # Stretch fits inside our allowed range → adjust cleanly.
        speed = max(required_speed, s.min_speed)
        return SyncPlan(
            status=SYNC_STATUS_ADJUSTED,
            target_duration_secs=target,
            original_duration_secs=source,
            final_duration_secs=source / speed,
            speed_factor=speed,
        )

    # Even at max_speed we overflow — surface as a warning but still
    # produce max-stretched audio (never worse than 1.20× by default).
    return SyncPlan(
        status=SYNC_STATUS_TOO_LONG,
        target_duration_secs=target,
        original_duration_secs=source,
        final_duration_secs=source / s.max_speed,
        speed_factor=s.max_speed,
    )
