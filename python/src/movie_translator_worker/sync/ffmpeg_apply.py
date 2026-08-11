"""FFmpeg wrapper for Phase 7 sync application.

Given a source TTS WAV and a :class:`SyncPlan`, produce the
timing-adjusted WAV on disk. We stay on top of a *single* FFmpeg
subprocess per segment because:

* the source files are typically well under a megabyte, so per-invoke
  overhead is negligible;
* it maps cleanly to per-segment cancellation (kill the child on
  request);
* it lets us keep the pipeline description short enough to reason
  about in a single ``-af`` filter chain.

Nothing about the *movie* soundtrack is touched here — that's Phase 8
mixing. This stage only reshapes per-segment WAVs.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional

from ..errors import RpcErrorCode
from ..tts.wav_io import probe_wav
from .models import (
    SYNC_STATUS_ADJUSTED,
    SYNC_STATUS_EMPTY,
    SYNC_STATUS_FITS,
    SYNC_STATUS_TOO_LONG,
    SyncPlan,
    SyncSettings,
)


class SyncApplyError(Exception):
    """Raised by :func:`apply_plan` when FFmpeg fails.

    ``code`` maps to :class:`~movie_translator_worker.errors.RpcErrorCode`
    so the caller can rethrow as a stable :class:`RpcError`.
    """

    def __init__(self, code: str, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable


def apply_plan(
    *,
    ffmpeg_bin: Optional[str],
    source_wav: Path,
    output_wav: Path,
    plan: SyncPlan,
    settings: SyncSettings,
    source_sample_rate: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout_secs: float = 60.0,
) -> tuple[float, int, int, int]:
    """Materialise ``plan`` into ``output_wav``.

    Returns ``(actual_duration_secs, sample_rate, channels, size_bytes)``
    of the resulting file, re-probed from disk so the value we return
    matches what downstream stages will actually read.

    If ``plan.status`` is ``empty``, we generate ``target_duration``
    seconds of pure silence via Python's stdlib ``wave`` module — no
    FFmpeg needed. That keeps segment ids in lockstep even when a
    subtitle has no usable voice yet.
    """
    if source_sample_rate is None or source_sample_rate <= 0:
        source_sample_rate = _probe_source_rate(source_wav)

    out_sr = settings.output_sample_rate or source_sample_rate
    out_channels = max(1, int(settings.output_channels))

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if plan.status == SYNC_STATUS_EMPTY:
        # Pure silence of the target duration — bypass FFmpeg.
        _write_silence_wav(
            output_wav,
            duration_secs=plan.target_duration_secs,
            sample_rate=out_sr,
            channels=out_channels,
        )
    else:
        if not source_wav.is_file() or source_wav.stat().st_size == 0:
            raise SyncApplyError(
                RpcErrorCode.SYNC_SOURCE_MISSING,
                f"source TTS wav not found or empty: {source_wav}",
            )
        _run_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            source_wav=source_wav,
            output_wav=output_wav,
            plan=plan,
            out_sr=out_sr,
            out_channels=out_channels,
            cancel_event=cancel_event,
            timeout_secs=timeout_secs,
        )

    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise SyncApplyError(
            RpcErrorCode.SYNC_ENGINE_FAILURE,
            f"ffmpeg produced an empty file at {output_wav}",
        )

    duration, sr, channels = probe_wav(output_wav)
    size = output_wav.stat().st_size
    return (duration, int(sr), int(channels), int(size))


# ---------------------------------------------------------------------- helpers


def _resolve_ffmpeg(ffmpeg_bin: Optional[str]) -> str:
    if ffmpeg_bin and Path(ffmpeg_bin).is_file():
        return ffmpeg_bin
    fallback = shutil.which("ffmpeg")
    if not fallback:
        raise SyncApplyError(
            RpcErrorCode.SYNC_FFMPEG_MISSING,
            "ffmpeg is not available; install it or set a custom path in Settings",
        )
    return fallback


def _run_ffmpeg(
    *,
    ffmpeg_bin: Optional[str],
    source_wav: Path,
    output_wav: Path,
    plan: SyncPlan,
    out_sr: int,
    out_channels: int,
    cancel_event: Optional[threading.Event],
    timeout_secs: float,
) -> None:
    ff = _resolve_ffmpeg(ffmpeg_bin)
    filter_chain = _build_filter_chain(plan)

    cmd: list[str] = [
        ff,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
    ]
    if filter_chain:
        cmd += ["-af", filter_chain]
    cmd += [
        "-ar",
        str(int(out_sr)),
        "-ac",
        str(int(out_channels)),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise SyncApplyError(
            RpcErrorCode.SYNC_FFMPEG_MISSING,
            f"ffmpeg binary vanished: {ff}",
        ) from e
    except OSError as e:
        raise SyncApplyError(
            RpcErrorCode.SYNC_ENGINE_FAILURE,
            f"failed to spawn ffmpeg: {e}",
        ) from e

    stderr_bytes = b""
    try:
        while True:
            try:
                _, stderr_bytes = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    proc.kill()
                    proc.wait(timeout=2.0)
                    # Best-effort: remove the partial file so a cache
                    # scan doesn't mistake it for a completed WAV.
                    try:
                        output_wav.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise SyncApplyError(
                        RpcErrorCode.CANCELLED,
                        "sync cancelled by user",
                    )
                if timeout_secs > 0 and proc.poll() is None:
                    # Keep going; timeout is a soft ceiling per segment
                    # and communicate() will eventually raise again if
                    # we exceed it.
                    timeout_secs -= 0.25
                    if timeout_secs <= 0:
                        proc.kill()
                        proc.wait(timeout=2.0)
                        raise SyncApplyError(
                            RpcErrorCode.SYNC_ENGINE_FAILURE,
                            f"ffmpeg timed out after {timeout_secs:.1f}s (segment)",
                        )
    finally:
        if proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        tail = _tail(stderr_bytes)
        raise SyncApplyError(
            _classify_ffmpeg_error(tail),
            f"ffmpeg failed ({proc.returncode}): {tail}" if tail else "ffmpeg failed",
        )


def _build_filter_chain(plan: SyncPlan) -> str:
    """Compose the `-af` chain for a given plan."""
    parts: list[str] = []
    speed = plan.speed_factor
    if plan.status == SYNC_STATUS_ADJUSTED or (
        plan.status == SYNC_STATUS_TOO_LONG and abs(speed - 1.0) > 1e-4
    ):
        parts.extend(_atempo_chain(speed))
    if plan.status == SYNC_STATUS_FITS and plan.target_duration_secs > 0:
        parts.append(f"apad=whole_dur={plan.target_duration_secs:.6f}")
    elif plan.status == SYNC_STATUS_ADJUSTED and plan.target_duration_secs > 0:
        # atempo output may land a hair short of target; pad any residue.
        parts.append(f"apad=whole_dur={plan.target_duration_secs:.6f}")
    return ",".join(parts)


def _atempo_chain(speed: float) -> list[str]:
    """Emit one or more chained ``atempo=…`` filters.

    Phase 7 caps ``speed`` at 1.20 by default, so one atempo filter
    suffices — but we keep the multi-stage code path around in case a
    later version raises the cap and needs values outside FFmpeg's
    supported [0.5, 100.0] atempo range.
    """
    speed = max(0.5, min(100.0, float(speed)))
    if abs(speed - 1.0) < 1e-4:
        return []
    remaining = speed
    stages: list[str] = []
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.6f}")
    return stages


def _write_silence_wav(
    path: Path,
    *,
    duration_secs: float,
    sample_rate: int,
    channels: int,
) -> None:
    frames = max(0, int(round(duration_secs * sample_rate)))
    bytes_per_sample = 2  # PCM16
    silence = b"\x00" * (frames * channels * bytes_per_sample)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(channels))
        w.setsampwidth(bytes_per_sample)
        w.setframerate(int(sample_rate))
        w.writeframes(silence)


def _probe_source_rate(path: Path) -> int:
    try:
        _, sr, _ = probe_wav(path)
        return int(sr) or 22050
    except Exception:
        return 22050


def _classify_ffmpeg_error(stderr_tail: str) -> str:
    haystack = stderr_tail.lower()
    if "no space left on device" in haystack:
        return RpcErrorCode.SYNC_DISK_FULL
    if "permission denied" in haystack:
        return RpcErrorCode.SYNC_ENGINE_FAILURE
    if "invalid data found" in haystack or "could not find codec" in haystack:
        return RpcErrorCode.SYNC_SOURCE_INVALID
    return RpcErrorCode.SYNC_ENGINE_FAILURE


def _tail(stderr_bytes: bytes, *, max_lines: int = 6) -> str:
    try:
        text = stderr_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-max_lines:])
