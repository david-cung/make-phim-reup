"""RPC handlers for the Phase 7 sync subsystem.

Methods:

* ``sync.env`` — sync — reports whether FFmpeg is available in the
  worker environment. The Rust host uses the same info, so this is
  mostly a diagnostic surface.
* ``sync.apply_one`` — **async** — apply the timing plan to one TTS
  WAV and return the resulting file's metadata. Used by the
  "Preview Synced" flow.
* ``sync.apply_batch`` — **async, cancellable** — walk through a list
  of segments the host asked to (re)sync, emit a
  ``sync.segment_completed`` notification after each one so the host
  persists ``synced/sync.json`` incrementally, and a coarser
  ``sync.progress`` notification for the UI progress bar.

The host (Rust ``SyncService``) is authoritative for cache-hit
decisions: it filters segments already in-sync and only sends the
outstanding subset. The worker never guesses whether something is
stale.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcError, RpcErrorCode
from ..rpc import HandlerContext
from ..tts.wav_io import probe_wav
from .ffmpeg_apply import SyncApplyError, apply_plan
from .models import (
    BatchSegment,
    SyncSettings,
    build_sync_cache_key,
    validate_batch,
)
from .planner import plan_segment

_FFMPEG_BIN_OVERRIDE: Optional[str] = None


def configure(*, ffmpeg_bin: Optional[str] = None) -> None:
    """Wire up per-process state.

    ``ffmpeg_bin`` optionally pins the FFmpeg binary the host
    resolved. Passing ``None`` (the default) falls back to
    ``shutil.which("ffmpeg")`` at each invocation.
    """
    global _FFMPEG_BIN_OVERRIDE
    _FFMPEG_BIN_OVERRIDE = ffmpeg_bin


# ---------------------------------------------------------------- sync methods


def sync_env(_params: dict[str, Any]) -> dict[str, Any]:
    ff = _FFMPEG_BIN_OVERRIDE or shutil.which("ffmpeg")
    return {
        "ffmpegAvailable": ff is not None,
        "ffmpegPath": ff,
        "defaultMinSpeed": 0.85,
        "defaultMaxSpeed": 1.20,
    }


# --------------------------------------------------------------- async methods


def sync_apply_one(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    ffmpeg_bin = _optional_str(params, "ffmpegPath") or _FFMPEG_BIN_OVERRIDE
    settings = SyncSettings.from_dict(params.get("settings"))
    project_root = Path(_require_str(params, "projectRoot"))

    seg_payload = params.get("segment")
    if not isinstance(seg_payload, dict):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "segment must be an object")
    try:
        seg = BatchSegment.from_wire(seg_payload)
        validate_batch([seg])
    except ValueError as e:
        raise RpcError(RpcErrorCode.SYNC_INVALID_TIMING, str(e)) from e

    if ctx.cancelled():
        raise RpcError(RpcErrorCode.CANCELLED, "cancelled before start")

    src = _resolve(project_root, seg.source_file)
    dst = _resolve(project_root, seg.output_file)

    source_duration, source_sr, _ = _probe_source(src, seg.source_sample_rate)
    plan = plan_segment(
        target_duration_secs=seg.target_duration_secs,
        source_duration_secs=source_duration,
        settings=settings,
    )
    try:
        duration, sr, channels, size = apply_plan(
            ffmpeg_bin=ffmpeg_bin,
            source_wav=src,
            output_wav=dst,
            plan=plan,
            settings=settings,
            source_sample_rate=seg.source_sample_rate or source_sr,
            cancel_event=ctx.cancel_event,
        )
    except SyncApplyError as e:
        raise RpcError(
            e.code,
            e.message,
            data={
                "segmentId": seg.id,
                "sourceFile": str(src),
                "outputFile": str(dst),
                "recoverable": e.recoverable,
            },
        ) from e

    cache_key = build_sync_cache_key(
        tts_cache_key=seg.tts_cache_key,
        target_duration_secs=seg.target_duration_secs,
        settings=settings,
    )
    return _entry_payload(
        segment_id=seg.id,
        plan=plan,
        source_duration=source_duration,
        final_duration=duration,
        cache_key=cache_key,
        tts_cache_key=seg.tts_cache_key,
        file=seg.output_file,
        sample_rate=sr,
        channels=channels,
        size_bytes=size,
        settings=settings,
    )


def sync_apply_batch(
    params: dict[str, Any], ctx: HandlerContext
) -> dict[str, Any]:
    ffmpeg_bin = _optional_str(params, "ffmpegPath") or _FFMPEG_BIN_OVERRIDE
    settings = SyncSettings.from_dict(params.get("settings"))
    project_root = Path(_require_str(params, "projectRoot"))

    raw = params.get("segments")
    if not isinstance(raw, list):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "segments must be a list")
    if not raw:
        ctx.emit_progress(
            "sync.progress",
            {
                "stage": "completed",
                "fraction": 1.0,
                "completedSegments": 0,
                "totalSegments": 0,
            },
        )
        return {"ok": True, "totalSegments": 0, "generatedSegments": 0}

    try:
        segments = [BatchSegment.from_wire(s) for s in raw]
        validate_batch(segments)
    except ValueError as e:
        raise RpcError(RpcErrorCode.SYNC_INVALID_TIMING, str(e)) from e

    total = len(segments)
    generated = 0
    log.info(
        "sync batch start",
        segments=total,
        project_root=str(project_root),
        output_channels=settings.output_channels,
        output_sample_rate=settings.output_sample_rate,
    )

    ctx.emit_progress(
        "sync.progress",
        {
            "stage": "starting",
            "fraction": 0.0,
            "completedSegments": 0,
            "totalSegments": total,
        },
    )

    for idx, seg in enumerate(segments):
        if ctx.cancelled():
            raise RpcError(RpcErrorCode.CANCELLED, "sync cancelled by user")

        src = _resolve(project_root, seg.source_file)
        dst = _resolve(project_root, seg.output_file)

        source_duration, source_sr, _ = _probe_source(src, seg.source_sample_rate)
        plan = plan_segment(
            target_duration_secs=seg.target_duration_secs,
            source_duration_secs=source_duration,
            settings=settings,
        )

        ctx.emit_progress(
            "sync.progress",
            {
                "stage": "syncing",
                "fraction": idx / total,
                "completedSegments": generated,
                "totalSegments": total,
                "currentSegmentId": seg.id,
            },
        )

        try:
            duration, sr, channels, size = apply_plan(
                ffmpeg_bin=ffmpeg_bin,
                source_wav=src,
                output_wav=dst,
                plan=plan,
                settings=settings,
                source_sample_rate=seg.source_sample_rate or source_sr,
                cancel_event=ctx.cancel_event,
            )
        except SyncApplyError as e:
            log.warn(
                "sync segment failed",
                segment=seg.id,
                code=e.code,
                message=e.message,
            )
            raise RpcError(
                e.code,
                e.message,
                data={
                    "segmentId": seg.id,
                    "sourceFile": str(src),
                    "outputFile": str(dst),
                    "recoverable": e.recoverable,
                },
            ) from e

        cache_key = build_sync_cache_key(
            tts_cache_key=seg.tts_cache_key,
            target_duration_secs=seg.target_duration_secs,
            settings=settings,
        )
        generated += 1
        log.info(
            "sync segment complete",
            segment=seg.id,
            source=str(src),
            output=str(dst),
            target_start=round(seg.target_start, 3),
            target_duration_secs=round(seg.target_duration_secs, 3),
            final_duration_secs=round(duration, 3),
            speed_factor=round(plan.speed_factor, 4),
            size_bytes=size,
            sample_rate=sr,
            channels=channels,
        )
        payload = _entry_payload(
            segment_id=seg.id,
            plan=plan,
            source_duration=source_duration,
            final_duration=duration,
            cache_key=cache_key,
            tts_cache_key=seg.tts_cache_key,
            file=seg.output_file,
            sample_rate=sr,
            channels=channels,
            size_bytes=size,
            settings=settings,
        )
        payload.update(
            {
                "completedSegments": generated,
                "totalSegments": total,
                "fraction": generated / total,
            }
        )
        ctx.emit_progress("sync.segment_completed", payload)

    ctx.emit_progress(
        "sync.progress",
        {
            "stage": "completed",
            "fraction": 1.0,
            "completedSegments": generated,
            "totalSegments": total,
        },
    )
    log.info(
        "sync batch complete",
        generated_segments=generated,
        total_segments=total,
    )

    return {
        "ok": True,
        "totalSegments": total,
        "generatedSegments": generated,
    }


# ----------------------------------------------------------------- registration


def install(dispatcher) -> None:
    dispatcher.register("sync.env", sync_env)
    dispatcher.register_async("sync.apply_one", sync_apply_one)
    dispatcher.register_async("sync.apply_batch", sync_apply_batch)


# ------------------------------------------------------------------- helpers


def _entry_payload(
    *,
    segment_id: int,
    plan,
    source_duration: float,
    final_duration: float,
    cache_key: str,
    tts_cache_key: str,
    file: str,
    sample_rate: int,
    channels: int,
    size_bytes: int,
    settings: SyncSettings,
) -> dict[str, Any]:
    return {
        "segmentId": segment_id,
        "status": plan.status,
        "targetDurationSecs": plan.target_duration_secs,
        "originalDurationSecs": source_duration,
        "finalDurationSecs": final_duration,
        "speedFactor": plan.speed_factor,
        "cacheKey": cache_key,
        "ttsCacheKey": tts_cache_key,
        "file": file,
        "sampleRate": int(sample_rate),
        "channels": int(channels),
        "sizeBytes": int(size_bytes),
        "settings": settings.normalised().to_dict(),
    }


def _resolve(project_root: Path, relative: str) -> Path:
    # Reject path escapes — the host is responsible for feeding us
    # in-project paths, but we double-check.
    p = (project_root / relative).resolve()
    root = project_root.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise RpcError(
            RpcErrorCode.INVALID_PARAMS,
            f"path escapes project root: {relative}",
        ) from e
    return p


def _probe_source(path: Path, hint_sr: Optional[int]) -> tuple[float, int, int]:
    """Return ``(duration_secs, sample_rate, channels)`` for the source
    TTS WAV. Falls back to a sensible default when the file is unreadable
    so the planner can still classify the segment as ``too_long``."""
    try:
        d, sr, ch = probe_wav(path)
        return (float(d), int(sr) or (hint_sr or 22050), int(ch) or 1)
    except FileNotFoundError as e:
        raise RpcError(
            RpcErrorCode.SYNC_SOURCE_MISSING,
            f"source TTS wav is missing: {path}",
        ) from e
    except Exception as e:
        raise RpcError(
            RpcErrorCode.SYNC_SOURCE_INVALID,
            f"cannot read source TTS wav {path}: {e}",
        ) from e


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RpcError(RpcErrorCode.INVALID_PARAMS, f"missing required string param: {key}")
    return value


def _optional_str(params: dict[str, Any], key: str) -> Optional[str]:
    value = params.get(key)
    if isinstance(value, str) and value:
        return value
    return None
