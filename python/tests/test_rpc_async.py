"""Tests for the async handler machinery added in Phase 3."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.errors import RpcErrorCode  # noqa: E402
from movie_translator_worker.rpc import (  # noqa: E402
    Dispatcher,
    HandlerContext,
    Request,
)


def _spawn(dispatcher: Dispatcher, method: str, params: dict, request_id: str) -> queue.Queue:
    out: queue.Queue = queue.Queue()
    dispatcher.spawn_async(
        Request(id=request_id, method=method, params=params), out,
    )
    return out


def _drain(out: queue.Queue, timeout: float = 2.0) -> list[dict]:
    frames: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = out.get(timeout=0.05)
        except queue.Empty:
            continue
        frames.append(item)
        if item.get("id") is not None:
            break
    return frames


def test_async_handler_emits_notifications_and_result() -> None:
    d = Dispatcher()

    def handler(params: dict, ctx: HandlerContext) -> dict:
        ctx.emit_progress("job.progress", {"fraction": 0.1})
        ctx.emit_progress("job.progress", {"fraction": 0.5})
        return {"ok": True, "n": int(params.get("n", 0))}

    d.register_async("run", handler)
    frames = _drain(_spawn(d, "run", {"n": 3}, "r-1"))
    notifs = [f for f in frames if "id" not in f or f.get("id") is None]
    finals = [f for f in frames if f.get("id") == "r-1"]
    assert len(notifs) == 2
    assert notifs[0]["method"] == "job.progress"
    assert notifs[0]["params"]["requestId"] == "r-1"
    assert notifs[0]["params"]["fraction"] == 0.1
    assert finals[-1]["result"] == {"ok": True, "n": 3}


def test_async_handler_cancellation() -> None:
    d = Dispatcher()
    started = threading.Event()

    def handler(_params: dict, ctx: HandlerContext) -> dict:
        started.set()
        for _ in range(100):
            if ctx.cancelled():
                from movie_translator_worker.errors import RpcError
                raise RpcError(RpcErrorCode.CANCELLED, "user cancelled")
            time.sleep(0.02)
        return {"ok": True}

    d.register_async("slow", handler)
    out = _spawn(d, "slow", {}, "cancel-me")
    assert started.wait(1.0)
    assert d.cancel("cancel-me") is True
    frames = _drain(out, timeout=2.0)
    finals = [f for f in frames if f.get("id") == "cancel-me"]
    assert finals[-1]["error"]["code"] == RpcErrorCode.CANCELLED


def test_cancel_unknown_returns_false() -> None:
    d = Dispatcher()
    assert d.cancel("nope") is False


def test_pending_count_tracks_active_jobs() -> None:
    d = Dispatcher()
    started = threading.Event()
    release = threading.Event()

    def handler(_params: dict, ctx: HandlerContext) -> dict:
        started.set()
        release.wait(2.0)
        return {"ok": True}

    d.register_async("wait", handler)
    out = _spawn(d, "wait", {}, "wait-1")
    assert started.wait(1.0)
    assert d.pending_count() == 1
    release.set()
    _drain(out, timeout=2.0)
    # Give the finally: block time to remove the entry.
    for _ in range(20):
        if d.pending_count() == 0:
            break
        time.sleep(0.05)
    assert d.pending_count() == 0


def test_async_handler_missing_id_is_error() -> None:
    d = Dispatcher()
    d.register_async("run", lambda p, ctx: {"ok": True})
    out: queue.Queue = queue.Queue()
    d.spawn_async(Request(id=None, method="run", params={}), out)
    frame = out.get(timeout=1.0)
    assert frame["error"]["code"] == RpcErrorCode.INVALID_REQUEST
