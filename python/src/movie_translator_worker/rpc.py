"""JSON-RPC 2.0 framing + dispatcher.

Wire format: one JSON object per line on stdin/stdout. Nothing else is
allowed on stdout — all logs and warnings go to stderr.

Two flavours of handler are supported:

* **Sync** handlers execute on the main serve thread and return a dict.
  Perfect for short, cheap methods (`ping`, `env_info`, `list_models`).

* **Async** handlers run in a background daemon thread. They receive a
  :class:`HandlerContext` whose ``cancel_event`` fires when a
  ``jsonrpc://cancel`` request arrives, and whose ``emit_progress``
  posts a *notification* frame (a JSON-RPC frame without an ``id``)
  back on stdout. The Rust side routes notifications to per-request
  progress callbacks.

The serve loop stays single-threaded on the read side and delegates all
stdout writes to a dedicated writer thread; that keeps interleaving
safe without any per-frame locking.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from . import logging as log
from .errors import RpcError, RpcErrorCode

SyncHandler = Callable[[dict[str, Any]], dict[str, Any]]
AsyncHandler = Callable[[dict[str, Any], "HandlerContext"], dict[str, Any]]

# Phase 11 — cap progress notifications at ~20 Hz per method so long
# runs (Whisper on a 2h movie, translation over 3k segments) don't
# flood the stdio bus. `0.0` and `1.0` bookends always go through so
# the UI never gets stuck part-way through a bar. Callers that need
# a different cadence can pass ``throttle_ms=None`` to opt out.
PROGRESS_THROTTLE_MS = 50


@dataclass
class Request:
    id: Optional[str]
    method: str
    params: dict[str, Any]


@dataclass
class HandlerContext:
    """Per-async-request context passed to :class:`AsyncHandler`\\ s."""

    request_id: str
    cancel_event: threading.Event
    outbox: "queue.Queue[Optional[dict[str, Any]]]"
    # Phase 11 — per-method {last_emit_ms, last_fraction}. Never
    # leaves the context so threading is not a concern (contexts are
    # per-request).
    _last_emit_ms: dict[str, float] = field(default_factory=dict)
    _last_fraction: dict[str, float] = field(default_factory=dict)

    def emit_progress(
        self,
        method: str,
        params: dict[str, Any],
        *,
        throttle_ms: Optional[int] = PROGRESS_THROTTLE_MS,
    ) -> None:
        """Send a JSON-RPC notification frame back to the host.

        The frame carries ``params["requestId"]`` automatically so the
        host can correlate the progress with the originating request.

        ``throttle_ms`` (Phase 11) coalesces frequent notifications so
        Whisper's inner-loop callbacks and llama-cpp's per-token
        callbacks don't overwhelm the stdio bus. Set to ``None`` to
        opt out. Fractions ``<= 0.0`` and ``>= 1.0`` always pass so
        the UI observes both endpoints of the progress bar.
        """
        # Throttling only applies to fine-grained progress bars —
        # per-segment `*.segment_completed` / `*.chunk_completed`
        # notifications are semantic events (they drive incremental
        # UI refreshes on the frontend) so they always go through.
        if (
            throttle_ms is not None
            and throttle_ms > 0
            and method.endswith(".progress")
        ):
            frac = params.get("fraction")
            if isinstance(frac, (int, float)):
                edge = frac <= 0.0 or frac >= 1.0
                now = time.monotonic() * 1000.0
                last_ms = self._last_emit_ms.get(method)
                last_frac = self._last_fraction.get(method)
                # Skip only when *both* time-since-last is small AND
                # the fraction has barely moved. That preserves the
                # semantic that a big jump (e.g. Whisper finishing a
                # long segment) always reaches the UI, while an
                # inner-loop dribble every few ms gets coalesced.
                if (
                    not edge
                    and last_ms is not None
                    and (now - last_ms) < float(throttle_ms)
                    and last_frac is not None
                    and abs(float(frac) - last_frac) < 0.01
                ):
                    return
                self._last_emit_ms[method] = now
                self._last_fraction[method] = float(frac)

        payload = {"jsonrpc": "2.0", "method": method, "params": {"requestId": self.request_id, **params}}
        self.outbox.put(payload)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


@dataclass
class _PendingJob:
    request_id: str
    method: str
    thread: threading.Thread
    cancel_event: threading.Event = field(default_factory=threading.Event)


class Dispatcher:
    """Method registry + one-shot request handling."""

    def __init__(self) -> None:
        self._sync: dict[str, SyncHandler] = {}
        self._async: dict[str, AsyncHandler] = {}
        self._pending: Dict[str, _PendingJob] = {}
        self._pending_lock = threading.Lock()

    def register(self, method: str, handler: SyncHandler) -> None:
        self._require_free(method)
        self._sync[method] = handler

    def register_async(self, method: str, handler: AsyncHandler) -> None:
        self._require_free(method)
        self._async[method] = handler

    def _require_free(self, method: str) -> None:
        if method in self._sync or method in self._async:
            raise ValueError(f"method already registered: {method}")

    def has(self, method: str) -> bool:
        return method in self._sync or method in self._async

    def is_async(self, method: str) -> bool:
        return method in self._async

    # ------------------------------------------------------------------ sync

    def handle_sync(self, req: Request) -> dict[str, Any]:
        handler = self._sync.get(req.method)
        if handler is None:
            return self._error_frame(
                req.id,
                RpcError(RpcErrorCode.METHOD_NOT_FOUND, f"unknown method: {req.method}"),
            )
        try:
            result = handler(req.params)
        except RpcError as e:
            return self._error_frame(req.id, e)
        except Exception as e:  # pragma: no cover - safety net
            log.error(
                "handler crashed",
                method=req.method,
                error=str(e),
                tb=traceback.format_exc(),
            )
            return self._error_frame(
                req.id,
                RpcError(RpcErrorCode.INTERNAL, f"{type(e).__name__}: {e}"),
            )
        if not isinstance(result, dict):
            return self._error_frame(
                req.id,
                RpcError(RpcErrorCode.INTERNAL, f"handler for {req.method} did not return a dict"),
            )
        return self._success_frame(req.id, result)

    # ---------------------------------------------------------------- async

    def spawn_async(self, req: Request, outbox: "queue.Queue[Optional[dict[str, Any]]]") -> None:
        """Start an async handler in a daemon thread.

        The thread writes its final success/error frame to ``outbox``
        when it completes, and can post progress notifications along
        the way via :class:`HandlerContext`.
        """
        handler = self._async.get(req.method)
        if handler is None:
            outbox.put(self._error_frame(
                req.id,
                RpcError(RpcErrorCode.METHOD_NOT_FOUND, f"unknown async method: {req.method}"),
            ))
            return
        if req.id is None:
            outbox.put(self._error_frame(
                None,
                RpcError(RpcErrorCode.INVALID_REQUEST, "async methods require an id"),
            ))
            return

        cancel = threading.Event()
        ctx = HandlerContext(request_id=req.id, cancel_event=cancel, outbox=outbox)

        def _run() -> None:
            try:
                result = handler(req.params, ctx)
            except RpcError as e:
                outbox.put(self._error_frame(req.id, e))
            except Exception as e:  # pragma: no cover - safety net
                log.error(
                    "async handler crashed",
                    method=req.method,
                    error=str(e),
                    tb=traceback.format_exc(),
                )
                outbox.put(self._error_frame(
                    req.id, RpcError(RpcErrorCode.INTERNAL, f"{type(e).__name__}: {e}"),
                ))
            else:
                if not isinstance(result, dict):
                    outbox.put(self._error_frame(
                        req.id,
                        RpcError(RpcErrorCode.INTERNAL, f"handler for {req.method} did not return a dict"),
                    ))
                else:
                    outbox.put(self._success_frame(req.id, result))
            finally:
                with self._pending_lock:
                    self._pending.pop(req.id or "", None)  # type: ignore[arg-type]

        thread = threading.Thread(target=_run, name=f"stt-{req.method}-{req.id}", daemon=True)
        with self._pending_lock:
            self._pending[req.id] = _PendingJob(
                request_id=req.id, method=req.method, thread=thread, cancel_event=cancel,
            )
        thread.start()

    def cancel(self, request_id: str) -> bool:
        """Signal cooperative cancellation for a pending async request.

        Returns True if the request was known (and its event set), False
        if it had already completed or was never registered.
        """
        with self._pending_lock:
            job = self._pending.get(request_id)
            if job is None:
                return False
            job.cancel_event.set()
        return True

    def pending_ids(self) -> list[str]:
        with self._pending_lock:
            return list(self._pending.keys())

    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    # ---------------------------------------------------------------- parse

    def handle_line(self, line: str) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
        """Convenience helper used by tests: parse a raw line and, if it
        maps to a *sync* handler, run it and return the resulting frame
        in one shot.

        Returns ``(method, frame)``. ``method`` is None when the line
        was a parse error or a blank line; ``frame`` is None only for
        blank/whitespace lines.

        Async handlers are not supported here — they need the full
        :func:`serve` loop with an outbox. Use :meth:`parse` +
        :meth:`spawn_async` in that case.
        """
        req, frame = self.parse(line)
        if frame is not None:
            return (None, frame)
        if req is None:
            return (None, None)
        if req.method in self._async:
            return (req.method, self._error_frame(
                req.id,
                RpcError(
                    RpcErrorCode.INTERNAL,
                    f"method {req.method} is async; use spawn_async",
                ),
            ))
        return (req.method, self.handle_sync(req))

    def parse(self, line: str) -> Tuple[Optional[Request], Optional[dict[str, Any]]]:
        """Parse a single stdin line.

        Returns ``(request, frame)``. Exactly one is non-None:
          * ``request`` when the line was a well-formed request that
            should be dispatched.
          * ``frame`` when the line was a parse error we should respond
            with immediately.
        A blank line returns ``(None, None)``.
        """
        line = line.strip()
        if not line:
            return (None, None)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            return (None, self._error_frame(None, RpcError(RpcErrorCode.PARSE, f"invalid JSON: {e}")))
        try:
            req = _parse_request(payload)
        except RpcError as e:
            return (None, self._error_frame(
                payload.get("id") if isinstance(payload, dict) else None, e,
            ))
        return (req, None)

    # ---------------------------------------------------------------- frames

    @staticmethod
    def _success_frame(request_id: Optional[str], result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error_frame(request_id: Optional[str], err: RpcError) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": err.to_dict()}


def _parse_request(payload: Any) -> Request:
    if not isinstance(payload, dict):
        raise RpcError(RpcErrorCode.INVALID_REQUEST, "request must be a JSON object")
    if payload.get("jsonrpc") != "2.0":
        raise RpcError(RpcErrorCode.INVALID_REQUEST, "jsonrpc must be '2.0'")
    method = payload.get("method")
    if not isinstance(method, str):
        raise RpcError(RpcErrorCode.INVALID_REQUEST, "method must be a string")
    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise RpcError(RpcErrorCode.INVALID_PARAMS, "params must be an object")
    return Request(id=payload.get("id"), method=method, params=params)


# ---------------------------------------------------------------------- serve


def serve(dispatcher: Dispatcher, *, stop_on: str = "shutdown") -> None:
    """Blocking main loop. Reads stdin line by line, writes to stdout via
    a dedicated writer thread so async handlers can interleave progress
    frames without racing the read side.
    """
    log.info("worker loop started", pid=os.getpid())
    stdin = sys.stdin
    stdout = sys.stdout

    outbox: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue()
    writer = threading.Thread(
        target=_writer_loop, args=(outbox, stdout), name="rpc-writer", daemon=True,
    )
    writer.start()

    stop_requested = False
    try:
        for line in stdin:
            req, frame = dispatcher.parse(line)
            if frame is not None:
                outbox.put(frame)
                continue
            if req is None:
                continue

            # Cancellation is a first-class control message.
            if req.method == "jsonrpc://cancel":
                target = req.params.get("requestId")
                cancelled = dispatcher.cancel(str(target)) if target else False
                outbox.put({
                    "jsonrpc": "2.0",
                    "id": req.id,
                    "result": {"cancelled": cancelled, "requestId": target},
                })
                continue

            if dispatcher.is_async(req.method):
                dispatcher.spawn_async(req, outbox)
                continue

            frame = dispatcher.handle_sync(req)
            outbox.put(frame)
            if req.method == stop_on and frame.get("result") is not None:
                log.info("shutdown requested; exiting cleanly")
                stop_requested = True
                break
    finally:
        # Signal every in-flight async job to bail out.
        for pid in dispatcher.pending_ids():
            dispatcher.cancel(pid)
        # Drain the outbox by asking the writer to stop.
        outbox.put(None)
        writer.join(timeout=2.0)
        if not stop_requested:
            log.info("stdin closed; exiting")


def _writer_loop(
    outbox: "queue.Queue[Optional[dict[str, Any]]]", stdout: Any,
) -> None:
    while True:
        item = outbox.get()
        if item is None:
            return
        try:
            stdout.write(json.dumps(item, ensure_ascii=False) + "\n")
            stdout.flush()
        except BrokenPipeError:
            log.warn("stdout closed; writer exiting")
            return
        except Exception as e:  # pragma: no cover - safety net
            log.error("writer failed", error=str(e))
            return
