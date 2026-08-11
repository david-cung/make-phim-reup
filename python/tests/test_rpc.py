"""Unit tests for the JSON-RPC dispatcher and Phase 1 handlers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Support running tests without an editable install by adding src/ to sys.path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker import handlers  # noqa: E402
from movie_translator_worker.errors import RpcError, RpcErrorCode  # noqa: E402
from movie_translator_worker.rpc import Dispatcher  # noqa: E402


@pytest.fixture
def dispatcher() -> Dispatcher:
    d = Dispatcher()
    handlers.install(d)
    return d


def _call(dispatcher: Dispatcher, method: str, params=None, *, id="1"):
    line = json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})
    _, frame = dispatcher.handle_line(line)
    assert frame is not None
    return frame


def test_ping(dispatcher: Dispatcher) -> None:
    frame = _call(dispatcher, "ping")
    assert frame["id"] == "1"
    assert frame["result"]["pong"] is True
    assert isinstance(frame["result"]["pid"], int)
    assert frame["result"]["uptimeMs"] >= 0


def test_env_info(dispatcher: Dispatcher) -> None:
    frame = _call(dispatcher, "env_info")
    r = frame["result"]
    assert "python" in r
    assert "platform" in r
    assert isinstance(r["cpuCount"], int) and r["cpuCount"] >= 1
    assert isinstance(r["ffmpegAvailable"], bool)


def test_initialize_updates_log_level(dispatcher: Dispatcher, monkeypatch) -> None:
    monkeypatch.delenv("LMT_WORKER_LOG_LEVEL", raising=False)
    frame = _call(
        dispatcher,
        "initialize",
        params={"app_version": "0.1.0", "log_level": "warn", "data_root": "/tmp/foo"},
    )
    assert frame["result"]["ok"] is True
    assert frame["result"]["pythonVersion"]
    import os as _os
    assert _os.environ["LMT_WORKER_LOG_LEVEL"] == "WARN"


def test_shutdown_returns_ok(dispatcher: Dispatcher) -> None:
    frame = _call(dispatcher, "shutdown")
    assert frame["result"] == {"ok": True}


def test_unknown_method(dispatcher: Dispatcher) -> None:
    frame = _call(dispatcher, "does_not_exist")
    assert frame["error"]["code"] == RpcErrorCode.METHOD_NOT_FOUND


def test_parse_error(dispatcher: Dispatcher) -> None:
    _, frame = dispatcher.handle_line("this is not JSON")
    assert frame is not None
    assert frame["error"]["code"] == RpcErrorCode.PARSE


def test_invalid_request_missing_method(dispatcher: Dispatcher) -> None:
    _, frame = dispatcher.handle_line(json.dumps({"jsonrpc": "2.0", "id": "1"}))
    assert frame is not None
    assert frame["error"]["code"] == RpcErrorCode.INVALID_REQUEST


def test_invalid_params_not_object(dispatcher: Dispatcher) -> None:
    _, frame = dispatcher.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "ping", "params": [1, 2]})
    )
    assert frame is not None
    assert frame["error"]["code"] == RpcErrorCode.INVALID_PARAMS


def test_wrong_jsonrpc_version(dispatcher: Dispatcher) -> None:
    _, frame = dispatcher.handle_line(
        json.dumps({"jsonrpc": "1.0", "id": "1", "method": "ping"})
    )
    assert frame is not None
    assert frame["error"]["code"] == RpcErrorCode.INVALID_REQUEST


def test_blank_line_returns_nothing(dispatcher: Dispatcher) -> None:
    method, frame = dispatcher.handle_line("   ")
    assert method is None and frame is None


def test_handler_raising_rpc_error_is_forwarded() -> None:
    d = Dispatcher()

    def bad(_params):
        raise RpcError("MY_CODE", "boom")

    d.register("bad", bad)
    _, frame = d.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "bad", "params": {}})
    )
    assert frame is not None
    assert frame["error"]["code"] == "MY_CODE"
    assert frame["error"]["message"] == "boom"


def test_handler_returning_non_dict_is_internal_error() -> None:
    d = Dispatcher()
    d.register("weird", lambda _p: 42)  # type: ignore[arg-type,return-value]
    _, frame = d.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "weird", "params": {}})
    )
    assert frame is not None
    assert frame["error"]["code"] == RpcErrorCode.INTERNAL


def test_duplicate_method_registration_raises() -> None:
    d = Dispatcher()
    d.register("x", lambda _p: {})
    with pytest.raises(ValueError):
        d.register("x", lambda _p: {})
