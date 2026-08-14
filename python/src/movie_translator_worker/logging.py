"""Tiny JSON logger to stderr.

We intentionally avoid the `logging` module's YAML/dict config surface —
this keeps the worker startup fast and dependency-free. Each record is a
single JSON object followed by a newline.

Log level is read from `LMT_WORKER_LOG_LEVEL` (default: `INFO`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

_LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_DEFAULT = "INFO"


def _current_level() -> int:
    name = os.environ.get("LMT_WORKER_LOG_LEVEL", _DEFAULT).upper()
    return _LEVELS.get(name, _LEVELS[_DEFAULT])


_LEVEL = _current_level()


def _log(level: str, msg: str, **fields: Any) -> None:
    if _LEVELS.get(level, 100) < _LEVEL:
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level.lower(),
        "component": "worker",
        "msg": msg,
    }
    if fields:
        record.update(fields)
    try:
        sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        # Never let logging kill the worker.
        pass


def trace(msg: str, **fields: Any) -> None:
    _log("TRACE", msg, **fields)


def debug(msg: str, **fields: Any) -> None:
    _log("DEBUG", msg, **fields)


def info(msg: str, **fields: Any) -> None:
    _log("INFO", msg, **fields)


def warn(msg: str, **fields: Any) -> None:
    _log("WARN", msg, **fields)


def error(msg: str, **fields: Any) -> None:
    _log("ERROR", msg, **fields)


def reload_level() -> None:
    """Re-read `LMT_WORKER_LOG_LEVEL`. Called by `handlers.initialize`."""
    global _LEVEL
    _LEVEL = _current_level()
