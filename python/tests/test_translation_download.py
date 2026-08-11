"""Phase 12 — auto-download handler in the Python worker.

Covers the two branches the Rust `TranslationService::download_model`
call site depends on: unknown preset → structured RPC error, and
already-installed file → idempotent success payload. The full
happy-path download talks to HuggingFace and is intentionally not
exercised here; it lives behind a network + several GB gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.errors import RpcError, RpcErrorCode  # noqa: E402
from movie_translator_worker.translation import handlers, registry  # noqa: E402


class _FakeCtx:
    """Minimal HandlerContext stand-in for the download handler.

    We only care that (a) `cancelled()` is honoured, and (b)
    `emit_progress` isn't crashing on unexpected shapes.
    """

    def __init__(self, *, cancelled: bool = False) -> None:
        self._cancelled = cancelled
        self.events: list[tuple[str, dict[str, Any]]] = []

    def cancelled(self) -> bool:
        return self._cancelled

    def emit_progress(self, method: str, params: dict[str, Any]) -> None:
        self.events.append((method, params))


def test_download_model_rejects_unknown_preset(tmp_path: Path) -> None:
    handlers.configure(models_root=tmp_path)
    with pytest.raises(RpcError) as exc:
        handlers.translate_download_model(
            {"preset": "totally-made-up"}, _FakeCtx()  # type: ignore[arg-type]
        )
    assert exc.value.code == RpcErrorCode.TRANSLATE_UNKNOWN_PRESET


def test_download_model_is_idempotent_when_file_exists(tmp_path: Path) -> None:
    handlers.configure(models_root=tmp_path)
    preset = registry.DEFAULT_RECOMMENDED_PRESET
    meta = registry.recommended_meta(preset)
    assert meta is not None
    filename = str(meta["filename"])

    # Seed the destination so the handler treats it as already
    # installed — this is the "user re-clicked Translate after a
    # successful download" branch.
    dest_dir = registry.ensure_models_dir(tmp_path)
    dest = dest_dir / filename
    dest.write_bytes(b"\x00" * 64)

    ctx = _FakeCtx()
    result = handlers.translate_download_model({"preset": preset}, ctx)  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["alreadyInstalled"] is True
    assert result["name"] == filename
    assert result["sizeBytes"] == 64
    # The 1.0 tick lets the UI close its download panel cleanly.
    assert any(
        method == "translate.download_progress" and params.get("fraction") == 1.0
        for method, params in ctx.events
    )
