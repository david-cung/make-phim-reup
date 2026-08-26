"""Phase 12 — TTS voice registry + auto-download handler tests.

Locks the camelCase wire shape the Rust host relies on
(`tts::models::RecommendedVoicePreset`) and covers the two branches
`TtsService::download_voice` will exercise without network:
`TTS_UNKNOWN_PRESET` for a bad preset name, and the idempotent
"already installed" fast-path when both voice files are already on
disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.errors import RpcError, RpcErrorCode  # noqa: E402
from movie_translator_worker.tts import handlers, registry  # noqa: E402


class _FakeCtx:
    def __init__(self, *, cancelled: bool = False) -> None:
        self._cancelled = cancelled
        self.events: list[tuple[str, dict[str, Any]]] = []

    def cancelled(self) -> bool:
        return self._cancelled

    def emit_progress(self, method: str, params: dict[str, Any]) -> None:
        self.events.append((method, params))


def test_recommended_voices_shape_matches_rust() -> None:
    presets = registry.recommended_voices()
    assert presets, "must ship at least one preset"
    for p in presets:
        assert {
            "preset",
            "engine",
            "voiceId",
            "language",
            "targetLanguages",
            "quality",
            "approxSizeBytes",
            "label",
            "isDefault",
        }.issubset(set(p.keys()))
        assert isinstance(p["approxSizeBytes"], int)
        assert isinstance(p["isDefault"], bool)
        assert isinstance(p["targetLanguages"], list)
        assert p["engine"] in {"piper", registry.F5_ENGINE}


def test_exactly_one_default_recommended_voice() -> None:
    defaults = [p for p in registry.recommended_voices() if p["isDefault"]]
    assert len(defaults) == 1
    assert defaults[0]["preset"] == registry.DEFAULT_RECOMMENDED_VOICE_PRESET


def test_vietnamese_target_has_a_preset() -> None:
    # The whole reason for auto-download: users translating INTO
    # Vietnamese (the app's primary use case) should get a voice
    # without touching the filesystem. Losing this coverage would
    # regress the "click Translate → click Generate all" flow.
    presets = registry.recommended_voices()
    assert any("vi" in p["targetLanguages"] for p in presets)


def test_download_voice_rejects_unknown_preset(tmp_path: Path) -> None:
    handlers.configure(models_root=tmp_path)
    with pytest.raises(RpcError) as exc:
        handlers.tts_download_voice(
            {"preset": "not-a-real-voice"}, _FakeCtx()  # type: ignore[arg-type]
        )
    assert exc.value.code == RpcErrorCode.TTS_UNKNOWN_PRESET


def test_download_voice_idempotent_when_files_exist(tmp_path: Path) -> None:
    handlers.configure(models_root=tmp_path)
    preset = registry.DEFAULT_RECOMMENDED_VOICE_PRESET
    meta = registry.recommended_voice_meta(preset)
    assert meta is not None
    voice_id = str(meta["voice_id"])

    voice_dir = registry.engine_root(tmp_path, "piper") / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx = voice_dir / f"{voice_id}.onnx"
    cfg = voice_dir / f"{voice_id}.onnx.json"
    onnx.write_bytes(b"\x00" * 128)
    cfg.write_text('{"audio": {"sample_rate": 22050}, "language": {"code": "vi_VN"}}')

    ctx = _FakeCtx()
    result = handlers.tts_download_voice({"preset": preset}, ctx)  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["alreadyInstalled"] is True
    assert result["voiceId"] == voice_id
    assert result["sizeBytes"] == 128
    assert any(
        method == "tts.download_progress" and params.get("fraction") == 1.0
        for method, params in ctx.events
    )
