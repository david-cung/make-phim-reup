"""End-to-end tests for the async STT handlers with a fake provider."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.errors import RpcErrorCode  # noqa: E402
from movie_translator_worker.rpc import Dispatcher  # noqa: E402
from movie_translator_worker.stt import handlers as stt_handlers  # noqa: E402
from movie_translator_worker.stt import registry  # noqa: E402
from movie_translator_worker.stt.models import Segment, TranscribeOptions  # noqa: E402
from movie_translator_worker.stt.provider import (  # noqa: E402
    ProviderCancelled,
    ProviderError,
    SpeechToTextProvider,
    TranscribeContext,
)


class _FakeProvider(SpeechToTextProvider):
    """Yields a small fixed set of segments, honouring cancel + progress."""

    name = "fake"

    def __init__(
        self,
        *,
        n_segments: int = 5,
        step_delay: float = 0.02,
        fail_with: ProviderError | None = None,
    ) -> None:
        self.n_segments = n_segments
        self.step_delay = step_delay
        self.fail_with = fail_with

    def transcribe(
        self,
        audio_path: str,
        options: TranscribeOptions,
        ctx: TranscribeContext,
    ) -> tuple[str, list[Segment]]:
        if self.fail_with is not None:
            raise self.fail_with
        ctx.on_progress(0.0, "loading_model", None)
        out: list[Segment] = []
        for i in range(self.n_segments):
            if ctx.cancelled():
                raise ProviderCancelled()
            time.sleep(self.step_delay)
            start = float(i)
            end = float(i) + 0.9
            out.append(Segment(id=i, start=start, end=end, text=f"seg-{i}"))
            ctx.on_progress((i + 1) / self.n_segments, "transcribing", None)
        return options.language or "en", out


@pytest.fixture()
def wired(tmp_path: Path):
    """Dispatcher wired with a fake provider + models_root."""
    d = Dispatcher()
    stt_handlers.configure(
        models_root=tmp_path / "models",
        provider=_FakeProvider(n_segments=3),
    )
    stt_handlers.install(d)
    return d


def _fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"\x00" * 128)
    return p


def _fake_model(tmp_path: Path, name: str) -> None:
    root = tmp_path / "models"
    d = registry.model_dir(root, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(b"\x00")
    (d / "config.json").write_text("{}")


def _drain(outbox: "queue.Queue", timeout: float = 5.0) -> list[dict]:
    frames = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        frames.append(item)
        # Stop once we see the final response for our request.
        if item.get("id") is not None:
            break
    return frames


def test_stt_env_lists_devices_and_installed_flag(tmp_path: Path, wired: Dispatcher) -> None:
    _, frame = wired.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "stt.env", "params": {}})
    )
    assert frame is not None
    r = frame["result"]
    assert "devices" in r and isinstance(r["devices"], list)
    assert any(d["kind"] == "cpu" for d in r["devices"])
    assert isinstance(r["whisperInstalled"], bool)


def test_stt_list_models_returns_all_known(tmp_path: Path, wired: Dispatcher) -> None:
    _, frame = wired.handle_line(
        json.dumps({"jsonrpc": "2.0", "id": "1", "method": "stt.list_models", "params": {}})
    )
    r = frame["result"]
    names = {m["name"] for m in r["models"]}
    assert {"small", "medium", "large-v3", "turbo"}.issubset(names)


def test_transcribe_end_to_end_emits_progress_and_result(
    tmp_path: Path, wired: Dispatcher,
) -> None:
    _fake_model(tmp_path, "small")
    audio = _fake_audio(tmp_path)

    outbox: queue.Queue = queue.Queue()
    req, _ = wired.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "stt.transcribe",
        "params": {
            "audioPath": str(audio),
            "audioHash": "sha256:test",
            "options": {"model": "small", "language": "en", "device": "cpu"},
        },
    }))
    assert req is not None
    wired.spawn_async(req, outbox)

    frames = _drain(outbox, timeout=3.0)
    assert frames, "no frames received"
    # progress notifications precede the final response
    notifications = [f for f in frames if "id" not in f or f.get("id") is None]
    responses = [f for f in frames if f.get("id") == "req-1"]
    assert notifications, "expected at least one progress notification"
    assert len(responses) == 1
    result = responses[0]["result"]
    assert result["language"] == "en"
    assert len(result["segments"]) == 3
    assert result["cacheKey"].startswith("sha256:")


def test_transcribe_requires_installed_model(tmp_path: Path, wired: Dispatcher) -> None:
    audio = _fake_audio(tmp_path)
    outbox: queue.Queue = queue.Queue()
    req, _ = wired.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "r",
        "method": "stt.transcribe",
        "params": {
            "audioPath": str(audio),
            "audioHash": "sha256:x",
            "options": {"model": "small"},
        },
    }))
    assert req is not None
    wired.spawn_async(req, outbox)
    frames = _drain(outbox)
    err = frames[-1]["error"]
    assert err["code"] == RpcErrorCode.STT_MODEL_NOT_INSTALLED


def test_transcribe_cancellation(tmp_path: Path) -> None:
    d = Dispatcher()
    stt_handlers.configure(
        models_root=tmp_path / "models",
        provider=_FakeProvider(n_segments=50, step_delay=0.05),
    )
    stt_handlers.install(d)
    _fake_model(tmp_path, "small")
    audio = _fake_audio(tmp_path)

    outbox: queue.Queue = queue.Queue()
    req, _ = d.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "cancel-me",
        "method": "stt.transcribe",
        "params": {
            "audioPath": str(audio),
            "audioHash": "sha256:x",
            "options": {"model": "small"},
        },
    }))
    assert req is not None
    d.spawn_async(req, outbox)

    time.sleep(0.15)
    assert d.cancel("cancel-me") is True

    frames = _drain(outbox, timeout=3.0)
    responses = [f for f in frames if f.get("id") == "cancel-me"]
    assert responses, "no final response after cancel"
    err = responses[-1].get("error")
    assert err is not None
    assert err["code"] == RpcErrorCode.CANCELLED


def test_transcribe_maps_provider_error(tmp_path: Path) -> None:
    d = Dispatcher()
    stt_handlers.configure(
        models_root=tmp_path / "models",
        provider=_FakeProvider(
            fail_with=ProviderError(
                RpcErrorCode.STT_OUT_OF_MEMORY, "too big for this box", recoverable=True,
            )
        ),
    )
    stt_handlers.install(d)
    _fake_model(tmp_path, "small")
    audio = _fake_audio(tmp_path)

    outbox: queue.Queue = queue.Queue()
    req, _ = d.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "oom",
        "method": "stt.transcribe",
        "params": {
            "audioPath": str(audio),
            "audioHash": "sha256:x",
            "options": {"model": "small"},
        },
    }))
    assert req is not None
    d.spawn_async(req, outbox)
    frames = _drain(outbox)
    err = frames[-1]["error"]
    assert err["code"] == RpcErrorCode.STT_OUT_OF_MEMORY
    assert err["data"] == {"recoverable": True, "model": "small"}


def test_download_model_uses_injected_downloader(tmp_path: Path, monkeypatch) -> None:
    d = Dispatcher()
    stt_handlers.configure(models_root=tmp_path / "models", provider=_FakeProvider())
    stt_handlers.install(d)

    calls: list[str] = []

    def _fake_download(repo_id: str, local_dir: Path, _progress) -> None:
        calls.append(repo_id)
        (Path(local_dir) / "model.bin").write_bytes(b"\x00")
        (Path(local_dir) / "config.json").write_text("{}")

    monkeypatch.setattr(stt_handlers, "_hf_download", _fake_download)

    outbox: queue.Queue = queue.Queue()
    req, _ = d.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "dl-1",
        "method": "stt.download_model",
        "params": {"name": "small"},
    }))
    assert req is not None
    d.spawn_async(req, outbox)
    frames = _drain(outbox)
    result = [f for f in frames if f.get("id") == "dl-1"][-1]["result"]
    assert result["ok"] is True
    assert result["alreadyInstalled"] is False
    assert calls == [registry.repo_for("small")]


def test_download_model_reports_already_installed(tmp_path: Path, monkeypatch) -> None:
    d = Dispatcher()
    stt_handlers.configure(models_root=tmp_path / "models", provider=_FakeProvider())
    stt_handlers.install(d)
    _fake_model(tmp_path, "small")

    def _boom(*_a, **_kw) -> None:
        raise AssertionError("must not download when already installed")

    monkeypatch.setattr(stt_handlers, "_hf_download", _boom)

    outbox: queue.Queue = queue.Queue()
    req, _ = d.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "dl-2",
        "method": "stt.download_model",
        "params": {"name": "small"},
    }))
    assert req is not None
    d.spawn_async(req, outbox)
    frames = _drain(outbox)
    result = [f for f in frames if f.get("id") == "dl-2"][-1]["result"]
    assert result["alreadyInstalled"] is True


def test_unknown_model_rejected(tmp_path: Path, wired: Dispatcher) -> None:
    audio = _fake_audio(tmp_path)
    outbox: queue.Queue = queue.Queue()
    req, _ = wired.parse(json.dumps({
        "jsonrpc": "2.0",
        "id": "?",
        "method": "stt.transcribe",
        "params": {
            "audioPath": str(audio),
            "audioHash": "sha256:x",
            "options": {"model": "no-such-model"},
        },
    }))
    assert req is not None
    wired.spawn_async(req, outbox)
    frames = _drain(outbox)
    err = frames[-1]["error"]
    assert err["code"] == RpcErrorCode.STT_UNKNOWN_MODEL
