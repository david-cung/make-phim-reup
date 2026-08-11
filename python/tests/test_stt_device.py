"""Device detection tests. These don't require ctranslate2/faster-whisper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.stt import device  # noqa: E402


def test_detect_devices_always_contains_cpu() -> None:
    devs = device.detect_devices()
    kinds = [d.kind for d in devs]
    assert "cpu" in kinds
    cpu = next(d for d in devs if d.kind == "cpu")
    assert cpu.supported is True


def test_default_device_prefers_cuda_when_available() -> None:
    from movie_translator_worker.stt.device import DeviceInfo, default_device
    with_cuda = [
        DeviceInfo(kind="cpu", label="CPU", supported=True),
        DeviceInfo(kind="cuda", label="CUDA", supported=True, count=1),
    ]
    assert default_device(with_cuda) == "cuda"


def test_default_device_falls_back_to_cpu() -> None:
    from movie_translator_worker.stt.device import DeviceInfo, default_device
    only_cpu = [DeviceInfo(kind="cpu", label="CPU", supported=True)]
    assert default_device(only_cpu) == "cpu"


def test_default_device_skips_unsupported_metal() -> None:
    from movie_translator_worker.stt.device import DeviceInfo, default_device
    metal_but_unsupported = [
        DeviceInfo(kind="cpu", label="CPU", supported=True),
        DeviceInfo(kind="metal", label="Metal", supported=False),
    ]
    assert default_device(metal_but_unsupported) == "cpu"
