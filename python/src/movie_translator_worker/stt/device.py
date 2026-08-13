"""Detect the compute backends available for Whisper inference.

CTranslate2 (faster-whisper's backend) supports ``cpu`` and ``cuda``
today. Apple Silicon inference still runs on the ``cpu`` device but
uses ARM-optimised routines. We surface ``metal`` as an *architectural*
placeholder for future Metal support so the UI and cache key can
already carry it.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # cpu | cuda | metal
    label: str
    supported: bool
    count: int = 1
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def detect_devices() -> list[DeviceInfo]:
    """Return the list of devices we can actually run Whisper on.

    We always report ``cpu``. ``cuda`` is only reported when
    CTranslate2 says a compatible device exists. ``metal`` is reported
    as *unsupported* on Apple Silicon so the UI can indicate the
    future capability without letting the user pick it today.
    """
    return list(_detect_devices_cached())


@lru_cache(maxsize=1)
def _detect_devices_cached() -> tuple[DeviceInfo, ...]:
    devices: list[DeviceInfo] = [
        DeviceInfo(kind="cpu", label="CPU", supported=True, detail=platform.machine()),
    ]

    cuda_count = _cuda_device_count()
    if cuda_count > 0:
        devices.append(
            DeviceInfo(kind="cuda", label="CUDA", supported=True, count=cuda_count),
        )

    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        devices.append(
            DeviceInfo(
                kind="metal",
                label="Apple Silicon (Metal)",
                supported=False,
                detail="planned; falls back to CPU today",
            ),
        )

    return tuple(devices)


def default_device(devices: Optional[list[DeviceInfo]] = None) -> str:
    devs = devices if devices is not None else detect_devices()
    # Prefer CUDA when present, otherwise CPU. Never default to a
    # device we know is not supported.
    for d in devs:
        if d.kind == "cuda" and d.supported:
            return "cuda"
    return "cpu"


def _cuda_device_count() -> int:
    try:
        import ctranslate2  # type: ignore[import-not-found]
    except Exception:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0
