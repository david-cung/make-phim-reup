"""Local hardware probe used to decide whether Whisper large-v3 can run.

Nothing here touches the network. Figures are conservative: loading
large-v3 and then transcribing a feature-length movie needs more RAM
than the on-disk snapshot size.
"""

from __future__ import annotations

import platform
from typing import Optional

from .device import default_device, detect_devices

# CTranslate2 large-v3 int8 working set on CPU is typically 4–6 GB plus
# OS/headroom. float16 on CUDA wants ~4 GB of VRAM.
LARGE_V3_CPU_RAM_GB = 8.0
LARGE_V3_CUDA_VRAM_GB = 4.0
LARGE_V3_TOTAL_RAM_GB = 12.0

PROFILE_MODELS = {
    "fast": "small",
    "balanced": "medium",
    "quality": "large-v3",
}


def memory_stats() -> dict:
    total, available = _physical_memory_bytes()
    return {
        "ramTotalGb": _bytes_to_gb(total),
        "ramAvailableGb": _bytes_to_gb(available),
        "os": platform.system(),
        "arch": platform.machine(),
    }


def large_v3_capability() -> dict:
    """Describe whether QUALITY / large-v3 can run on this machine.

    ``canRun`` is True when we believe inference will start. It is NOT
    a guarantee the whole movie will finish without swapping.
    """
    mem = memory_stats()
    devices = detect_devices()
    device = default_device(devices)
    cuda = next((d for d in devices if d.kind == "cuda" and d.supported), None)
    vram_gb = _cuda_vram_gb() if cuda is not None else None

    if cuda is not None and (vram_gb is None or vram_gb >= LARGE_V3_CUDA_VRAM_GB):
        return {
            "canRun": True,
            "model": "large-v3",
            "device": "cuda",
            "computeType": "float16",
            "fallbackModel": "medium",
            "reason": None,
            "warning": (
                "Whisper large-v3 provides higher transcription quality but "
                "requires significantly more system resources."
            ),
            **mem,
            "vramGb": vram_gb,
        }

    ram_ok = (mem["ramAvailableGb"] or 0) >= LARGE_V3_CPU_RAM_GB or (
        mem["ramTotalGb"] or 0
    ) >= LARGE_V3_TOTAL_RAM_GB
    if ram_ok:
        return {
            "canRun": True,
            "model": "large-v3",
            "device": device,
            "computeType": "int8",
            "fallbackModel": "medium",
            "reason": None,
            "warning": (
                "Whisper large-v3 provides higher transcription quality but "
                "requires significantly more system resources. CPU inference "
                "will be slow on a long movie."
            ),
            **mem,
            "vramGb": vram_gb,
        }

    return {
        "canRun": False,
        "model": "large-v3",
        "device": device,
        "computeType": "int8",
        "fallbackModel": "medium",
        "reason": (
            "Whisper large-v3 cannot run with the current hardware "
            "configuration. Use Balanced (Whisper medium) instead."
        ),
        "warning": (
            "Whisper large-v3 provides higher transcription quality but "
            "requires significantly more system resources."
        ),
        **mem,
        "vramGb": vram_gb,
    }


def profile_params(profile: str) -> dict:
    """Default inference knobs for FAST / BALANCED / QUALITY."""
    name = (profile or "balanced").lower()
    if name not in PROFILE_MODELS:
        name = "balanced"
    model = PROFILE_MODELS[name]
    if name == "fast":
        return {
            "profile": name,
            "model": model,
            "beamSize": 1,
            "wordTimestamps": True,
            "vadFilter": True,
            "temperature": 0.0,
        }
    if name == "quality":
        return {
            "profile": name,
            "model": model,
            "beamSize": 8,
            "wordTimestamps": True,
            "vadFilter": True,
            "temperature": 0.0,
        }
    return {
        "profile": name,
        "model": model,
        "beamSize": 5,
        "wordTimestamps": True,
        "vadFilter": True,
        "temperature": 0.0,
    }


def _bytes_to_gb(value: Optional[int]) -> Optional[float]:
    if not value:
        return None
    return round(value / (1024.0 ** 3), 2)


def _physical_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    if platform.system() == "Windows":
        return _windows_memory()
    try:
        page = int(getattr(__import__("os"), "sysconf")("SC_PAGE_SIZE"))
        total = page * int(getattr(__import__("os"), "sysconf")("SC_PHYS_PAGES"))
        available = None
        try:
            available = page * int(
                getattr(__import__("os"), "sysconf")("SC_AVPHYS_PAGES")
            )
        except (ValueError, OSError, TypeError):
            available = total
        return total, available
    except Exception:
        return None, None


def _windows_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
            return None, None
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception:
        return None, None


def _cuda_vram_gb() -> Optional[float]:
    try:
        import ctranslate2  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        return None
    if count <= 0:
        return None
    # CTranslate2 does not always expose per-device VRAM. Presence of a
    # CUDA device is treated as "likely enough" unless a later load fails.
    return None
