"""Hardware capability reporting for the optional F5-TTS engine."""

from __future__ import annotations

import platform
from typing import Any

from ..stt.hardware import memory_stats

F5_RECOMMENDED_RAM_GB = 16.0
F5_RECOMMENDED_VRAM_GB = 8.0


def f5_hardware_capability() -> dict[str, Any]:
    memory = memory_stats()
    cuda_available = False
    mps_available = False
    gpu_name = None
    vram_gb = None
    runtime_error = None
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(float(props.total_memory) / (1024.0**3), 2)
    except Exception as exc:
        runtime_error = str(exc)

    total_ram = float(memory.get("ramTotalGb") or 0.0)
    available_ram = float(memory.get("ramAvailableGb") or 0.0)
    gpu_recommended = cuda_available and (
        vram_gb is None or vram_gb >= F5_RECOMMENDED_VRAM_GB
    )
    ram_recommended = (
        total_ram >= F5_RECOMMENDED_RAM_GB and available_ram >= 8.0
    )
    recommended = bool(gpu_recommended and ram_recommended)
    backend = "cuda" if cuda_available else "mps" if mps_available else "cpu"

    if recommended:
        warning = None
    elif cuda_available:
        warning = (
            "F5-TTS QUALITY mode is available, but at least 8 GB VRAM and "
            "16 GB system RAM are recommended. Generation may be slow or run out of memory."
        )
    else:
        warning = (
            "F5-TTS QUALITY mode can run on CPU/MPS but is designed for a capable GPU. "
            "Generation on this machine may be very slow and requires substantial RAM."
        )

    return {
        "backend": backend,
        "cudaAvailable": cuda_available,
        "mpsAvailable": mps_available,
        "gpuName": gpu_name,
        "vramGb": vram_gb,
        "ramTotalGb": memory.get("ramTotalGb"),
        "ramAvailableGb": memory.get("ramAvailableGb"),
        "os": platform.system(),
        "recommended": recommended,
        "warning": warning,
        "runtimeError": runtime_error,
    }
