"""Stable error codes returned to the Rust host.

Codes are UPPER_SNAKE and prefixed by subsystem. The Rust host maps a few
of these to structured `AppError`s for the UI; everything else is surfaced
verbatim in the worker log and as a generic "worker error" toast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RpcErrorCode:
    PARSE = "E_PARSE"
    INVALID_REQUEST = "E_INVALID_REQUEST"
    METHOD_NOT_FOUND = "E_METHOD"
    INVALID_PARAMS = "E_INVALID_PARAMS"
    INTERNAL = "E_INTERNAL"
    CANCELLED = "E_CANCELLED"

    # -------------------- STT (Phase 3) --------------------
    STT_WHISPER_NOT_INSTALLED = "STT_WHISPER_NOT_INSTALLED"
    STT_MODEL_NOT_INSTALLED = "STT_MODEL_NOT_INSTALLED"
    STT_MODEL_LOAD_FAILED = "STT_MODEL_LOAD_FAILED"
    STT_INVALID_AUDIO = "STT_INVALID_AUDIO"
    STT_UNSUPPORTED_AUDIO = "STT_UNSUPPORTED_AUDIO"
    STT_OUT_OF_MEMORY = "STT_OUT_OF_MEMORY"
    STT_UNKNOWN_MODEL = "STT_UNKNOWN_MODEL"
    STT_DOWNLOAD_FAILED = "STT_DOWNLOAD_FAILED"
    STT_WORKER_CRASH = "STT_WORKER_CRASH"

    # -------------------- Translation (Phase 4) --------------------
    TRANSLATE_LLAMA_NOT_INSTALLED = "TRANSLATE_LLAMA_NOT_INSTALLED"
    TRANSLATE_MODEL_NOT_INSTALLED = "TRANSLATE_MODEL_NOT_INSTALLED"
    TRANSLATE_MODEL_LOAD_FAILED = "TRANSLATE_MODEL_LOAD_FAILED"
    TRANSLATE_UNKNOWN_PROMPT = "TRANSLATE_UNKNOWN_PROMPT"
    TRANSLATE_INVALID_JSON = "TRANSLATE_INVALID_JSON"
    TRANSLATE_INCOMPLETE_RESPONSE = "TRANSLATE_INCOMPLETE_RESPONSE"
    TRANSLATE_OUT_OF_MEMORY = "TRANSLATE_OUT_OF_MEMORY"
    TRANSLATE_LLM_FAILURE = "TRANSLATE_LLM_FAILURE"
    TRANSLATE_WORKER_CRASH = "TRANSLATE_WORKER_CRASH"
    TRANSLATE_UNKNOWN_PRESET = "TRANSLATE_UNKNOWN_PRESET"
    TRANSLATE_DOWNLOAD_FAILED = "TRANSLATE_DOWNLOAD_FAILED"

    # -------------------- TTS (Phase 6) --------------------
    TTS_ENGINE_UNAVAILABLE = "TTS_ENGINE_UNAVAILABLE"
    TTS_VOICE_MISSING = "TTS_VOICE_MISSING"
    TTS_MODEL_INVALID = "TTS_MODEL_INVALID"
    TTS_INVALID_TEXT = "TTS_INVALID_TEXT"
    TTS_ENGINE_FAILURE = "TTS_ENGINE_FAILURE"
    TTS_OUT_OF_MEMORY = "TTS_OUT_OF_MEMORY"
    TTS_DISK_FULL = "TTS_DISK_FULL"
    TTS_WORKER_CRASH = "TTS_WORKER_CRASH"
    TTS_UNKNOWN_PRESET = "TTS_UNKNOWN_PRESET"
    TTS_DOWNLOAD_FAILED = "TTS_DOWNLOAD_FAILED"

    # -------------------- Sync (Phase 7) --------------------
    SYNC_FFMPEG_MISSING = "SYNC_FFMPEG_MISSING"
    SYNC_SOURCE_MISSING = "SYNC_SOURCE_MISSING"
    SYNC_SOURCE_INVALID = "SYNC_SOURCE_INVALID"
    SYNC_INVALID_TIMING = "SYNC_INVALID_TIMING"
    SYNC_ENGINE_FAILURE = "SYNC_ENGINE_FAILURE"
    SYNC_DISK_FULL = "SYNC_DISK_FULL"
    SYNC_WORKER_CRASH = "SYNC_WORKER_CRASH"


@dataclass(frozen=True)
class RpcError(Exception):
    code: str
    message: str
    data: Any = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out
