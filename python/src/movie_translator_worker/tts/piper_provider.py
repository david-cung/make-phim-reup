"""Piper-backed :class:`TTSProvider`.

Piper (https://github.com/rhasspy/piper) is a small, offline,
permissively-licensed neural TTS engine with very good Vietnamese
coverage. It fits our constraints:

* CPU-friendly (ONNX Runtime), no CUDA required.
* Apple Silicon compatible (falls back to CPU cleanly).
* Small memory footprint (~30–120 MB per voice).
* Model files are single ``.onnx`` + ``.onnx.json`` pairs users can
  drop in manually.
* MIT licensed.

Nothing in this module is imported at package import time — Piper
brings in ``onnxruntime`` which we don't want to pay for on cold RPC
calls that never touch TTS.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Optional

from .. import logging as log
from ..errors import RpcErrorCode
from .models import SynthesisResult, TTSSettings, VoiceInfo
from .provider import ProviderError, TTSProvider
from .registry import list_piper_voices, resolve_piper_voice
from .wav_io import apply_volume_pcm16, probe_wav, write_pcm16_mono


class PiperTTSProvider(TTSProvider):
    """Concrete provider that runs Piper locally.

    Uses the Python bindings when available (they let us stream sample
    chunks and keep the model in-process), and falls back to the CLI
    when only the ``piper`` binary is installed. Both paths produce
    the same WAV file on disk.
    """

    name = "piper"

    def __init__(self, models_root: Path) -> None:
        self._models_root = Path(models_root)
        self._voice: Any = None
        self._voice_id: Optional[str] = None
        self._voice_sample_rate: Optional[int] = None
        self._use_binary: Optional[bool] = None

    # ---------------------------------------------------------- interface

    def get_voices(self) -> list[VoiceInfo]:
        return list_piper_voices(self._models_root)

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        settings: TTSSettings,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise ProviderError(
                RpcErrorCode.TTS_INVALID_TEXT,
                "cannot synthesise empty text",
            )
        info = resolve_piper_voice(self._models_root, voice_id)
        if info is None:
            raise ProviderError(
                RpcErrorCode.TTS_VOICE_MISSING,
                f"piper voice {voice_id!r} is not installed; drop it into <models>/tts/piper/{voice_id}/",
                recoverable=True,
            )

        opts = settings.normalised()
        length_scale = 1.0 / max(0.1, opts.speed)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        use_binary = self._resolve_backend()
        if use_binary:
            self._synthesize_via_cli(
                info=info,
                text=text,
                length_scale=length_scale,
                dst=dst,
            )
        else:
            self._synthesize_via_bindings(
                info=info,
                text=text,
                length_scale=length_scale,
                dst=dst,
            )

        if not dst.is_file() or dst.stat().st_size == 0:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                "piper produced an empty audio file",
            )

        # Apply post-processing (volume) if the engine didn't handle it.
        if abs(opts.volume - 1.0) > 1e-4:
            self._apply_volume(dst, opts.volume)

        duration, sample_rate, channels = probe_wav(dst)
        return SynthesisResult(
            file_path=str(dst),
            duration_secs=duration,
            sample_rate=sample_rate,
            channels=channels,
            size_bytes=dst.stat().st_size,
        )

    def unload(self) -> None:
        self._voice = None
        self._voice_id = None
        self._voice_sample_rate = None

    # ---------------------------------------------------------- internals

    def _resolve_backend(self) -> bool:
        """Return True when we should call the ``piper`` CLI, False for
        in-process bindings. Cached across calls."""
        if self._use_binary is not None:
            return self._use_binary
        try:
            import importlib.util

            has_bindings = importlib.util.find_spec("piper") is not None
        except Exception:  # pragma: no cover - defensive
            has_bindings = False
        if has_bindings:
            self._use_binary = False
        elif shutil.which("piper") is not None:
            self._use_binary = True
        else:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "piper is not installed; add it to the worker environment (`pip install piper-tts`) or install the CLI",
                recoverable=True,
            )
        return self._use_binary

    def _synthesize_via_bindings(
        self,
        *,
        info: VoiceInfo,
        text: str,
        length_scale: float,
        dst: Path,
    ) -> None:
        try:
            from piper.voice import PiperVoice, SynthesisConfig  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - handled by _resolve_backend
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "piper bindings not importable",
                recoverable=True,
            ) from e

        if self._voice_id != info.id:
            self.unload()
            try:
                self._voice = PiperVoice.load(info.model_path, config_path=info.config_path)
            except MemoryError as e:  # pragma: no cover
                raise ProviderError(
                    RpcErrorCode.TTS_OUT_OF_MEMORY,
                    f"not enough memory to load voice {info.id}",
                    recoverable=True,
                ) from e
            except Exception as e:
                log.warn("piper voice load failed", voice=info.id, error=str(e))
                raise ProviderError(
                    RpcErrorCode.TTS_MODEL_INVALID,
                    f"failed to load piper voice {info.id}: {e}",
                ) from e
            self._voice_id = info.id
            self._voice_sample_rate = getattr(
                getattr(self._voice, "config", None), "sample_rate", info.sample_rate
            )

        cfg = SynthesisConfig(length_scale=float(length_scale))
        chunks: list[bytes] = []
        try:
            for audio in self._voice.synthesize(text, syn_config=cfg):
                data = getattr(audio, "audio_int16_bytes", None)
                if data:
                    chunks.append(data)
        except MemoryError as e:  # pragma: no cover
            raise ProviderError(
                RpcErrorCode.TTS_OUT_OF_MEMORY,
                "piper ran out of memory during synthesis",
                recoverable=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                f"piper synthesis failed: {e}",
            ) from e

        if not chunks:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                "piper produced no audio samples",
            )
        sample_rate = int(self._voice_sample_rate or info.sample_rate)
        write_pcm16_mono(dst, b"".join(chunks), sample_rate=sample_rate)

    def _synthesize_via_cli(
        self,
        *,
        info: VoiceInfo,
        text: str,
        length_scale: float,
        dst: Path,
    ) -> None:
        # ``piper --model X --output_file Y`` reads text from stdin.
        cmd = [
            "piper",
            "--model",
            str(info.model_path),
            "--length_scale",
            f"{length_scale:.4f}",
            "--output_file",
            str(dst),
        ]
        if info.config_path:
            cmd += ["--config", str(info.config_path)]
        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError as e:  # pragma: no cover - handled by _resolve_backend
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "piper CLI vanished during run",
                recoverable=True,
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                "piper CLI timed out after 120s",
            ) from e

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                "piper CLI failed: " + " | ".join(tail) if tail else "piper CLI failed",
            )

    @staticmethod
    def _apply_volume(dst: Path, volume: float) -> None:
        import wave

        with wave.open(str(dst), "rb") as r:
            channels = r.getnchannels()
            sampwidth = r.getsampwidth()
            rate = r.getframerate()
            frames = r.readframes(r.getnframes())
        if sampwidth != 2:
            # Piper always ships PCM16; if a future voice deviates we
            # skip volume rather than corrupt the buffer.
            return
        # apply_volume_pcm16 works on interleaved PCM16 regardless of
        # channel count.
        _ = struct  # keep the import used for clarity above
        scaled = apply_volume_pcm16(frames, float(volume))
        tmp = dst.with_suffix(".tmp.wav")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(sampwidth)
            w.setframerate(rate)
            w.writeframes(scaled)
        tmp.replace(dst)
