"""High-quality local Vietnamese speech using F5-TTS ViVoice."""

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import sys
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .. import logging as log
from ..errors import RpcErrorCode
from . import registry
from .models import SynthesisResult, TTSSettings, VoiceInfo
from .provider import ProviderCancelled, ProviderError, TTSProvider
from .wav_io import inspect_pcm16_wav, postprocess_pcm16_wav, probe_wav


class F5VietnameseTTSProvider(TTSProvider):
    """Load the ViVoice checkpoint/vocoder once and reuse them for a job."""

    name = registry.F5_ENGINE

    def __init__(self, models_root: Path) -> None:
        self._models_root = Path(models_root)
        self._runtime: Any = None
        self._runtime_device: Optional[str] = None
        self._runtime_identity: Optional[str] = None
        self._reference_cache: dict[str, tuple[str, str]] = {}
        self._cancelled: Callable[[], bool] = lambda: False

    def set_cancel_checker(self, checker: Callable[[], bool]) -> None:
        self._cancelled = checker

    def get_voices(self) -> list[VoiceInfo]:
        return registry.list_f5_voices(self._models_root)

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        settings: TTSSettings,
    ) -> SynthesisResult:
        if not text.strip():
            raise ProviderError(RpcErrorCode.TTS_INVALID_TEXT, "cannot synthesise empty text")
        voice = registry.resolve_f5_voice(self._models_root, voice_id)
        if voice is None:
            raise ProviderError(
                RpcErrorCode.TTS_VOICE_MISSING,
                f"F5 voice profile {voice_id!r} is missing or has no valid reference audio/text",
                recoverable=True,
            )
        if not voice.reference_audio_path or not voice.reference_text:
            raise ProviderError(
                RpcErrorCode.TTS_MODEL_INVALID,
                "F5 voice profiles require both reference audio and its exact transcript",
                recoverable=True,
            )

        opts = settings.normalised()
        if self._cancelled():
            raise ProviderCancelled()
        device = self._select_device(opts.device)
        runtime = self._ensure_loaded(device, voice.cache_identity or "")
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".f5.tmp.wav")
        tmp.unlink(missing_ok=True)
        try:
            with _offline_inference_environment():
                with redirect_stdout(sys.stderr):
                    from f5_tts.infer.utils_infer import (  # type: ignore[import-not-found]
                        infer_process,
                    )
                    from f5_tts.model.utils import seed_everything  # type: ignore[import-not-found]
                    import soundfile as sf  # type: ignore[import-not-found]

                    ref_audio, ref_text = self._prepare_reference(voice)
                    seed_material = (
                        f"{voice.cache_identity}:{text}:{opts.speed:.4f}:{device}"
                    ).encode("utf-8")
                    seed_everything(
                        int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
                    )
                    wav, sample_rate, _spec = infer_process(
                        ref_audio,
                        ref_text,
                        text,
                        runtime.ema_model,
                        runtime.vocoder,
                        runtime.mel_spec_type,
                        show_info=lambda *_args, **_kwargs: None,
                        progress=None,
                        speed=max(0.90, min(1.12, opts.speed)),
                        nfe_step=32,
                        cfg_strength=2.0,
                        cross_fade_duration=0.12,
                        device=runtime.device,
                    )
                    if self._cancelled():
                        raise ProviderCancelled()
                    sf.write(str(tmp), wav, sample_rate, subtype="PCM_16")
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise ProviderError(
                    RpcErrorCode.TTS_ENGINE_FAILURE,
                    "F5-TTS produced no audio",
                )
            tmp.replace(dst)
            postprocess_pcm16_wav(dst, volume=opts.volume)
        except (ProviderError, ProviderCancelled):
            tmp.unlink(missing_ok=True)
            raise
        except MemoryError as exc:
            tmp.unlink(missing_ok=True)
            raise ProviderError(
                RpcErrorCode.TTS_OUT_OF_MEMORY,
                "F5-TTS ran out of memory; use a GPU with more VRAM, CPU mode, or Piper FAST",
                recoverable=True,
            ) from exc
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if "out of memory" in str(exc).lower():
                raise ProviderError(
                    RpcErrorCode.TTS_OUT_OF_MEMORY,
                    f"F5-TTS ran out of memory on {device}: {exc}",
                    recoverable=True,
                ) from exc
            log.warn("f5 synthesis failed", voice=voice_id, device=device, error=str(exc))
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                f"F5-TTS synthesis failed: {exc}",
            ) from exc

        duration, sample_rate, channels = probe_wav(dst)
        metrics = inspect_pcm16_wav(dst)
        if duration <= 0.02 or float(metrics["rms"]) < 2.0:
            dst.unlink(missing_ok=True)
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_FAILURE,
                "F5-TTS produced silent or invalid audio",
            )
        return SynthesisResult(
            file_path=str(dst),
            duration_secs=duration,
            sample_rate=sample_rate,
            channels=channels,
            size_bytes=dst.stat().st_size,
        )

    def unload(self) -> None:
        for prepared, _text in self._reference_cache.values():
            try:
                prepared_path = Path(prepared).resolve()
                if Path(tempfile.gettempdir()).resolve() in prepared_path.parents:
                    prepared_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._reference_cache.clear()
        self._runtime = None
        self._runtime_device = None
        self._runtime_identity = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self, device: str, voice_identity: str) -> Any:
        paths = registry.f5_model_paths(self._models_root)
        status = registry.f5_model_status(self._models_root)
        if not status["installed"]:
            raise ProviderError(
                RpcErrorCode.TTS_VOICE_MISSING,
                "F5-TTS Vietnamese model is not installed. Install the QUALITY model in Voice Over settings first.",
                recoverable=True,
            )
        model_identity = ":".join(
            (
                registry.F5_MODEL_REVISION,
                _path_identity(paths["checkpoint"]),
                _path_identity(paths["vocab"]),
                _path_identity(paths["vocoder_config"]),
                _path_identity(paths["vocoder_model"]),
            )
        )
        if (
            self._runtime is not None
            and self._runtime_device == device
            and self._runtime_identity == model_identity
        ):
            return self._runtime
        self.unload()
        try:
            with _offline_inference_environment(), redirect_stdout(sys.stderr):
                from f5_tts.api import F5TTS  # type: ignore[import-not-found]

                runtime = F5TTS(
                    model="F5TTS_Base",
                    ckpt_file=str(paths["checkpoint"]),
                    vocab_file=str(paths["vocab"]),
                    vocoder_local_path=str(paths["vocoder_config"].parent),
                    device=device,
                )
        except ImportError as exc:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "F5-TTS runtime is not installed; install the worker [f5] extra",
                recoverable=True,
            ) from exc
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise ProviderError(
                    RpcErrorCode.TTS_OUT_OF_MEMORY,
                    f"not enough memory to load F5-TTS on {device}: {exc}",
                    recoverable=True,
                ) from exc
            raise ProviderError(
                RpcErrorCode.TTS_MODEL_INVALID,
                f"failed to load local F5-TTS model: {exc}",
            ) from exc
        self._runtime = runtime
        self._runtime_device = device
        self._runtime_identity = model_identity
        log.info("f5 model loaded", device=device, voice_identity=voice_identity)
        return runtime

    def _prepare_reference(self, voice: VoiceInfo) -> tuple[str, str]:
        identity = voice.cache_identity or voice.id
        cached = self._reference_cache.get(identity)
        if cached and Path(cached[0]).is_file():
            return cached
        from f5_tts.infer.utils_infer import (  # type: ignore[import-not-found]
            preprocess_ref_audio_text,
        )

        with redirect_stdout(sys.stderr):
            prepared_audio, prepared_text = preprocess_ref_audio_text(
                voice.reference_audio_path,
                voice.reference_text,
                show_info=lambda *_args, **_kwargs: None,
                device=self._runtime_device or "cpu",
            )
        cached = (str(prepared_audio), str(prepared_text))
        self._reference_cache[identity] = cached
        return cached

    @staticmethod
    def _select_device(requested: str) -> str:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "PyTorch is required for F5-TTS",
                recoverable=True,
            ) from exc
        request = (requested or "auto").lower()
        cuda = bool(torch.cuda.is_available())
        mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if request == "cuda" and not cuda:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "CUDA was selected for F5-TTS but no compatible CUDA device is available",
                recoverable=True,
            )
        if request == "mps" and not mps:
            raise ProviderError(
                RpcErrorCode.TTS_ENGINE_UNAVAILABLE,
                "MPS was selected for F5-TTS but is unavailable",
                recoverable=True,
            )
        if request in {"cpu", "cuda", "mps"}:
            return request
        return "cuda" if cuda else "mps" if mps else "cpu"


@contextmanager
def _offline_inference_environment() -> Iterator[None]:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _path_identity(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
