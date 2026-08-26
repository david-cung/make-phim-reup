"""Focused tests for TTS handler edge cases."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.tts.handlers import _synthesize_fitting  # noqa: E402
from movie_translator_worker.tts.models import SynthesisResult, TTSSettings  # noqa: E402


class _ExplodingProvider:
    name = "fake"

    def get_voices(self):
        return []

    def synthesize(self, text, voice_id, output_path, settings):
        raise AssertionError("punctuation-only text should not call the provider")

    def unload(self):
        pass


def test_punctuation_only_tts_writes_silence(tmp_path: Path) -> None:
    out = tmp_path / "ellipsis.wav"
    result = _synthesize_fitting(
        _ExplodingProvider(),
        text="…",
        voice_id="vi_VN-vais1000-medium",
        output_path=str(out),
        settings=TTSSettings(),
        target_duration=1.25,
    )

    assert isinstance(result, SynthesisResult)
    assert out.is_file()
    assert result.duration_secs > 0
    assert abs(result.duration_secs - 1.25) < 0.001
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getnframes() == int(22050 * 1.25)
