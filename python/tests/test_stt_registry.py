"""Model catalogue + install detection tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.stt import registry  # noqa: E402


def _fake_snapshot(root: Path, name: str) -> None:
    d = registry.model_dir(root, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(b"\x00" * 128)
    (d / "config.json").write_text("{}")


def test_known_models_contain_required_set() -> None:
    names = registry.known_model_names()
    for wanted in ("small", "medium", "large", "large-v3", "turbo"):
        assert wanted in names


def test_list_models_marks_installed(tmp_path: Path) -> None:
    _fake_snapshot(tmp_path, "small")
    models = {m.name: m for m in registry.list_models(tmp_path)}
    assert models["small"].installed is True
    assert models["small"].size_bytes is not None and models["small"].size_bytes > 0
    assert models["small"].path is not None
    assert models["medium"].installed is False
    assert models["medium"].size_bytes is None


def test_ensure_downloaded_short_circuits_when_installed(tmp_path: Path) -> None:
    _fake_snapshot(tmp_path, "small")

    def _stub(*_a, **_kw) -> None:
        raise AssertionError("downloader must not be called when model is present")

    path = registry.ensure_downloaded(tmp_path, "small", hf_downloader=_stub)
    assert path == registry.model_dir(tmp_path, "small")


def test_ensure_downloaded_invokes_downloader(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def _downloader(repo_id: str, local_dir: Path, _progress) -> None:
        calls.append((repo_id, local_dir))
        (Path(local_dir) / "model.bin").write_bytes(b"\x00")
        (Path(local_dir) / "config.json").write_text("{}")

    registry.ensure_downloaded(tmp_path, "medium", hf_downloader=_downloader)
    assert len(calls) == 1
    assert calls[0][0] == registry.repo_for("medium")
    assert registry.is_installed(tmp_path, "medium")


def test_ensure_downloaded_unknown_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        registry.ensure_downloaded(tmp_path, "not-a-model", hf_downloader=lambda *a, **k: None)


def test_ensure_downloaded_missing_files_raises(tmp_path: Path) -> None:
    def _downloader(*_a, **_kw) -> None:
        return

    with pytest.raises(RuntimeError):
        registry.ensure_downloaded(tmp_path, "small", hf_downloader=_downloader)


def test_remove_model_is_idempotent(tmp_path: Path) -> None:
    _fake_snapshot(tmp_path, "small")
    assert registry.is_installed(tmp_path, "small")
    registry.remove_model(tmp_path, "small")
    assert not registry.is_installed(tmp_path, "small")
    # A second call must not raise.
    registry.remove_model(tmp_path, "small")
