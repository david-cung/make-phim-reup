"""Phase 12 — recommended presets + camelCase serialisation contract.

These tests protect the Python↔Rust wire shape that
``TranslationService::list_recommended`` relies on. When any of the
field names below drift, the Rust side deserialises into an empty
list and the "Download & translate" auto-flow silently falls back
to the "no models installed" dead end.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_translator_worker.translation import registry  # noqa: E402


def test_recommended_presets_shape_matches_rust() -> None:
    presets = registry.recommended_presets()
    assert presets, "at least one preset must be shipped"
    for p in presets:
        # Fields Rust `RecommendedPreset` expects (camelCase per
        # `#[serde(rename_all = "camelCase")]`). Fail loudly if any
        # future edit renames them back to snake_case.
        assert set(p.keys()) == {
            "preset",
            "repo",
            "filename",
            "approxSizeBytes",
            "label",
            "isDefault",
        }
        assert isinstance(p["approxSizeBytes"], int)
        assert isinstance(p["isDefault"], bool)
        assert p["filename"].endswith(".gguf")
        assert "/" in p["repo"], "repo must be a HuggingFace-style owner/name id"


def test_exactly_one_default_preset() -> None:
    presets = registry.recommended_presets()
    defaults = [p for p in presets if p["isDefault"]]
    assert len(defaults) == 1, (
        "Frontend picks the first isDefault preset — having zero or "
        "more than one makes the auto-download flow ambiguous."
    )
    assert defaults[0]["preset"] == registry.DEFAULT_RECOMMENDED_PRESET


def test_recommended_meta_lookup() -> None:
    assert registry.recommended_meta(registry.DEFAULT_RECOMMENDED_PRESET) is not None
    assert registry.recommended_meta("does-not-exist") is None


def test_translation_model_info_serialises_to_camel_case(tmp_path: Path) -> None:
    d = registry.ensure_models_dir(tmp_path)
    (d / "example.gguf").write_bytes(b"\x00" * 32)
    models = registry.list_models(tmp_path)
    assert len(models) == 1
    payload = models[0].to_dict()
    assert set(payload.keys()) == {"name", "path", "sizeBytes", "isDefault"}
    assert payload["isDefault"] is True
