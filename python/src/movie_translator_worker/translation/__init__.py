"""Local LLM translation (Phase 4).

Public surface kept narrow so nothing else depends on ``llama.cpp``
directly. Consumers must go through :class:`TranslationProvider` and the
transport-shaped dataclasses in :mod:`.models`.
"""

from __future__ import annotations

from . import prompts, registry
from .models import (
    TRANSLATION_SCHEMA_VERSION,
    TranslateOptions,
    TranslatedSegment,
    TranslationChunk,
    TranslationDoc,
    build_cache_key,
    chunk_segments,
    doc_from_dict,
    doc_to_dict,
    merge_translations,
    validate_translated_segments,
)
from .provider import (
    ProgressCallback,
    ProviderCancelled,
    ProviderError,
    TranslateContext,
    TranslationProvider,
)

__all__ = [
    "ProgressCallback",
    "ProviderCancelled",
    "ProviderError",
    "TRANSLATION_SCHEMA_VERSION",
    "TranslateContext",
    "TranslateOptions",
    "TranslatedSegment",
    "TranslationChunk",
    "TranslationDoc",
    "TranslationProvider",
    "build_cache_key",
    "chunk_segments",
    "doc_from_dict",
    "doc_to_dict",
    "merge_translations",
    "prompts",
    "registry",
    "validate_translated_segments",
]
