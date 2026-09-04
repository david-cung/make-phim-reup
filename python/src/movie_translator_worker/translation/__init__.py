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
    TranslationMetadata,
    TranslationResult,
    build_cache_key,
    chunk_segments,
    chunk_segments_with_context,
    doc_from_dict,
    doc_to_dict,
    ensure_translation_result,
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
from .identity import (
    ActiveSpeakerCandidate,
    ActiveSpeakerEvidence,
    ActiveSpeakerResolver,
    BasicFaceIdentityMatcher,
    BasicVoiceIdentityMatcher,
    BoundingBox,
    CharacterIdentity,
    CharacterIdentityResolver,
    FaceObservation,
    FaceTrack,
    IdentityEvidence,
    MultimodalIdentityGraph,
    SegmentIdentityResolution,
    SpeakerIdentity,
    integrate_identity_graph_with_context_store,
    speaker_identities_from_segments,
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
    "TranslationMetadata",
    "TranslationResult",
    "TranslationProvider",
    "ActiveSpeakerCandidate",
    "ActiveSpeakerEvidence",
    "ActiveSpeakerResolver",
    "BasicFaceIdentityMatcher",
    "BasicVoiceIdentityMatcher",
    "BoundingBox",
    "CharacterIdentity",
    "CharacterIdentityResolver",
    "FaceObservation",
    "FaceTrack",
    "IdentityEvidence",
    "MultimodalIdentityGraph",
    "SegmentIdentityResolution",
    "SpeakerIdentity",
    "build_cache_key",
    "chunk_segments",
    "chunk_segments_with_context",
    "doc_from_dict",
    "doc_to_dict",
    "ensure_translation_result",
    "merge_translations",
    "prompts",
    "registry",
    "integrate_identity_graph_with_context_store",
    "speaker_identities_from_segments",
    "validate_translated_segments",
]
