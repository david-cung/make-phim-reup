"""Transport-shaped dataclasses for the local-LLM translation subsystem.

Deliberately free of any ``llama-cpp-python`` types so this module can
be imported without the heavy runtime dependency and reused verbatim
by tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

TRANSLATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TranslatedSegment:
    """A single subtitle segment paired with its translation.

    Timing (``start``/``end``) is duplicated from the transcript so this
    file is self-describing on disk — but the transcript remains the
    single source of truth. If the two disagree, the transcript wins.
    ``edited`` records whether the user hand-corrected the translation
    in the editor after the LLM produced it.
    """

    id: int
    source_text: str
    translation: str
    start: float
    end: float
    edited: bool = False


@dataclass
class TranslationChunk:
    """A window of segments handed to the LLM as one request.

    ``chunk_index`` starts at 0 and increments per request. ``segment_ids``
    identifies which transcript segments must be translated in this
    chunk. ``context_before_ids``/``context_after_ids`` provide the
    dialogue context the LLM sees but must not re-translate.
    """

    chunk_index: int
    segment_ids: list[int]
    context_before_ids: list[int] = field(default_factory=list)
    context_after_ids: list[int] = field(default_factory=list)


@dataclass
class TranslateOptions:
    """Every parameter that materially affects translation output.

    Anything not in here is assumed not to invalidate the cache.
    """

    model: str
    source_language: str = "en"
    target_language: str = "vi"
    prompt_version: str = "translation_prompt_v1"
    chunk_size: int = 30
    context_before: int = 4
    context_after: int = 2
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048

    def normalised(self) -> "TranslateOptions":
        return TranslateOptions(
            model=self.model,
            source_language=(self.source_language or "").lower() or "en",
            target_language=(self.target_language or "").lower() or "vi",
            prompt_version=self.prompt_version or "translation_prompt_v1",
            chunk_size=max(1, int(self.chunk_size)),
            context_before=max(0, int(self.context_before)),
            context_after=max(0, int(self.context_after)),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            max_tokens=max(64, int(self.max_tokens)),
        )


@dataclass
class TranslationDoc:
    """The persisted deliverable at
    ``<project>/translation/translation.json``.
    """

    version: int
    source_language: str
    target_language: str
    segments: list[TranslatedSegment]
    model: str
    prompt_version: str
    cache_key: str
    transcript_cache_key: str
    audio_hash: str
    created_at: str
    updated_at: str
    provider: str = "llama.cpp"
    options: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- codec


def doc_to_dict(doc: TranslationDoc) -> dict[str, Any]:
    """Serialise :class:`TranslationDoc` to the on-disk camelCase schema.

    Timing is *always* preserved so the file can be reopened even if the
    canonical transcript is missing.
    """
    return {
        "version": doc.version,
        "sourceLanguage": doc.source_language,
        "targetLanguage": doc.target_language,
        "segments": [_segment_to_dict(s) for s in doc.segments],
        "model": doc.model,
        "promptVersion": doc.prompt_version,
        "cacheKey": doc.cache_key,
        "transcriptCacheKey": doc.transcript_cache_key,
        "audioHash": doc.audio_hash,
        "createdAt": doc.created_at,
        "updatedAt": doc.updated_at,
        "provider": doc.provider,
        "options": doc.options,
    }


def doc_from_dict(payload: dict[str, Any]) -> TranslationDoc:
    if not isinstance(payload, dict):
        raise ValueError("translation must be a JSON object")
    version = payload.get("version")
    if version != TRANSLATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported translation schema version: {version}")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")
    segments = [_segment_from_dict(s) for s in raw_segments]
    validate_translated_segments(segments)
    return TranslationDoc(
        version=version,
        source_language=str(payload.get("sourceLanguage") or ""),
        target_language=str(payload.get("targetLanguage") or ""),
        segments=segments,
        model=str(payload.get("model") or ""),
        prompt_version=str(payload.get("promptVersion") or "translation_prompt_v1"),
        cache_key=str(payload.get("cacheKey") or ""),
        transcript_cache_key=str(payload.get("transcriptCacheKey") or ""),
        audio_hash=str(payload.get("audioHash") or ""),
        created_at=str(payload.get("createdAt") or ""),
        updated_at=str(payload.get("updatedAt") or ""),
        provider=str(payload.get("provider") or "llama.cpp"),
        options=dict(payload.get("options") or {}),
    )


def validate_translated_segments(segments: Iterable[TranslatedSegment]) -> None:
    """Raise ``ValueError`` if any segment is malformed or if ordering is
    not monotonically non-decreasing."""

    last_start = -1.0
    seen_ids: set[int] = set()
    for seg in segments:
        if not isinstance(seg, TranslatedSegment):
            raise ValueError(
                f"expected TranslatedSegment, got {type(seg).__name__}"
            )
        if seg.id < 0:
            raise ValueError(f"segment id must be non-negative: {seg.id}")
        if seg.id in seen_ids:
            raise ValueError(f"duplicate segment id: {seg.id}")
        seen_ids.add(seg.id)
        if seg.start < 0 or seg.end < 0:
            raise ValueError(f"segment {seg.id} has negative timestamps")
        if seg.end < seg.start:
            raise ValueError(
                f"segment {seg.id} has end < start ({seg.end} < {seg.start})"
            )
        if seg.start < last_start - 1e-3:
            raise ValueError(
                f"segment {seg.id} starts before previous ({seg.start} < {last_start})"
            )
        last_start = seg.start


def build_cache_key(
    *,
    transcript_cache_key: str,
    audio_hash: str,
    options: TranslateOptions,
) -> str:
    """Deterministic hash over every input that materially affects
    translation output.
    """
    opts = options.normalised()
    parts = [
        "translation_v1",
        transcript_cache_key or "",
        audio_hash or "",
        opts.source_language,
        opts.target_language,
        opts.model,
        opts.prompt_version,
        f"chunk={opts.chunk_size}",
        f"before={opts.context_before}",
        f"after={opts.context_after}",
        f"temp={opts.temperature:.4f}",
        f"top_p={opts.top_p:.4f}",
        f"max_tokens={opts.max_tokens}",
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --------------------------------------------------------------- chunking


def chunk_segments(
    segment_ids: list[int],
    *,
    chunk_size: int,
    context_before: int,
    context_after: int,
) -> list[TranslationChunk]:
    """Split ``segment_ids`` into overlapping chunks.

    The chunk itself is the window that must be translated. The
    ``context_before``/``context_after`` windows are extra ids the
    prompt should show (but the model must not re-translate).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not segment_ids:
        return []
    chunks: list[TranslationChunk] = []
    total = len(segment_ids)
    idx = 0
    for start in range(0, total, chunk_size):
        window = segment_ids[start : start + chunk_size]
        before = segment_ids[max(0, start - context_before) : start]
        after = segment_ids[
            start + len(window) : start + len(window) + context_after
        ]
        chunks.append(
            TranslationChunk(
                chunk_index=idx,
                segment_ids=list(window),
                context_before_ids=list(before),
                context_after_ids=list(after),
            )
        )
        idx += 1
    return chunks


def merge_translations(
    base: list[TranslatedSegment],
    updates: dict[int, str],
    *,
    mark_edited: bool = False,
) -> list[TranslatedSegment]:
    """Return a copy of ``base`` with the given ``updates`` applied.

    Segments not in ``updates`` are left untouched (their translation
    field, if any, is preserved). Used both by the provider (fresh LLM
    output) and by the manual editor (user overrides).
    """
    by_id = {s.id: s for s in base}
    out: list[TranslatedSegment] = []
    for seg in base:
        if seg.id in updates:
            out.append(
                TranslatedSegment(
                    id=seg.id,
                    source_text=seg.source_text,
                    translation=updates[seg.id],
                    start=seg.start,
                    end=seg.end,
                    edited=True if mark_edited else seg.edited,
                )
            )
        else:
            out.append(seg)
    # If updates contain ids that weren't in base, ignore them silently:
    # timestamps must come from the transcript.
    _ = by_id
    return out


# ------------------------------------------------------------------ helpers


def _segment_to_dict(s: TranslatedSegment) -> dict[str, Any]:
    return {
        "id": s.id,
        "sourceText": s.source_text,
        "translation": s.translation,
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "edited": bool(s.edited),
    }


def _segment_from_dict(payload: Any) -> TranslatedSegment:
    if not isinstance(payload, dict):
        raise ValueError("translated segment must be an object")
    try:
        return TranslatedSegment(
            id=int(payload["id"]),
            source_text=str(payload.get("sourceText", "")),
            translation=str(payload.get("translation", "")),
            start=float(payload.get("start", 0.0)),
            end=float(payload.get("end", 0.0)),
            edited=bool(payload.get("edited", False)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid translated segment: {e}") from e


# ------------------------------------------------------------ query helpers


def missing_ids(doc_segments: list[TranslatedSegment]) -> list[int]:
    """Return ids of segments that still need translation (empty
    translation string). Used by the service to resume partial work.
    """
    return [s.id for s in doc_segments if not s.translation.strip() and not s.edited]


def unique_language(lang: Optional[str], fallback: str) -> str:
    return (lang or fallback).lower()
