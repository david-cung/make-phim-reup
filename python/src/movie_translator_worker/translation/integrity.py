"""Phase 8.0.1 subtitle translation integrity checks.

These validators keep context grouping separate from subtitle output:
nearby rows may inform a translation, but every accepted result must map
back to exactly one requested source segment.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from .models import TranslatedSegment, TranslationResult, ensure_translation_result
from .target_language import validate_target_language
from .units import (
    ALIGNMENT_MISMATCH,
    CONTENT_LEAKAGE,
    CONVERSATION_BOUNDARY_VIOLATION,
    DUPLICATE_RESULT,
    MISSING_RESULT,
    SEMANTIC_OMISSION,
    UNKNOWN_RESULT_ID,
    TranslationContractIssue,
    validate_translation_contract,
)


UNKNOWN_SPEAKER = "UNKNOWN"

UNTRANSLATED_SOURCE_FRAGMENT = "UNTRANSLATED_SOURCE_FRAGMENT"
FOREIGN_LANGUAGE_CONTAMINATION = "FOREIGN_LANGUAGE_CONTAMINATION"
CANDIDATE_LEAKAGE = "CANDIDATE_LEAKAGE"
TRANSLATOR_COMMENTARY_LEAK = "TRANSLATOR_COMMENTARY_LEAK"
EMPTY_TRANSLATION = "EMPTY_TRANSLATION"
SEGMENT_MERGE_ERROR = "SEGMENT_MERGE_ERROR"
SOURCE_ALIGNMENT_DRIFT = "SOURCE_ALIGNMENT_DRIFT"
SUSPICIOUS_TRANSLATION_DUPLICATION = "SUSPICIOUS_TRANSLATION_DUPLICATION"
BATCH_CARDINALITY_ERROR = "BATCH_CARDINALITY_ERROR"
SPEAKER_BOUNDARY_ERROR = "SPEAKER_BOUNDARY_ERROR"

INTEGRITY_ERROR_CODES = {
    UNTRANSLATED_SOURCE_FRAGMENT,
    FOREIGN_LANGUAGE_CONTAMINATION,
    CANDIDATE_LEAKAGE,
    TRANSLATOR_COMMENTARY_LEAK,
    EMPTY_TRANSLATION,
    SEGMENT_MERGE_ERROR,
    SOURCE_ALIGNMENT_DRIFT,
    SUSPICIOUS_TRANSLATION_DUPLICATION,
    BATCH_CARDINALITY_ERROR,
    SPEAKER_BOUNDARY_ERROR,
    MISSING_RESULT,
    DUPLICATE_RESULT,
    UNKNOWN_RESULT_ID,
    SEMANTIC_OMISSION,
    CONTENT_LEAKAGE,
    ALIGNMENT_MISMATCH,
    CONVERSATION_BOUNDARY_VIOLATION,
}

_VI_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ]+", re.UNICODE)

_SOURCE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("结婚", ("ket hon", "cuoi", "hon nhan")),
    ("結婚", ("ket hon", "cuoi", "hon nhan")),
    ("回国", ("ve nuoc",)),
    ("回國", ("ve nuoc",)),
    ("老宅", ("nha cu", "nha to", "nha chinh")),
    ("沈总", ("tong giam doc", "sep", "tham")),
    ("總", ("tong giam doc", "sep", "giam doc")),
    ("总", ("tong giam doc", "sep", "giam doc")),
    ("谢谢", ("cam on",)),
    ("謝謝", ("cam on",)),
    ("送我回家", ("dua toi ve nha", "cho toi ve nha")),
)


@dataclass(frozen=True)
class OutputValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "errors": list(self.errors)}


@dataclass(frozen=True)
class BatchIntegrityReport:
    source_segment_count: int
    translated_segment_count: int
    missing: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unknown: list[int] = field(default_factory=list)
    language_errors: dict[int, list[str]] = field(default_factory=dict)
    alignment_warnings: dict[int, list[str]] = field(default_factory=dict)
    speaker_boundary_errors: list[int] = field(default_factory=list)
    contract_issues: list[dict[str, object]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not (
            self.missing
            or self.duplicates
            or self.unknown
            or self.language_errors
            or self.alignment_warnings
            or self.speaker_boundary_errors
            or self.contract_issues
        )

    def metrics(self) -> dict[str, int]:
        language_flags = [flag for flags in self.language_errors.values() for flag in flags]
        alignment_flags = [flag for flags in self.alignment_warnings.values() for flag in flags]
        return {
            "source_segment_count": self.source_segment_count,
            "translated_segment_count": self.translated_segment_count,
            "missing_segment_count": len(self.missing),
            "duplicate_segment_count": len(self.duplicates),
            "foreign_residue_count": language_flags.count(UNTRANSLATED_SOURCE_FRAGMENT)
            + language_flags.count(FOREIGN_LANGUAGE_CONTAMINATION),
            "candidate_leak_count": language_flags.count(CANDIDATE_LEAKAGE),
            "alignment_error_count": alignment_flags.count(SOURCE_ALIGNMENT_DRIFT),
            "speaker_merge_error_count": alignment_flags.count(SEGMENT_MERGE_ERROR)
            + len(self.speaker_boundary_errors),
            "contract_issue_count": len(self.contract_issues),
        }

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "source_segments": self.source_segment_count,
            "translation_results": self.translated_segment_count,
            "validated_results": self.translated_segment_count
            - len(self.missing)
            - len(self.duplicates)
            - len(self.unknown),
            "missing": list(self.missing),
            "duplicates": list(self.duplicates),
            "unknown": list(self.unknown),
            "alignment_warnings": dict(self.alignment_warnings),
            "language_errors": dict(self.language_errors),
            "speaker_boundary_errors": list(self.speaker_boundary_errors),
            "contract_issues": list(self.contract_issues),
            "metrics": self.metrics(),
        }


def stable_segment_id(seg: TranslatedSegment) -> str:
    return seg.source_segment_id or f"seg_{seg.id:05d}"


def normalized_speaker_id(speaker_id: str | None) -> str:
    speaker = (speaker_id or "").strip()
    return speaker if speaker else UNKNOWN_SPEAKER


def speaker_merge_allowed(left: TranslatedSegment, right: TranslatedSegment) -> bool:
    left_speaker = normalized_speaker_id(left.speaker_id)
    right_speaker = normalized_speaker_id(right.speaker_id)
    if left_speaker == UNKNOWN_SPEAKER or right_speaker == UNKNOWN_SPEAKER:
        return False
    return left_speaker == right_speaker


def validate_vietnamese_output(text: str, *, source: str = "") -> OutputValidation:
    validation = validate_target_language("vi", text, source=source)
    return OutputValidation(valid=validation.valid, errors=list(validation.errors))


def validate_batch_integrity(
    *,
    expected_ids: list[int],
    translations: dict[int, TranslationResult],
    segments_by_id: dict[int, TranslatedSegment],
    target_language: str,
) -> BatchIntegrityReport:
    expected = set(expected_ids)
    actual_ids = list(translations.keys())
    actual = set(actual_ids)
    duplicates = sorted({sid for sid in actual_ids if actual_ids.count(sid) > 1})
    language_errors: dict[int, list[str]] = {}
    alignment_warnings: dict[int, list[str]] = {}
    contract_report = validate_translation_contract(
        expected_unit_ids=expected_ids,
        results=translations,
    )
    contract_issues = [issue.to_dict() for issue in contract_report.issues]
    for issue in contract_report.issues:
        _apply_contract_issue_to_legacy_fields(
            issue,
            language_errors=language_errors,
            alignment_warnings=alignment_warnings,
        )
    for sid in expected.intersection(actual):
        seg = segments_by_id.get(sid)
        result = translations.get(sid)
        if seg is None or result is None:
            continue
        validation = validate_target_language(
            target_language,
            result.translation,
            source=seg.source_text,
            metadata={
                "sourceUnitId": stable_segment_id(seg),
                "speakerRef": normalized_speaker_id(seg.speaker_id),
            },
        )
        if validation.errors:
            language_errors[sid] = list(
                dict.fromkeys([*(language_errors.get(sid) or []), *validation.errors])
            )
    ordered = _ordered_ids(segments_by_id)
    for sid in expected.intersection(actual):
        seg = segments_by_id.get(sid)
        result = translations.get(sid)
        if seg is None or result is None:
            continue
        warnings = alignment_issues(
            segment_id=sid,
            translation=result.translation,
            segments_by_id=segments_by_id,
            ordered_ids=ordered,
        )
        if warnings:
            alignment_warnings[sid] = warnings
    return BatchIntegrityReport(
        source_segment_count=len(expected_ids),
        translated_segment_count=len(translations),
        missing=[sid for sid in expected_ids if sid not in actual],
        duplicates=duplicates,
        unknown=sorted(actual - expected),
        language_errors=language_errors,
        alignment_warnings=alignment_warnings,
        contract_issues=contract_issues,
    )


def invalid_translation_ids(report: BatchIntegrityReport) -> list[int]:
    ids: list[int] = []
    ids.extend(report.missing)
    ids.extend(report.duplicates)
    ids.extend(report.unknown)
    ids.extend(report.language_errors)
    ids.extend(report.alignment_warnings)
    ids.extend(report.speaker_boundary_errors)
    return list(dict.fromkeys(ids))


def _apply_contract_issue_to_legacy_fields(
    issue: TranslationContractIssue,
    *,
    language_errors: dict[int, list[str]],
    alignment_warnings: dict[int, list[str]],
) -> None:
    try:
        sid = int(str(issue.unit_id)) if issue.unit_id is not None else None
    except ValueError:
        sid = None
    if sid is None:
        return
    if issue.code in {MISSING_RESULT, DUPLICATE_RESULT, UNKNOWN_RESULT_ID}:
        return
    if issue.code == SEMANTIC_OMISSION:
        language_errors.setdefault(sid, []).append(EMPTY_TRANSLATION)
        return
    if issue.code in {
        CONTENT_LEAKAGE,
        ALIGNMENT_MISMATCH,
        CONVERSATION_BOUNDARY_VIOLATION,
    }:
        alignment_warnings.setdefault(sid, []).append(issue.code)


def duplicate_translation_issues(
    translations: dict[int, TranslationResult],
    segments_by_id: dict[int, TranslatedSegment],
) -> dict[int, list[str]]:
    by_text: dict[str, list[int]] = {}
    for sid, result in translations.items():
        key = _normalise_vi(result.translation)
        if len(key) < 24:
            continue
        by_text.setdefault(key, []).append(sid)
    issues: dict[int, list[str]] = {}
    for ids in by_text.values():
        if len(ids) < 2:
            continue
        sources = {_normalise_source(segments_by_id.get(sid).source_text if sid in segments_by_id else "") for sid in ids}
        if len(sources) <= 1:
            continue
        for sid in ids:
            issues.setdefault(sid, []).append(SUSPICIOUS_TRANSLATION_DUPLICATION)
    return issues


def alignment_issues(
    *,
    segment_id: int,
    translation: str,
    segments_by_id: dict[int, TranslatedSegment],
    ordered_ids: list[int] | None = None,
) -> list[str]:
    seg = segments_by_id.get(segment_id)
    if seg is None:
        return []
    ordered = ordered_ids or _ordered_ids(segments_by_id)
    try:
        pos = ordered.index(segment_id)
    except ValueError:
        return []
    current_markers = _markers_for(seg.source_text)
    near_ids = ordered[max(0, pos - 2) : pos] + ordered[pos + 1 : pos + 3]
    neighbor_markers = {
        marker
        for near_id in near_ids
        for marker in _markers_for(segments_by_id.get(near_id).source_text if near_id in segments_by_id else "")
    }
    rendered = _normalise_vi(translation)
    current_hits = sum(1 for marker in current_markers if _contains_any(rendered, marker[1]))
    neighbor_hits = [
        marker for marker in neighbor_markers if marker not in current_markers and _contains_any(rendered, marker[1])
    ]
    issues: list[str] = []
    if neighbor_hits and current_hits == 0:
        issues.append(SOURCE_ALIGNMENT_DRIFT)
    if len(neighbor_hits) >= 2:
        issues.append(SEGMENT_MERGE_ERROR)
    if _multiple_dialogue_acts(rendered) and _has_cross_speaker_neighbor(seg, near_ids, segments_by_id):
        issues.append(SEGMENT_MERGE_ERROR)
    return list(dict.fromkeys(issues))


def merge_metadata_validation(
    result: TranslationResult,
    *,
    errors: Iterable[str],
) -> TranslationResult:
    parsed = ensure_translation_result(result)
    errors = list(dict.fromkeys(str(error) for error in errors if str(error).strip()))
    if not errors:
        return parsed
    metadata = parsed.metadata
    flags = list(dict.fromkeys([*metadata.reason_flags, *errors]))
    validation = dict(metadata.validation)
    validation["valid"] = False
    validation["issues"] = list(dict.fromkeys([*(validation.get("issues") or []), *errors]))
    return TranslationResult(
        parsed.translation,
        type(metadata)(
            confidence=metadata.confidence,
            needs_review=True,
            retry_count=metadata.retry_count,
            translation_method=metadata.translation_method,
            reason_flags=flags,
            validation=validation,
        ),
    )


def _ordered_ids(segments_by_id: dict[int, TranslatedSegment]) -> list[int]:
    return [
        sid
        for sid, _seg in sorted(
            segments_by_id.items(),
            key=lambda item: (item[1].start, stable_segment_id(item[1]), item[0]),
        )
    ]


def _markers_for(source: str) -> set[tuple[str, tuple[str, ...]]]:
    return {item for item in _SOURCE_MARKERS if item[0] in (source or "")}


def _normalise_source(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _normalise_vi(text: str) -> str:
    words = _VI_WORD_RE.findall(text or "")
    return " ".join(_strip_accents(word).casefold() for word in words)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d")


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _multiple_dialogue_acts(text: str) -> bool:
    if text.count("?") + text.count("!") >= 2:
        return True
    return bool(re.search(r"\b(thi sao|toi khong biet|sao .+ khong|cam on .+ sep)\b", text))


def _has_cross_speaker_neighbor(
    seg: TranslatedSegment,
    near_ids: list[int],
    segments_by_id: dict[int, TranslatedSegment],
) -> bool:
    speaker = normalized_speaker_id(seg.speaker_id)
    if speaker == UNKNOWN_SPEAKER:
        return True
    for near_id in near_ids:
        near = segments_by_id.get(near_id)
        if near is None:
            continue
        near_speaker = normalized_speaker_id(near.speaker_id)
        if near_speaker != UNKNOWN_SPEAKER and near_speaker != speaker:
            return True
    return False
