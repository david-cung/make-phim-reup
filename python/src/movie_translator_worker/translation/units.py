"""Generic translation-unit ownership and contract helpers.

The translation engine works on source-owned units. Surrounding units
can be read as context, but they never become owned content for the
current result unless they are explicitly listed as targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import TranslatedSegment, TranslationResult, ensure_translation_result


OUTPUT_SCHEMA_ERROR = "OUTPUT_SCHEMA_ERROR"
MISSING_RESULT = "MISSING_RESULT"
DUPLICATE_RESULT = "DUPLICATE_RESULT"
UNKNOWN_RESULT_ID = "UNKNOWN_RESULT_ID"
SEMANTIC_OMISSION = "SEMANTIC_OMISSION"
SEMANTIC_ADDITION = "SEMANTIC_ADDITION"
CONTENT_LEAKAGE = "CONTENT_LEAKAGE"
ALIGNMENT_MISMATCH = "ALIGNMENT_MISMATCH"
CONVERSATION_BOUNDARY_VIOLATION = "CONVERSATION_BOUNDARY_VIOLATION"
TARGET_LANGUAGE_CONFORMITY_ERROR = "TARGET_LANGUAGE_CONFORMITY_ERROR"
UNRESOLVED_TRANSLATION_FAILURE = "UNRESOLVED_TRANSLATION_FAILURE"

GENERIC_TRANSLATION_ISSUE_CODES = {
    OUTPUT_SCHEMA_ERROR,
    MISSING_RESULT,
    DUPLICATE_RESULT,
    UNKNOWN_RESULT_ID,
    SEMANTIC_OMISSION,
    SEMANTIC_ADDITION,
    CONTENT_LEAKAGE,
    ALIGNMENT_MISMATCH,
    CONVERSATION_BOUNDARY_VIOLATION,
    TARGET_LANGUAGE_CONFORMITY_ERROR,
    UNRESOLVED_TRANSLATION_FAILURE,
}


@dataclass(frozen=True)
class TimeRange:
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class SpeakerRef:
    ref: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"speaker_ref": self.ref, "speakerRef": self.ref, "confidence": self.confidence}


@dataclass(frozen=True)
class AtomicSourceUnit:
    unit_id: str
    source_text: str
    time_range: TimeRange = field(default_factory=TimeRange)
    speaker: SpeakerRef = field(default_factory=SpeakerRef)
    source_type: str = "transcript"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unitId": self.unit_id,
            "source_text": self.source_text,
            "sourceText": self.source_text,
            "time_range": self.time_range.to_dict(),
            "timeRange": self.time_range.to_dict(),
            "speaker_ref": self.speaker.ref,
            "speakerRef": self.speaker.ref,
            "speakerConfidence": self.speaker.confidence,
            "source_type": self.source_type,
            "sourceType": self.source_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConversationBoundary:
    from_unit_id: str | None
    to_unit_id: str
    relation: str
    confidence: float
    discourse_state: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fromUnitId": self.from_unit_id,
            "toUnitId": self.to_unit_id,
            "relation": self.relation,
            "confidence": round(float(self.confidence), 3),
            "discourseState": self.discourse_state,
        }


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    unit_ids: list[str]
    speaker_ref: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turnId": self.turn_id,
            "unitIds": list(self.unit_ids),
            "speakerRef": self.speaker_ref,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass(frozen=True)
class SemanticGroup:
    semantic_group_id: str
    member_unit_ids: list[str]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "semanticGroupId": self.semantic_group_id,
            "memberUnitIds": list(self.member_unit_ids),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass(frozen=True)
class TranslationProvenance:
    translation_id: str
    source_unit_ids: list[str]
    context_unit_ids: list[str] = field(default_factory=list)
    semantic_group_id: str | None = None
    conversation_turn_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "translationId": self.translation_id,
            "sourceUnitIds": list(self.source_unit_ids),
            "contextUnitIds": list(self.context_unit_ids),
            "semanticGroupId": self.semantic_group_id,
            "conversationTurnId": self.conversation_turn_id,
        }


@dataclass(frozen=True)
class TranslationContractIssue:
    code: str
    severity: str
    unit_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "unitId": self.unit_id,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class TranslationContractReport:
    valid: bool
    issues: list[TranslationContractIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def unit_id_for_segment(seg: TranslatedSegment) -> str:
    return seg.source_segment_id or f"seg_{seg.id:05d}"


def source_unit_from_segment(
    seg: TranslatedSegment,
    *,
    source_type: str = "transcript",
) -> AtomicSourceUnit:
    speaker_ref = _known_speaker(seg.speaker_id)
    return AtomicSourceUnit(
        unit_id=unit_id_for_segment(seg),
        source_text=seg.source_text,
        time_range=TimeRange(start=seg.start, end=seg.end),
        speaker=SpeakerRef(
            ref=speaker_ref,
            confidence=float(seg.speaker_confidence or 0.0) if speaker_ref else 0.0,
        ),
        source_type=source_type,
        metadata={
            key: value
            for key, value in {
                "rawText": seg.raw_source_text,
                "normalizedText": seg.normalized_source_text,
                "sourceSegmentId": seg.source_segment_id,
                "sourceSubSegmentId": seg.source_sub_segment_id,
                "sourceQuality": seg.source_quality,
                "semanticFacts": seg.semantic_facts,
            }.items()
            if value not in (None, "", [], {})
        },
    )


def source_units_from_segments(segments: Iterable[TranslatedSegment]) -> list[AtomicSourceUnit]:
    return [source_unit_from_segment(seg) for seg in segments]


def context_unit_ids(
    *,
    before: Iterable[int],
    after: Iterable[int],
    segments_by_id: Mapping[int, TranslatedSegment],
) -> list[str]:
    ids: list[str] = []
    for sid in list(before) + list(after):
        seg = segments_by_id.get(sid)
        if seg is not None:
            ids.append(unit_id_for_segment(seg))
    return ids


def provenance_for_segment(
    *,
    seg: TranslatedSegment,
    context_ids: Iterable[str] = (),
    semantic_group_id: str | None = None,
    conversation_turn_id: str | None = None,
) -> TranslationProvenance:
    unit_id = unit_id_for_segment(seg)
    return TranslationProvenance(
        translation_id=f"tr_{unit_id}",
        source_unit_ids=[unit_id],
        context_unit_ids=list(dict.fromkeys(context_ids)),
        semantic_group_id=semantic_group_id,
        conversation_turn_id=conversation_turn_id,
    )


def resolve_conversation_structure(
    units: list[AtomicSourceUnit],
    *,
    continuation_gap: float = 1.25,
    scene_gap: float = 3.0,
) -> tuple[list[ConversationTurn], list[ConversationBoundary], list[SemanticGroup]]:
    if not units:
        return [], [], []

    boundaries: list[ConversationBoundary] = []
    turns: list[ConversationTurn] = []
    current_units: list[str] = [units[0].unit_id]
    current_speaker = units[0].speaker.ref
    current_confidence = units[0].speaker.confidence

    for previous, current in zip(units, units[1:]):
        relation, confidence = _structural_relation(
            previous,
            current,
            continuation_gap=continuation_gap,
            scene_gap=scene_gap,
        )
        boundaries.append(
            ConversationBoundary(
                from_unit_id=previous.unit_id,
                to_unit_id=current.unit_id,
                relation=relation,
                confidence=confidence,
                discourse_state=_discourse_state(previous.source_text),
            )
        )
        if relation == "same_speaker_continuation" and confidence >= 0.8:
            current_units.append(current.unit_id)
            current_confidence = min(current_confidence or confidence, current.speaker.confidence or confidence)
            continue
        turns.append(
            ConversationTurn(
                turn_id=f"turn_{len(turns) + 1:05d}",
                unit_ids=list(current_units),
                speaker_ref=current_speaker,
                confidence=current_confidence if current_speaker else 0.0,
            )
        )
        current_units = [current.unit_id]
        current_speaker = current.speaker.ref
        current_confidence = current.speaker.confidence

    turns.append(
        ConversationTurn(
            turn_id=f"turn_{len(turns) + 1:05d}",
            unit_ids=list(current_units),
            speaker_ref=current_speaker,
            confidence=current_confidence if current_speaker else 0.0,
        )
    )
    semantic_groups = [
        SemanticGroup(
            semantic_group_id=f"sg_{idx + 1:05d}",
            member_unit_ids=list(turn.unit_ids),
            confidence=turn.confidence if len(turn.unit_ids) > 1 else 1.0,
        )
        for idx, turn in enumerate(turns)
    ]
    return turns, boundaries, semantic_groups


def validate_translation_contract(
    *,
    expected_unit_ids: Iterable[int | str],
    results: Mapping[int | str, str | TranslationResult],
) -> TranslationContractReport:
    expected = [str(unit_id) for unit_id in expected_unit_ids]
    actual = [str(unit_id) for unit_id in results.keys()]
    expected_set = set(expected)
    actual_set = set(actual)
    issues: list[TranslationContractIssue] = []

    for unit_id in expected:
        if unit_id not in actual_set:
            issues.append(
                TranslationContractIssue(
                    code=MISSING_RESULT,
                    severity="error",
                    unit_id=unit_id,
                )
            )
    for unit_id in sorted(actual_set - expected_set):
        issues.append(
            TranslationContractIssue(
                code=UNKNOWN_RESULT_ID,
                severity="error",
                unit_id=unit_id,
            )
        )
    for unit_id in sorted({unit_id for unit_id in actual if actual.count(unit_id) > 1}):
        issues.append(
            TranslationContractIssue(
                code=DUPLICATE_RESULT,
                severity="error",
                unit_id=unit_id,
            )
        )
    for unit_id, value in results.items():
        if str(unit_id) not in expected_set:
            continue
        try:
            parsed = ensure_translation_result(value)
        except Exception as exc:  # pragma: no cover - defensive around protocol inputs
            issues.append(
                TranslationContractIssue(
                    code=OUTPUT_SCHEMA_ERROR,
                    severity="error",
                    unit_id=str(unit_id),
                    evidence={"error": str(exc)},
                )
            )
            continue
        if not parsed.translation.strip():
            issues.append(
                TranslationContractIssue(
                    code=SEMANTIC_OMISSION,
                    severity="error",
                    unit_id=str(unit_id),
                    evidence={"reason": "empty_translation"},
                )
            )

    return TranslationContractReport(valid=not issues, issues=_unique_issues(issues))


def ownership_payload(
    *,
    source_unit_id: str,
    context_unit_ids: Iterable[str],
    target: bool,
) -> dict[str, Any]:
    return {
        "target": bool(target),
        "sourceUnitIds": [source_unit_id],
        "contextUnitIds": list(dict.fromkeys(context_unit_ids)),
        "rule": "context is read-only evidence; only sourceUnitIds are owned by this result",
    }


def _unique_issues(issues: Iterable[TranslationContractIssue]) -> list[TranslationContractIssue]:
    out: list[TranslationContractIssue] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for issue in issues:
        key = (
            issue.code,
            issue.severity,
            issue.unit_id,
            repr(sorted((str(k), repr(v)) for k, v in issue.evidence.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _known_speaker(speaker_id: str | None) -> str | None:
    speaker = (speaker_id or "").strip()
    if not speaker or speaker.upper() == "UNKNOWN":
        return None
    return speaker


def _structural_relation(
    previous: AtomicSourceUnit,
    current: AtomicSourceUnit,
    *,
    continuation_gap: float,
    scene_gap: float,
) -> tuple[str, float]:
    gap = _gap(previous, current)
    if gap is not None and gap >= scene_gap:
        return "scene_boundary", 0.95
    if previous.speaker.ref and current.speaker.ref:
        confidence = min(previous.speaker.confidence, current.speaker.confidence)
        if previous.speaker.ref == current.speaker.ref:
            if gap is None or gap <= continuation_gap:
                return "same_speaker_continuation", max(0.5, confidence)
            return "same_speaker_new_turn", max(0.5, confidence)
        return "speaker_transition", max(0.5, confidence)
    return "unknown_speaker_transition", 0.35


def _gap(previous: AtomicSourceUnit, current: AtomicSourceUnit) -> float | None:
    if previous.time_range.end is None or current.time_range.start is None:
        return None
    return max(0.0, float(current.time_range.start) - float(previous.time_range.end))


def _discourse_state(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "UNKNOWN"
    if stripped.endswith(("...", "…", "-", "—")):
        return "INTERRUPTED"
    if stripped.endswith((",", ";", ":")):
        return "CONTINUATION"
    return "COMPLETE"
