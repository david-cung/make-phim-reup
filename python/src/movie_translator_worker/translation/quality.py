"""Lightweight validation for context-aware translation output."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .memory import (
    PRONOUN_PLAN_ENFORCE_THRESHOLD,
    PronounPlan,
    TranslationMemory,
    extract_name_mentions,
    suspicious_pronoun_shift,
)
from .models import (
    TranslatedSegment,
    TranslateOptions,
    TranslationMetadata,
    TranslationResult,
    ensure_translation_result,
)
from ..source_protection import (
    source_protection_payload,
    validate_translation_against_source,
)
from .semantic_realization import (
    SEMANTIC_ERROR_CODES,
    analyze_source_semantics,
    compact_semantic_payload,
    realization_critic_issues,
    requires_deeper_reasoning,
    source_has_contextual_terms,
)

SERIOUS_REASON_FLAGS = {
    "AMBIGUOUS_PRONOUN",
    "AMBIGUOUS_REFERENT",
    "POSSIBLE_GENDER_AMBIGUITY",
    "MEANING_UNCERTAIN",
    "POSSIBLE_PRONOUN_INCONSISTENCY",
    "POSSIBLE_REVERSED_RELATIONSHIP",
    "POSSIBLE_TITLE_INCONSISTENCY",
    "POSSIBLE_MEANING_CHANGE",
    "POSSIBLE_MISSING_MEANING",
    "POSSIBLE_HALLUCINATION",
    "POSSIBLE_CHARACTER_REFERENCE_CONFLICT",
    "POSSIBLE_GLOBAL_INCONSISTENCY",
}.union(SEMANTIC_ERROR_CODES)

SEMANTIC_REVISION_FLAGS = {
    "POSSIBLE_MEANING_CHANGE",
    "POSSIBLE_MISSING_MEANING",
    "POSSIBLE_HALLUCINATION",
    "POSSIBLE_CHARACTER_REFERENCE_CONFLICT",
    "POSSIBLE_GLOBAL_INCONSISTENCY",
}.union(SERIOUS_REASON_FLAGS)


def needs_retry(result: str | TranslationResult, options: TranslateOptions) -> bool:
    parsed = ensure_translation_result(result)
    metadata = parsed.metadata
    if metadata.confidence is not None and metadata.confidence < options.low_confidence_threshold:
        return True
    return bool(SERIOUS_REASON_FLAGS.intersection(metadata.reason_flags))


def validate_result(
    *,
    segment_id: int,
    source: str,
    result: str | TranslationResult,
    memory: TranslationMemory,
    options: TranslateOptions,
) -> TranslationResult:
    return semantic_validate_result(
        segment_id=segment_id,
        source=source,
        source_protection=None,
        result=result,
        memory=memory,
        options=options,
        context_before=[],
        context_after=[],
    )


def semantic_validate_result(
    *,
    segment_id: int,
    source: str,
    source_protection: dict | None = None,
    result: str | TranslationResult,
    memory: TranslationMemory,
    options: TranslateOptions,
    context_before: list[TranslatedSegment],
    context_after: list[TranslatedSegment],
    revision_attempt: int = 0,
) -> TranslationResult:
    parsed = ensure_translation_result(result)
    flags = list(dict.fromkeys(parsed.metadata.reason_flags))
    confidence = parsed.metadata.confidence
    if confidence is None:
        confidence = infer_confidence(flags)
    pronoun_plan = memory.pronoun_plan_for_segment(segment_id)
    representation = analyze_source_semantics(
        segment_id=segment_id,
        source=source,
        speaker_id=None,
        listener_id=None,
        pronoun_plan=pronoun_plan,
    )

    if suspicious_pronoun_shift(
        segment_id=segment_id,
        translation=parsed.translation,
        memory=memory,
    ):
        flags.append("POSSIBLE_PRONOUN_INCONSISTENCY")
    flags.extend(
        validate_pronoun_plan(
            translation=parsed.translation,
            plan=pronoun_plan,
        )
    )
    if should_validate_semantically(
        parsed,
        memory=memory,
        segment_id=segment_id,
        source=source,
    ):
        flags.extend(
            semantic_issues(
                segment_id=segment_id,
                source=source,
                source_protection=source_protection,
                translation=parsed.translation,
                memory=memory,
                context_before=context_before,
                context_after=context_after,
                representation=representation,
            )
        )
    flags = list(dict.fromkeys(flags))

    validation_issues = []
    if not parsed.translation.strip():
        validation_issues.append("EMPTY_TRANSLATION")
    if flags:
        validation_issues.extend(flags)
    validation_confidence = validation_confidence_for(flags)
    translation_confidence = max(0.0, min(1.0, float(confidence)))
    final_confidence = round((translation_confidence + validation_confidence) / 2, 3)

    needs_review = (
        parsed.metadata.needs_review
        or bool(SERIOUS_REASON_FLAGS.intersection(flags))
        or confidence < options.low_confidence_threshold
    )

    metadata = replace(
        parsed.metadata,
        confidence=final_confidence,
        needs_review=needs_review,
        reason_flags=list(dict.fromkeys(flags)),
        validation={
            "valid": not needs_review,
            "confidence": validation_confidence,
            "translation_confidence": translation_confidence,
            "translationConfidence": translation_confidence,
            "validation_confidence": validation_confidence,
            "validationConfidence": validation_confidence,
            "final_confidence": final_confidence,
            "finalConfidence": final_confidence,
            "issues": list(dict.fromkeys(validation_issues)),
            "sourceLength": len(source.strip()),
            "sourceProtection": source_protection
            or source_protection_payload(segment_id=segment_id, text=source),
            "semanticRepresentation": compact_semantic_payload(representation),
            "ambiguityScore": representation.ambiguity_score,
            "addressResolution": memory.address_debug_for_segment(segment_id),
            "semanticCritic": {
                "checked": should_validate_semantically(
                    parsed,
                    memory=memory,
                    segment_id=segment_id,
                    source=source,
                ),
                "errorTaxonomy": sorted(SEMANTIC_ERROR_CODES),
            },
            "revisionAttempt": revision_attempt,
            "checked": should_validate_semantically(
                parsed,
                memory=memory,
                segment_id=segment_id,
                source=source,
            ),
        },
    )
    return TranslationResult(parsed.translation, metadata)


def infer_confidence(flags: Iterable[str]) -> float:
    flags = set(flags)
    if flags.intersection(SERIOUS_REASON_FLAGS):
        return 0.72
    if flags:
        return 0.82
    return 0.9


def validate_pronoun_plan(
    *,
    translation: str,
    plan: PronounPlan | None,
) -> list[str]:
    if plan is None or plan.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD:
        return []
    expected = {
        item.casefold()
        for item in (plan.self_pronoun, plan.target_pronoun)
        if item and "/" not in item
    }
    if not expected:
        return []
    words = {word.casefold() for word in _vi_words(translation)}
    if not words:
        return []
    issues: list[str] = []
    wrong_pronouns = words.intersection(_KNOWN_VI_PRONOUNS - expected)
    if wrong_pronouns:
        issues.append("POSSIBLE_PRONOUN_INCONSISTENCY")
    if plan.self_pronoun and plan.target_pronoun:
        self_ref = plan.self_pronoun.casefold()
        target_ref = plan.target_pronoun.casefold()
        if self_ref in words and target_ref not in words and _looks_reversed(plan, words):
            issues.append("POSSIBLE_REVERSED_RELATIONSHIP")
        if target_ref in {"sếp", "bác sĩ"} and target_ref not in words:
            issues.append("POSSIBLE_TITLE_INCONSISTENCY")
    return list(dict.fromkeys(issues))


def should_validate_semantically(
    result: str | TranslationResult,
    *,
    memory: TranslationMemory,
    segment_id: int,
    source: str,
) -> bool:
    parsed = ensure_translation_result(result)
    if parsed.metadata.needs_review or parsed.metadata.retry_count > 0:
        return True
    if parsed.metadata.confidence is not None and parsed.metadata.confidence < 0.86:
        return True
    if SEMANTIC_REVISION_FLAGS.intersection(parsed.metadata.reason_flags):
        return True
    plan = memory.pronoun_plan_for_segment(segment_id)
    if plan is not None and plan.confidence >= PRONOUN_PLAN_ENFORCE_THRESHOLD:
        return True
    representation = analyze_source_semantics(
        segment_id=segment_id,
        source=source,
        pronoun_plan=plan,
    )
    if requires_deeper_reasoning(representation):
        return True
    if _SOURCE_NEGATION_RE.search(source):
        return True
    if _SOURCE_PROTECTED_FACT_RE.search(source):
        return True
    if source_has_contextual_terms(source):
        return True
    return bool(extract_name_mentions(source))


def semantic_issues(
    *,
    segment_id: int,
    source: str,
    source_protection: dict | None = None,
    translation: str,
    memory: TranslationMemory,
    context_before: list[TranslatedSegment],
    context_after: list[TranslatedSegment],
    representation=None,
) -> list[str]:
    base = TranslationResult(translation)
    if not should_validate_semantically(
        base,
        memory=memory,
        segment_id=segment_id,
        source=source,
    ):
        return []
    issues: list[str] = []
    protection = source_protection or source_protection_payload(
        segment_id=segment_id,
        text=source,
    )
    plan = memory.pronoun_plan_for_segment(segment_id)
    if representation is None:
        representation = analyze_source_semantics(
            segment_id=segment_id,
            source=source,
            pronoun_plan=plan,
        )
    phase7_issues = validate_translation_against_source(
        source=source,
        translation=translation,
        protection=protection,
    )
    phase7_map = {
        "UNTRANSLATED_CHINESE": "POSSIBLE_MISSING_MEANING",
        "NUMBER_MISMATCH": "POSSIBLE_MEANING_CHANGE",
        "DURATION_MISMATCH": "POSSIBLE_MEANING_CHANGE",
        "QUANTITY_MISMATCH": "POSSIBLE_MEANING_CHANGE",
        "MISSING_NEGATION": "POSSIBLE_MEANING_CHANGE",
        "QUESTION_CHANGED_TO_STATEMENT": "POSSIBLE_MEANING_CHANGE",
        "MISSING_ACTION": "POSSIBLE_MISSING_MEANING",
    }
    issues.extend(phase7_map.get(issue, "POSSIBLE_MEANING_CHANGE") for issue in phase7_issues)
    if _contains_cjk(translation):
        issues.append("POSSIBLE_MISSING_MEANING")
    if _missing_negation(source, translation):
        issues.append("POSSIBLE_MEANING_CHANGE")
    if _missing_character_reference(source, translation, memory):
        issues.append("POSSIBLE_CHARACTER_REFERENCE_CONFLICT")
    if _looks_hallucinated(source, translation, context_before, context_after):
        issues.append("POSSIBLE_HALLUCINATION")
    if _too_short_for_source(source, translation):
        issues.append("POSSIBLE_MISSING_MEANING")
    issues.extend(
        realization_critic_issues(
            source=source,
            translation=translation,
            representation=representation,
            pronoun_plan=plan,
        )
    )
    issues.extend(
        memory.address_consistency_issues(
            segment_id=segment_id,
            source=source,
            translation=translation,
        )
    )
    return list(dict.fromkeys(issues))


def validation_confidence_for(flags: Iterable[str]) -> float:
    flags = set(flags)
    if flags.intersection(SEMANTIC_ERROR_CODES):
        return 0.54
    if flags.intersection(
        {
            "POSSIBLE_MEANING_CHANGE",
            "POSSIBLE_MISSING_MEANING",
            "POSSIBLE_HALLUCINATION",
            "POSSIBLE_CHARACTER_REFERENCE_CONFLICT",
        }
    ):
        return 0.56
    if flags.intersection(SERIOUS_REASON_FLAGS):
        return 0.68
    if flags:
        return 0.78
    return 0.94


def select_best_candidate(
    *,
    segment_id: int,
    source: str,
    candidates: list[TranslationResult],
    memory: TranslationMemory,
    options: TranslateOptions,
    context_before: list[TranslatedSegment] | None = None,
    context_after: list[TranslatedSegment] | None = None,
) -> TranslationResult | None:
    if not candidates:
        return None
    scored: list[tuple[float, TranslationResult]] = []
    for index, candidate in enumerate(candidates):
        validated = semantic_validate_result(
            segment_id=segment_id,
            source=source,
            result=candidate,
            memory=memory,
            options=options,
            context_before=context_before or [],
            context_after=context_after or [],
        )
        score = _candidate_score(validated) - index * 0.001
        scored.append((score, validated))
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]
    metadata = replace(
        best.metadata,
        translation_method=best.metadata.translation_method or "candidate_selection",
        validation={
            **best.metadata.validation,
            "candidateCount": len(candidates),
            "selectedCandidateScore": round(scored[0][0], 3),
        },
    )
    return TranslationResult(best.translation, metadata)


def global_consistency_issues(
    *,
    translations: dict[int, TranslationResult],
    segments_by_id: dict[int, TranslatedSegment],
    memory: TranslationMemory,
) -> dict[int, list[str]]:
    issues: dict[int, list[str]] = {}
    name_forms: dict[str, dict[str, list[int]]] = {}
    for sid, result in translations.items():
        seg = segments_by_id.get(sid)
        if seg is None:
            continue
        for name in extract_name_mentions(seg.source_text):
            forms = name_forms.setdefault(name, {})
            key = _name_rendering_hint(result.translation)
            forms.setdefault(key, []).append(sid)
    for forms in name_forms.values():
        real_forms = {key: ids for key, ids in forms.items() if key}
        if len(real_forms) <= 1:
            continue
        canonical = max(real_forms.items(), key=lambda item: len(item[1]))[0]
        for rendering, ids in real_forms.items():
            if rendering == canonical:
                continue
            for sid in ids:
                issues.setdefault(sid, []).append("POSSIBLE_GLOBAL_INCONSISTENCY")
                issues[sid].append("POSSIBLE_CHARACTER_REFERENCE_CONFLICT")

    pronoun_pairs: dict[tuple[str | None, str | None], dict[tuple[str, str], list[int]]] = {}
    for sid, result in translations.items():
        plan = memory.pronoun_plan_for_segment(sid)
        if plan is None or plan.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD:
            continue
        pair = _first_pronoun_pair(result.translation)
        if pair is None:
            continue
        key = (plan.speaker, plan.listener)
        pronoun_pairs.setdefault(key, {}).setdefault(pair, []).append(sid)
    for pairs in pronoun_pairs.values():
        if len(pairs) <= 1:
            continue
        canonical = max(pairs.items(), key=lambda item: len(item[1]))[0]
        for pair, ids in pairs.items():
            if pair == canonical:
                continue
            for sid in ids:
                issues.setdefault(sid, []).append("POSSIBLE_GLOBAL_INCONSISTENCY")
                issues[sid].append("POSSIBLE_PRONOUN_INCONSISTENCY")
    return {sid: list(dict.fromkeys(flags)) for sid, flags in issues.items()}


def _candidate_score(result: TranslationResult) -> float:
    validation = result.metadata.validation
    final_confidence = validation.get("finalConfidence")
    try:
        score = float(final_confidence)
    except (TypeError, ValueError):
        score = float(result.metadata.confidence or 0.75)
    if result.metadata.needs_review:
        score -= 0.18
    score -= 0.04 * len(result.metadata.reason_flags)
    return score


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_SOURCE_NEGATION_RE = re.compile(
    r"(不|没|沒有|不是|不要|不能|别|無|未|never|not|no\b|don't|can't)",
    re.IGNORECASE,
)
_SOURCE_PROTECTED_FACT_RE = re.compile(
    r"(\d|[零一二两三四五六七八九十百千万]+(?:个)?(?:小时|分钟|分鐘|年|个月|月|天|人|次|个)|面试|面試|介绍|介紹|结婚|結婚|等)"
)
_VI_NEGATION_WORDS = {
    "không",
    "chưa",
    "đừng",
    "chẳng",
    "chả",
    "khỏi",
    "đâu",
}


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _missing_negation(source: str, translation: str) -> bool:
    if not _SOURCE_NEGATION_RE.search(source):
        return False
    lowered = _normalise_text(translation)
    if re.search(r"没\s*什么\s*难|沒\s*什麼\s*難|没什么难|沒什麼難", source):
        if any(word in lowered for word in ("đơn giản", "dễ", "khó gì")):
            return False
    words = {word.casefold() for word in _vi_words(translation)}
    return not bool(words.intersection(_VI_NEGATION_WORDS))


def _missing_character_reference(
    source: str,
    translation: str,
    memory: TranslationMemory,
) -> bool:
    names = extract_name_mentions(source)
    if not names:
        return False
    remembered = memory.names
    for name in names:
        if name in translation:
            continue
        rendered = remembered.get(name)
        if rendered and _name_rendering_hint(rendered) in _normalise_text(translation):
            continue
        if _likely_title(name) and any(
            title in _normalise_text(translation)
            for title in _title_translations(name)
        ):
            continue
        return True
    return False


def _looks_hallucinated(
    source: str,
    translation: str,
    context_before: list[TranslatedSegment],
    context_after: list[TranslatedSegment],
) -> bool:
    if _contains_cjk(source):
        return False
    source_words = set(_ascii_words(source))
    context_words = {
        word
        for seg in context_before + context_after
        for word in _ascii_words(seg.source_text)
    }
    translation_words = set(_ascii_words(translation))
    if len(translation_words) < 8:
        return False
    unexplained = translation_words - source_words - context_words - _COMMON_VI_WORDS
    return len(unexplained) >= max(6, len(translation_words) // 2)


def _too_short_for_source(source: str, translation: str) -> bool:
    source_units = len(_CJK_RE.findall(source)) or len(_ascii_words(source))
    translated_units = len(_vi_words(translation))
    return source_units >= 10 and translated_units <= 2


def _name_rendering_hint(text: str) -> str:
    words = _vi_words(text)
    proper = [word for word in words if word[:1].isupper()]
    if len(proper) >= 2:
        return " ".join(proper[:3]).casefold()
    if proper:
        return proper[0].casefold()
    lowered = _normalise_text(text)
    for title in ("sếp", "bác sĩ", "thầy", "cô", "mẹ", "bố", "ông", "bà"):
        if title in lowered:
            return title
    return ""


def _first_pronoun_pair(text: str) -> tuple[str, str] | None:
    found = [
        word.casefold()
        for word in _vi_words(text)
        if word.casefold() in _KNOWN_VI_PRONOUNS
    ]
    if len(found) < 2:
        return None
    return found[0], found[1]


def _likely_title(name: str) -> bool:
    return any(token in name for token in ("总", "老板", "老师", "医生", "哥", "姐", "妈", "爸"))


def _title_translations(name: str) -> list[str]:
    if "总" in name or "老板" in name:
        return ["sếp", "tổng giám đốc", "giám đốc"]
    if "医生" in name:
        return ["bác sĩ"]
    if "老师" in name:
        return ["thầy", "cô giáo", "giáo viên"]
    if "哥" in name:
        return ["anh"]
    if "姐" in name:
        return ["chị"]
    if "妈" in name:
        return ["mẹ"]
    if "爸" in name:
        return ["bố", "ba"]
    return []


def _normalise_text(text: str) -> str:
    return " ".join(_vi_words(text)).casefold()


def _ascii_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ỹ]+", text.casefold())


_COMMON_VI_WORDS = {
    "anh",
    "chị",
    "em",
    "tôi",
    "mình",
    "là",
    "đã",
    "đang",
    "sẽ",
    "có",
    "không",
    "chưa",
    "này",
    "đó",
    "rồi",
    "và",
    "nhưng",
    "vì",
    "nên",
    "hãy",
    "nghe",
    "nói",
    "đi",
    "về",
    "ở",
    "trong",
    "ngoài",
    "với",
    "cho",
    "của",
}


def _vi_words(text: str) -> list[str]:
    words: list[str] = []
    for raw in text.replace("/", " ").split():
        token = "".join(ch for ch in raw if ch.isalpha())
        if token:
            words.append(token)
    return words


_KNOWN_VI_PRONOUNS = {
    "anh",
    "chị",
    "em",
    "tôi",
    "mình",
    "con",
    "mẹ",
    "bố",
    "ba",
    "cháu",
    "ông",
    "bà",
    "sếp",
    "cậu",
    "thầy",
    "cô",
    "bác",
    "chú",
}


def _looks_reversed(plan: PronounPlan, words: set[str]) -> bool:
    if plan.self_pronoun == "em" and {"anh", "chị"}.intersection(words):
        return True
    if plan.self_pronoun == "con" and {"mẹ", "bố", "ba"}.intersection(words):
        return True
    if plan.self_pronoun == "cháu" and {"ông", "bà"}.intersection(words):
        return True
    return False
