"""Phase 8 semantic representation and Vietnamese realization safeguards.

This module keeps source-language facts separate from Vietnamese surface
choices. It intentionally models relationship/address words as semantic
concepts first, then lets prompt guidance and validation decide whether a
Vietnamese form of address is justified by context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .memory import PRONOUN_PLAN_ENFORCE_THRESHOLD, PronounPlan


SEMANTIC_ERROR_CODES = {
    "REFERENT_ERROR",
    "RELATIONSHIP_REALIZATION_ERROR",
    "POSSESSION_ERROR",
    "DIRECT_ADDRESS_ERROR",
    "UNSUPPORTED_PRONOUN_INFERENCE",
    "RELATIONSHIP_HALLUCINATION",
    "RELATIONSHIP_INFORMATION_LOSS",
    "TITLE_ROLE_ERROR",
    "SPEAKER_LISTENER_ERROR",
    "CONTEXT_CONSISTENCY_ERROR",
    "SEMANTIC_OMISSION",
    "SEMANTIC_ADDITION",
    "UNNATURAL_VIETNAMESE",
    "UNJUSTIFIED_ADDRESS_SHIFT",
    "SPEAKER_LISTENER_MISMATCH",
    "RELATIONSHIP_CONTRADICTION",
    "ADDRESS_PAIR_INCONSISTENCY",
    "UNSUPPORTED_SOCIAL_PRONOUN",
    "WRONG_RELATIONSHIP_DIRECTION",
    "THIRD_PERSON_AS_DIRECT_ADDRESS",
    "DIRECT_ADDRESS_AS_THIRD_PERSON",
    "TITLE_RELATIONSHIP_CONFUSION",
    "CHARACTER_IDENTITY_CONFLICT",
    "MEMORY_CONTAMINATION",
}

AMBIGUITY_DEEP_REASONING_THRESHOLD = 0.70


@dataclass(frozen=True)
class SemanticTerm:
    source: str
    canonical: str
    category: str
    relation: str | None = None
    domain: str | None = None
    social_address_possible: bool = False
    gender_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "canonical": self.canonical,
            "category": self.category,
            "relation": self.relation,
            "domain": self.domain,
            "socialAddressPossible": self.social_address_possible,
            "genderHint": self.gender_hint,
        }


@dataclass(frozen=True)
class SourceSemanticRepresentation:
    segment_id: int | str
    source_text: str
    speaker_id: str | None = None
    listener_id: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    propositions: list[dict[str, Any]] = field(default_factory=list)
    terms: list[SemanticTerm] = field(default_factory=list)
    discourse_role: str = "utterance"
    unresolved: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0
    realization_guidance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentId": self.segment_id,
            "speakerId": self.speaker_id,
            "listenerId": self.listener_id,
            "entities": dict(self.entities),
            "propositions": list(self.propositions),
            "terms": [term.to_dict() for term in self.terms],
            "discourseRole": self.discourse_role,
            "unresolved": list(self.unresolved),
            "ambiguityScore": round(float(self.ambiguity_score), 3),
            "realizationGuidance": dict(self.realization_guidance),
        }


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WHITESPACE_RE = re.compile(r"\s+")
_THIRD_PERSON_RE = re.compile(r"(他|她|他们|她们|他們|她們)")
_FIRST_PERSON_RE = re.compile(r"(我|我们|我們)")
_SECOND_PERSON_RE = re.compile(r"(你|您|你们|你們)")
_POSSESSIVE_RE_TEMPLATE = r"(我(?:的)?|我们(?:的)?|我們(?:的)?)(?P<term>{term})"
_ASSERTION_RE_TEMPLATE = r"(他|她|这|這|那|那个|那個|这个|這個).{{0,4}}是.{{0,3}}(我(?:的)?|我们(?:的)?|我們(?:的)?)(?P<term>{term})"
_VOCATIVE_TRAIL_RE = re.compile(r"^[\s，,、。.!！?？:：；;]+")

_TERM_SPECS: tuple[SemanticTerm, ...] = (
    SemanticTerm("哥哥", "older_brother", "KINSHIP", "older_brother_of_speaker", "family", True, "male"),
    SemanticTerm("哥", "older_brother", "KINSHIP", "older_brother_of_speaker", "family", True, "male"),
    SemanticTerm("姐姐", "older_sister", "KINSHIP", "older_sister_of_speaker", "family", True, "female"),
    SemanticTerm("姐", "older_sister", "KINSHIP", "older_sister_of_speaker", "family", True, "female"),
    SemanticTerm("妹妹", "younger_sister", "KINSHIP", "younger_sister_of_speaker", "family", False, "female"),
    SemanticTerm("妹", "younger_sister", "KINSHIP", "younger_sister_of_speaker", "family", False, "female"),
    SemanticTerm("弟弟", "younger_brother", "KINSHIP", "younger_brother_of_speaker", "family", False, "male"),
    SemanticTerm("弟", "younger_brother", "KINSHIP", "younger_brother_of_speaker", "family", False, "male"),
    SemanticTerm("妈妈", "mother", "KINSHIP", "mother_of_speaker", "family", True, "female"),
    SemanticTerm("媽", "mother", "KINSHIP", "mother_of_speaker", "family", True, "female"),
    SemanticTerm("妈", "mother", "KINSHIP", "mother_of_speaker", "family", True, "female"),
    SemanticTerm("母亲", "mother", "KINSHIP", "mother_of_speaker", "family", False, "female"),
    SemanticTerm("爸爸", "father", "KINSHIP", "father_of_speaker", "family", True, "male"),
    SemanticTerm("爸", "father", "KINSHIP", "father_of_speaker", "family", True, "male"),
    SemanticTerm("父亲", "father", "KINSHIP", "father_of_speaker", "family", False, "male"),
    SemanticTerm("爷爷", "grandfather", "KINSHIP", "grandfather_of_speaker", "family", True, "male"),
    SemanticTerm("奶奶", "grandmother", "KINSHIP", "grandmother_of_speaker", "family", True, "female"),
    SemanticTerm("外公", "grandfather", "KINSHIP", "grandfather_of_speaker", "family", True, "male"),
    SemanticTerm("外婆", "grandmother", "KINSHIP", "grandmother_of_speaker", "family", True, "female"),
    SemanticTerm("阿姨", "aunt_or_older_woman", "SOCIAL_KINSHIP_ADDRESS", "older_woman_address", "social_or_family", True, "female"),
    SemanticTerm("叔叔", "uncle_or_older_man", "SOCIAL_KINSHIP_ADDRESS", "older_man_address", "social_or_family", True, "male"),
    SemanticTerm("老师", "teacher", "PROFESSIONAL_TITLE", "teacher_role", "professional", True, None),
    SemanticTerm("醫生", "doctor", "PROFESSIONAL_TITLE", "doctor_role", "professional", True, None),
    SemanticTerm("医生", "doctor", "PROFESSIONAL_TITLE", "doctor_role", "professional", True, None),
    SemanticTerm("警官", "police_officer", "PROFESSIONAL_TITLE", "police_role", "professional", True, None),
    SemanticTerm("老板", "boss", "ORGANIZATIONAL_ROLE", "boss_role", "workplace", True, None),
    SemanticTerm("老闆", "boss", "ORGANIZATIONAL_ROLE", "boss_role", "workplace", True, None),
    SemanticTerm("陈总", "executive", "ORGANIZATIONAL_ROLE", "executive_role", "workplace", True, None),
    SemanticTerm("總", "executive", "ORGANIZATIONAL_ROLE", "executive_role", "workplace", True, None),
    SemanticTerm("总", "executive", "ORGANIZATIONAL_ROLE", "executive_role", "workplace", True, None),
    SemanticTerm("先生", "mister", "HONORIFIC", "honorific_male", "social", True, "male"),
    SemanticTerm("小姐", "miss", "HONORIFIC", "honorific_female", "social", True, "female"),
    SemanticTerm("太太", "madam", "HONORIFIC", "honorific_female", "social", True, "female"),
)

_TERMS_BY_LENGTH = sorted(_TERM_SPECS, key=lambda item: len(item.source), reverse=True)
_TERM_PATTERN = "|".join(re.escape(item.source) for item in _TERMS_BY_LENGTH)

_RELATIONSHIP_VI_MARKERS = {
    "older_brother": ("anh trai", "anh ruột", "anh họ"),
    "older_sister": ("chị gái", "chị ruột", "chị họ"),
    "younger_sister": ("em gái",),
    "younger_brother": ("em trai",),
    "mother": ("mẹ", "má", "mẫu thân"),
    "father": ("bố", "ba", "cha", "phụ thân"),
    "grandfather": ("ông", "ông nội", "ông ngoại"),
    "grandmother": ("bà", "bà nội", "bà ngoại"),
    "aunt_or_older_woman": ("dì", "cô", "bác gái", "thím", "mợ"),
    "uncle_or_older_man": ("chú", "bác", "cậu", "dượng"),
    "teacher": ("thầy", "cô giáo", "giáo viên"),
    "doctor": ("bác sĩ",),
    "police_officer": ("cảnh sát", "sĩ quan", "công an"),
    "boss": ("sếp", "ông chủ", "bà chủ", "chủ"),
    "executive": ("giám đốc", "tổng giám đốc", "sếp"),
}
_TITLE_CANONICALS = {"teacher", "doctor", "police_officer", "boss", "executive"}
_FAMILY_CANONICALS = {
    "older_brother",
    "older_sister",
    "younger_sister",
    "younger_brother",
    "mother",
    "father",
    "grandfather",
    "grandmother",
}
_SOCIAL_PRONOUNS = {
    "anh",
    "chị",
    "em",
    "con",
    "cháu",
    "mẹ",
    "bố",
    "ba",
    "ông",
    "bà",
    "cô",
    "chú",
    "bác",
    "thầy",
    "sếp",
}
_FIRST_PERSON_SAFE = {"tôi", "ta", "mình"}
_THIRD_PERSON_MARKERS = ("anh ấy", "chị ấy", "cô ấy", "ông ấy", "bà ấy", "người đó", "người này")


def source_has_contextual_terms(text: str) -> bool:
    normalized = _normalize_source(text)
    return bool(_TERM_PATTERN and re.search(_TERM_PATTERN, normalized))


def analyze_source_semantics(
    *,
    segment_id: int | str,
    source: str,
    speaker_id: str | None = None,
    listener_id: str | None = None,
    pronoun_plan: PronounPlan | None = None,
) -> SourceSemanticRepresentation:
    normalized = _normalize_source(source)
    terms = _extract_terms(normalized)
    listener = _resolved_listener(listener_id, pronoun_plan)
    entities: dict[str, Any] = {
        "speaker": speaker_id or (pronoun_plan.speaker if pronoun_plan else None),
        "listener": listener,
        "referents": [],
    }
    propositions: list[dict[str, Any]] = []
    unresolved: list[str] = []
    discourse_role = _discourse_role(normalized, terms)

    if _SECOND_PERSON_RE.search(normalized) and listener is None:
        unresolved.append("listener_unknown")
    if terms and listener is None and discourse_role == "direct_address":
        unresolved.append("direct_address_listener_unresolved")
    if _FIRST_PERSON_RE.search(normalized) and not _has_enforced_plan(pronoun_plan):
        unresolved.append("speaker_listener_pronoun_pair_unresolved")
    if _THIRD_PERSON_RE.search(normalized):
        entities["referents"].append({"role": "third_person", "id": None})
        unresolved.append("third_person_referent_unresolved")

    for term in terms:
        if discourse_role == "direct_address" and term.social_address_possible:
            propositions.append(
                {
                    "type": "address",
                    "usage": "direct_address",
                    "category": term.category,
                    "semanticRole": term.canonical,
                    "entity": listener,
                    "source": term.source,
                }
            )
            if term.category == "SOCIAL_KINSHIP_ADDRESS":
                unresolved.append("kinship_vs_social_address_unresolved")
        elif _is_possessive_relationship(normalized, term):
            propositions.append(
                {
                    "type": "relationship_assertion",
                    "subject": "third_person" if _THIRD_PERSON_RE.search(normalized) else None,
                    "relationship": term.canonical,
                    "relativeTo": "speaker",
                    "category": term.category,
                    "source": term.source,
                }
            )
        else:
            propositions.append(
                {
                    "type": "reference",
                    "discourseRole": discourse_role,
                    "category": term.category,
                    "semanticRole": term.canonical,
                    "source": term.source,
                }
            )
            if term.social_address_possible and term.category in {"KINSHIP", "SOCIAL_KINSHIP_ADDRESS"}:
                unresolved.append("relationship_sense_unresolved")

    ambiguity_score = _ambiguity_score(normalized, terms, unresolved, pronoun_plan)
    guidance = _realization_guidance(
        discourse_role=discourse_role,
        terms=terms,
        pronoun_plan=pronoun_plan,
        unresolved=unresolved,
    )
    return SourceSemanticRepresentation(
        segment_id=segment_id,
        source_text=normalized,
        speaker_id=speaker_id or (pronoun_plan.speaker if pronoun_plan else None),
        listener_id=listener,
        entities=entities,
        propositions=propositions,
        terms=terms,
        discourse_role=discourse_role,
        unresolved=list(dict.fromkeys(unresolved)),
        ambiguity_score=ambiguity_score,
        realization_guidance=guidance,
    )


def compact_semantic_payload(representation: SourceSemanticRepresentation) -> dict[str, Any]:
    payload = representation.to_dict()
    if len(payload["terms"]) > 4:
        payload["terms"] = payload["terms"][:4]
    if len(payload["propositions"]) > 4:
        payload["propositions"] = payload["propositions"][:4]
    return {key: value for key, value in payload.items() if value not in (None, [], {}, "")}


def requires_deeper_reasoning(representation: SourceSemanticRepresentation) -> bool:
    return representation.ambiguity_score >= AMBIGUITY_DEEP_REASONING_THRESHOLD


def realization_critic_issues(
    *,
    source: str,
    translation: str,
    representation: SourceSemanticRepresentation,
    pronoun_plan: PronounPlan | None = None,
) -> list[str]:
    normalized_source = _normalize_source(source)
    normalized_vi = _normalize_vi(translation)
    issues: list[str] = []

    if not translation.strip():
        return ["SEMANTIC_OMISSION"]

    enforced = _has_enforced_plan(pronoun_plan)
    has_terms = bool(representation.terms)
    has_first_person = bool(_FIRST_PERSON_RE.search(normalized_source))
    has_second_person = bool(_SECOND_PERSON_RE.search(normalized_source))

    if has_first_person and not enforced and _unsupported_self_reference(normalized_vi):
        issues.append("UNSUPPORTED_PRONOUN_INFERENCE")

    if not has_terms and not enforced and _hallucinated_relationship(normalized_vi):
        issues.append("RELATIONSHIP_HALLUCINATION")

    for proposition in representation.propositions:
        ptype = proposition.get("type")
        role = str(proposition.get("semanticRole") or proposition.get("relationship") or "")
        if ptype == "relationship_assertion":
            if not _relationship_rendered(role, normalized_vi):
                issues.append("RELATIONSHIP_INFORMATION_LOSS")
            if _is_title_role(role) and _family_role_rendered(normalized_vi):
                issues.append("TITLE_ROLE_ERROR")
            if not _possession_rendered(normalized_vi, enforced=enforced):
                issues.append("POSSESSION_ERROR")
            if not enforced and _unsupported_possessive_social_pronoun(normalized_vi):
                issues.append("UNSUPPORTED_PRONOUN_INFERENCE")
                issues.append("RELATIONSHIP_REALIZATION_ERROR")
        if ptype == "address" and _starts_with_third_person_reference(normalized_vi):
            issues.append("DIRECT_ADDRESS_ERROR")
        if ptype == "reference" and role:
            if _is_title_role(role) and _family_role_rendered(normalized_vi):
                issues.append("TITLE_ROLE_ERROR")

    if representation.discourse_role == "direct_address":
        if _starts_with_third_person_reference(normalized_vi):
            issues.append("DIRECT_ADDRESS_ERROR")

    if enforced and pronoun_plan is not None:
        issues.extend(_enforced_plan_issues(normalized_vi, pronoun_plan))

    return list(dict.fromkeys(issues))


def semantic_categories_for_terms(text: str) -> list[dict[str, Any]]:
    return [term.to_dict() for term in _extract_terms(_normalize_source(text))]


def _normalize_source(text: str) -> str:
    return _WHITESPACE_RE.sub("", text or "").strip()


def _normalize_vi(text: str) -> str:
    lowered = (text or "").casefold()
    lowered = re.sub(r"[^\wÀ-ỹ\s]", " ", lowered, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def _extract_terms(text: str) -> list[SemanticTerm]:
    found: list[SemanticTerm] = []
    occupied: list[tuple[int, int]] = []
    for spec in _TERMS_BY_LENGTH:
        for match in re.finditer(re.escape(spec.source), text):
            span = match.span()
            if any(not (span[1] <= start or span[0] >= end) for start, end in occupied):
                continue
            found.append(spec)
            occupied.append(span)
    found.sort(key=lambda term: text.find(term.source))
    return found


def _resolved_listener(listener_id: str | None, pronoun_plan: PronounPlan | None) -> str | None:
    if listener_id:
        return listener_id
    if pronoun_plan is not None and pronoun_plan.listener:
        return pronoun_plan.listener
    return None


def _has_enforced_plan(plan: PronounPlan | None) -> bool:
    return plan is not None and plan.confidence >= PRONOUN_PLAN_ENFORCE_THRESHOLD


def _discourse_role(text: str, terms: list[SemanticTerm]) -> str:
    for term in terms:
        if text.startswith(term.source):
            rest = _VOCATIVE_TRAIL_RE.sub("", text[len(term.source):])
            if rest.startswith(("你", "您", "听", "聽", "过来", "過來", "等等", "别", "不要", "请", "說", "说")):
                return "direct_address"
            if text[len(term.source): len(term.source) + 1] in {"，", ",", "、", "：", ":"}:
                return "direct_address"
    if any(_is_possessive_relationship(text, term) for term in terms):
        return "possessive_relationship"
    if _THIRD_PERSON_RE.search(text) and terms:
        return "third_person_reference"
    if terms:
        return "subject_or_object_reference"
    return "utterance"


def _is_possessive_relationship(text: str, term: SemanticTerm) -> bool:
    pattern = _POSSESSIVE_RE_TEMPLATE.format(term=re.escape(term.source))
    assertion = _ASSERTION_RE_TEMPLATE.format(term=re.escape(term.source))
    return bool(re.search(pattern, text) or re.search(assertion, text))


def _ambiguity_score(
    text: str,
    terms: list[SemanticTerm],
    unresolved: list[str],
    pronoun_plan: PronounPlan | None,
) -> float:
    score = 0.0
    score += min(0.48, 0.12 * len(set(unresolved)))
    if terms:
        score += 0.18
    if any(term.social_address_possible for term in terms):
        score += 0.14
    if _FIRST_PERSON_RE.search(text) or _SECOND_PERSON_RE.search(text):
        score += 0.10
    if pronoun_plan is None:
        score += 0.12
    elif pronoun_plan.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD:
        score += 0.10
    return round(min(1.0, score), 3)


def _realization_guidance(
    *,
    discourse_role: str,
    terms: list[SemanticTerm],
    pronoun_plan: PronounPlan | None,
    unresolved: list[str],
) -> dict[str, Any]:
    enforced = _has_enforced_plan(pronoun_plan)
    return {
        "layer": "vietnamese_social_realization",
        "preserveSourceMeaningFirst": True,
        "discourseRole": discourse_role,
        "termCategories": list(dict.fromkeys(term.category for term in terms)),
        "relationshipTermsAreFactsNotSurfaceWords": True,
        "preferNeutralSelfReference": not enforced,
        "allowPronounInference": enforced,
        "unknownsMustStayUnknown": bool(unresolved),
        "evidencePriority": [
            "current_source_sentence",
            "explicit_source_context",
            "verified_relationship_memory",
            "scene_context",
            "accepted_translation_style",
        ],
    }


def _relationship_rendered(role: str, normalized_vi: str) -> bool:
    markers = _RELATIONSHIP_VI_MARKERS.get(role, ())
    return bool(markers and any(marker in normalized_vi for marker in markers))


def _possession_rendered(normalized_vi: str, *, enforced: bool) -> bool:
    if "của tôi" in normalized_vi or "của ta" in normalized_vi or "của mình" in normalized_vi:
        return True
    if enforced and any(f"của {word}" in normalized_vi for word in _SOCIAL_PRONOUNS):
        return True
    return bool(re.search(r"\b(tôi|ta|mình)\s+(có|là)\b", normalized_vi))


def _unsupported_self_reference(normalized_vi: str) -> bool:
    if re.search(r"\b(tôi|ta|mình)\b", normalized_vi):
        return False
    if re.search(r"\b(anh ấy|chị ấy|cô ấy|ông ấy|bà ấy)\b", normalized_vi):
        stripped = re.sub(r"\b(anh ấy|chị ấy|cô ấy|ông ấy|bà ấy)\b", " ", normalized_vi)
    else:
        stripped = normalized_vi
    return bool(re.search(r"\b(anh|chị|em|con|cháu|mẹ|bố|ba|ông|bà)\b", stripped))


def _unsupported_possessive_social_pronoun(normalized_vi: str) -> bool:
    return bool(re.search(r"\bcủa\s+(em|con|cháu|anh|chị)\b", normalized_vi))


def _hallucinated_relationship(normalized_vi: str) -> bool:
    return bool(
        re.search(
            r"\b(anh trai|chị gái|em gái|em trai|mẹ|bố|ba|cha|con|cháu|ông|bà|sếp|thầy|cô giáo|bác sĩ)\b",
            normalized_vi,
        )
    )


def _starts_with_third_person_reference(normalized_vi: str) -> bool:
    return normalized_vi.startswith(_THIRD_PERSON_MARKERS)


def _is_title_role(role: str) -> bool:
    return role in _TITLE_CANONICALS


def _family_role_rendered(normalized_vi: str) -> bool:
    stripped = re.sub(r"\b(anh ấy|chị ấy|cô ấy|ông ấy|bà ấy)\b", " ", normalized_vi)
    return any(
        re.search(rf"\b{re.escape(marker)}\b", stripped)
        for role in _FAMILY_CANONICALS
        for marker in _RELATIONSHIP_VI_MARKERS.get(role, ())
    )


def _enforced_plan_issues(normalized_vi: str, plan: PronounPlan) -> list[str]:
    expected = {
        item.casefold()
        for item in (plan.self_pronoun, plan.target_pronoun)
        if item and "/" not in item
    }
    if not expected:
        return []
    words = set(normalized_vi.split())
    wrong = words.intersection(_SOCIAL_PRONOUNS - expected - _FIRST_PERSON_SAFE)
    return ["SPEAKER_LISTENER_ERROR"] if wrong else []
