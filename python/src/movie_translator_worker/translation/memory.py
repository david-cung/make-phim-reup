"""Job-scoped translation memory for one movie translation run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import TranslatedSegment, TranslationResult, ensure_translation_result

_CJK_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")
_CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_CJK_TITLED_NAME_RE = re.compile(
    r"(?:[\u4e00-\u9fff]{1,2}(?:哥|姐|总|叔|姨|妈|爸)|[\u4e00-\u9fff]{1,3}(?:先生|小姐|太太|老师|医生|警官|老板))"
)
_ADDRESS_TERMS = (
    "哥哥",
    "姐姐",
    "老师",
    "医生",
    "警官",
    "老板",
    "先生",
    "小姐",
    "太太",
    "哥",
    "姐",
    "妈",
    "爸",
)
_COMMON_CJK_TOKENS = {
    "你们",
    "我们",
    "他们",
    "她们",
    "这个",
    "那个",
    "什么",
    "怎么",
    "为什么",
    "现在",
    "今天",
    "明天",
    "昨天",
    "这里",
    "那里",
    "可以",
    "没有",
    "不是",
    "就是",
    "知道",
    "起来",
    "回去",
    "过来",
    "因为",
    "所以",
    "如果",
    "但是",
    "然后",
    "已经",
    "不要",
    "不能",
    "告诉",
    "觉得",
    "自己",
    "快走",
    "别走",
    "听见",
    "外面",
    "马上",
    "等等",
}
_NON_NAME_HINT_CHARS = set("我你他她它的是了吗呢啊吧不很在有就都和来去走听见说帮会等快别马外面")
PRONOUN_PLAN_ENFORCE_THRESHOLD = 0.80

_RELATIONSHIP_HINTS: tuple[
    tuple[
        tuple[str, ...],
        str,
        str | None,
        str | None,
        str,
        str | None,
        str | None,
    ],
    ...,
] = (
    (
        ("哥哥", "哥"),
        "younger_to_older_brother",
        "em",
        "anh",
        "older_brother_to_younger",
        "anh",
        "em",
    ),
    (
        ("姐姐", "姐"),
        "younger_to_older_sister",
        "em",
        "chị",
        "older_sister_to_younger",
        "chị",
        "em",
    ),
    (
        ("妈妈", "妈", "母亲"),
        "child_to_mother",
        "con",
        "mẹ",
        "mother_to_child",
        "mẹ",
        "con",
    ),
    (
        ("爸爸", "爸", "父亲"),
        "child_to_father",
        "con",
        "bố",
        "father_to_child",
        "bố",
        "con",
    ),
    (
        ("爷爷", "外公"),
        "grandchild_to_grandfather",
        "cháu",
        "ông",
        "grandfather_to_grandchild",
        "ông",
        "cháu",
    ),
    (
        ("奶奶", "外婆"),
        "grandchild_to_grandmother",
        "cháu",
        "bà",
        "grandmother_to_grandchild",
        "bà",
        "cháu",
    ),
    (
        ("老板", "总"),
        "employee_to_boss",
        "tôi",
        "sếp",
        "boss_to_employee",
        "tôi",
        "cậu",
    ),
    (
        ("老师", "先生"),
        "student_to_teacher",
        "em",
        "thầy/cô",
        "teacher_to_student",
        "thầy/cô",
        "em",
    ),
    (
        ("医生",),
        "patient_to_doctor",
        "tôi",
        "bác sĩ",
        "doctor_to_patient",
        "tôi",
        "anh/chị",
    ),
    (
        ("男朋友", "女朋友", "老公", "老婆"),
        "romantic_partner",
        "tôi",
        None,
        "romantic_partner",
        "tôi",
        None,
    ),
)


@dataclass
class CharacterMemory:
    id: str
    speaker_ids: list[str] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    target_name: str = ""
    gender: str = "unknown"
    age: int | None = None
    age_group: str | None = None
    occupation: str | None = None
    roles: list[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker_ids": list(self.speaker_ids),
            "source_names": list(self.source_names),
            "aliases": list(self.aliases),
            "target_name": self.target_name,
            "gender": self.gender,
            "age": self.age,
            "age_group": self.age_group,
            "occupation": self.occupation,
            "roles": list(self.roles),
            "confidence": round(float(self.confidence), 3),
            "confidence_details": {
                key: round(float(value), 3)
                for key, value in self.confidence_details.items()
            },
        }


@dataclass
class SceneMemory:
    scene_id: str
    summary: str
    participants: list[str]
    segments: list[int]
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "summary": self.summary,
            "participants": list(self.participants),
            "segments": [f"segment_{sid}" for sid in self.segments],
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
        }


@dataclass
class RelationshipMemory:
    from_character: str
    to_character: str
    relationship: str = "unknown"
    confidence: float = 0.0
    addressing: dict[str, str | None] = field(default_factory=dict)
    scene_id: str | None = None
    evidence_segments: list[int] = field(default_factory=list)
    relation_domain: str = "unknown"
    evidence_source: str = "source_dialogue"
    evidence_kind: str = "inferred"
    semantic_category: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "from_character": self.from_character,
            "to_character": self.to_character,
            "relationship": self.relationship,
            "relationshipFact": {
                "from_entity": self.from_character,
                "to_entity": self.to_character,
                "relation": self.relationship,
                "relation_domain": self.relation_domain,
                "semantic_category": self.semantic_category,
                "confidence": round(float(self.confidence), 3),
                "evidence_source": self.evidence_source,
                "evidence_kind": self.evidence_kind,
            },
            "confidence": round(float(self.confidence), 3),
            "addressing": {
                "from_self": self.addressing.get("from_self"),
                "from_target": self.addressing.get("from_target"),
            },
            "surfaceRealizationSuggestion": {
                "from_self": self.addressing.get("from_self"),
                "from_target": self.addressing.get("from_target"),
                "derived_from": "relationship_fact",
            },
            "evidence_source": self.evidence_source,
            "evidence_kind": self.evidence_kind,
            "evidence_segments": [f"segment_{sid}" for sid in self.evidence_segments[:8]],
        }
        if self.scene_id:
            payload["scene_id"] = self.scene_id
        return payload


@dataclass
class PronounPlan:
    segment_id: int
    speaker: str | None
    listener: str | None
    relationship: str = "unknown"
    self_pronoun: str | None = None
    target_pronoun: str | None = None
    confidence: float = 0.0
    scene_id: str | None = None
    source: str = "automatic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": f"segment_{self.segment_id}",
            "speaker": self.speaker,
            "listener": self.listener,
            "relationship": self.relationship,
            "self_pronoun": self.self_pronoun,
            "target_pronoun": self.target_pronoun,
            "confidence": round(float(self.confidence), 3),
            "scene_id": self.scene_id,
            "source": self.source,
            "enforce": self.confidence >= PRONOUN_PLAN_ENFORCE_THRESHOLD,
            "context_request": self.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD,
        }


@dataclass
class GraphEvidence:
    segment_id: int
    source_text: str = ""
    evidence_type: str = "source_context"
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": f"segment_{self.segment_id}",
            "source_text": self.source_text,
            "evidence_type": self.evidence_type,
            "weight": round(float(self.weight), 3),
        }


@dataclass
class CharacterGraphNode:
    character_id: str
    names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    gender: str | None = None
    age: int | None = None
    age_group: str | None = None
    occupation: str | None = None
    roles: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "names": list(self.names),
            "aliases": list(self.aliases),
            "speaker_ids": list(self.speaker_ids),
            "gender": self.gender,
            "age": self.age,
            "age_group": self.age_group,
            "occupation": self.occupation,
            "roles": list(self.roles),
            "confidence": {
                key: round(float(value), 3)
                for key, value in self.confidence.items()
            },
        }


@dataclass
class RelationshipFact:
    from_character: str
    to_character: str
    relationship_type: str
    domain: str = "unknown"
    confidence: float = 0.0
    status: str = "inferred"
    evidence: list[GraphEvidence] = field(default_factory=list)
    scene_scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "from": self.from_character,
            "to": self.to_character,
            "relationship": {
                "domain": self.domain,
                "type": self.relationship_type,
            },
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
            "evidence": [item.to_dict() for item in self.evidence[:8]],
        }
        if self.scene_scope:
            payload["scene_scope"] = self.scene_scope
        return payload


@dataclass
class VietnameseAddressPattern:
    speaker_id: str
    listener_id: str
    semantic_relationship_type: str = "unknown"
    speaker_self_form: str | None = None
    listener_form: str | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    scene_scope: str | None = None
    source: str = "relationship_fact"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "speaker_id": self.speaker_id,
            "listener_id": self.listener_id,
            "semantic_relationship": {
                "type": self.semantic_relationship_type,
                "confidence": round(float(self.confidence), 3),
            },
            "preferred_realization": {
                "self": self.speaker_self_form,
                "other": self.listener_form,
            },
            "confidence": round(float(self.confidence), 3),
            "evidence": list(self.evidence),
            "source": self.source,
        }
        if self.scene_scope:
            payload["scene_scope"] = self.scene_scope
        return payload


@dataclass
class RelationshipContradiction:
    from_character: str
    to_character: str
    existing_relation: str
    new_relation: str
    existing_confidence: float
    new_confidence: float
    scene_scope: str | None = None
    conflict: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "from": self.from_character,
            "to": self.to_character,
            "existing_relation": self.existing_relation,
            "new_relation": self.new_relation,
            "existing_confidence": round(float(self.existing_confidence), 3),
            "new_confidence": round(float(self.new_confidence), 3),
            "conflict": self.conflict,
        }
        if self.scene_scope:
            payload["scene_scope"] = self.scene_scope
        return payload


@dataclass
class CharacterGraph:
    characters: dict[str, CharacterGraphNode] = field(default_factory=dict)
    relationship_facts: list[RelationshipFact] = field(default_factory=list)
    address_patterns: list[VietnameseAddressPattern] = field(default_factory=list)
    contradictions: list[RelationshipContradiction] = field(default_factory=list)
    relationship_timeline: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recent_address_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": [node.to_dict() for node in self.characters.values()],
            "relationshipFacts": [fact.to_dict() for fact in self.relationship_facts],
            "addressPatterns": [pattern.to_dict() for pattern in self.address_patterns],
            "contradictions": [item.to_dict() for item in self.contradictions],
            "relationshipTimeline": dict(self.relationship_timeline),
            "recentAddressHistory": dict(self.recent_address_history),
        }

    def relevant_payload(
        self,
        *,
        character_ids: set[str],
        scene_ids: set[str],
    ) -> dict[str, Any]:
        facts = [
            fact
            for fact in self.relationship_facts
            if fact.from_character in character_ids
            and fact.to_character in character_ids
            and (fact.scene_scope is None or fact.scene_scope in scene_ids)
        ][:16]
        patterns = [
            pattern
            for pattern in self.address_patterns
            if pattern.speaker_id in character_ids
            and pattern.listener_id in character_ids
            and (pattern.scene_scope is None or pattern.scene_scope in scene_ids)
        ][:16]
        contradictions = [
            item
            for item in self.contradictions
            if item.from_character in character_ids
            and item.to_character in character_ids
            and (item.scene_scope is None or item.scene_scope in scene_ids)
        ][:8]
        pair_keys = {
            _pair_key(pattern.speaker_id, pattern.listener_id)
            for pattern in patterns
        }
        history = {
            key: rows[-4:]
            for key, rows in self.recent_address_history.items()
            if key in pair_keys
        }
        return {
            "characters": [
                self.characters[cid].to_dict()
                for cid in character_ids
                if cid in self.characters
            ][:12],
            "relationshipFacts": [fact.to_dict() for fact in facts],
            "addressPatterns": [pattern.to_dict() for pattern in patterns],
            "contradictions": [item.to_dict() for item in contradictions],
            "recentAddressHistory": history,
        }

    def address_pattern_for(
        self,
        speaker: str | None,
        listener: str | None,
        scene_id: str | None,
    ) -> VietnameseAddressPattern | None:
        if speaker is None or listener is None:
            return None
        scoped = [
            pattern
            for pattern in self.address_patterns
            if pattern.speaker_id == speaker
            and pattern.listener_id == listener
            and pattern.scene_scope == scene_id
        ]
        if scoped:
            return max(scoped, key=lambda item: item.confidence)
        global_patterns = [
            pattern
            for pattern in self.address_patterns
            if pattern.speaker_id == speaker
            and pattern.listener_id == listener
            and pattern.scene_scope is None
        ]
        if global_patterns:
            return max(global_patterns, key=lambda item: item.confidence)
        return None

    def record_translation_address_pattern(
        self,
        *,
        segment_id: int,
        speaker: str | None,
        listener: str | None,
        pair: tuple[str, str] | None,
    ) -> None:
        if speaker is None or listener is None or pair is None:
            return
        key = _pair_key(speaker, listener)
        rows = self.recent_address_history.setdefault(key, [])
        rows.append(
            {
                "segment_id": f"segment_{segment_id}",
                "self": pair[0],
                "other": pair[1],
                "confidence": 0.45,
                "source": "accepted_translation_style",
            }
        )
        self.recent_address_history[key] = rows[-12:]


@dataclass
class MovieMemory:
    movie_summary: str = ""
    characters: list[CharacterMemory] = field(default_factory=list)
    speaker_character_mapping: dict[str, str] = field(default_factory=dict)
    known_names: dict[str, dict[str, Any]] = field(default_factory=dict)
    translation_memory: dict[str, Any] = field(default_factory=dict)
    scenes: list[SceneMemory] = field(default_factory=list)
    relationships: list[RelationshipMemory] = field(default_factory=list)
    scene_relationship_overrides: dict[str, list[RelationshipMemory]] = field(
        default_factory=dict
    )
    pronoun_plans: dict[int, PronounPlan] = field(default_factory=dict)
    character_graph: CharacterGraph | None = None

    def relevant_payload(
        self,
        *,
        nearby_ids: list[int],
        translation_memory: dict[str, Any],
    ) -> dict[str, Any]:
        nearby = set(nearby_ids)
        scenes = [scene for scene in self.scenes if nearby.intersection(scene.segments)]
        scene_participants = {
            participant for scene in scenes for participant in scene.participants
        }
        current_speakers = {
            speaker
            for speaker, character_id in self.speaker_character_mapping.items()
            if character_id in scene_participants
        }
        character_ids = set(scene_participants)
        for speaker in current_speakers:
            mapped = self.speaker_character_mapping.get(speaker)
            if mapped:
                character_ids.add(mapped)

        characters = [
            character.to_dict()
            for character in self.characters
            if character.id in character_ids
        ]
        relationship_rows = [
            relationship.to_dict()
            for relationship in self.relationships
            if relationship.from_character in character_ids
            and relationship.to_character in character_ids
        ][:16]
        scene_relationship_rows = [
            relationship.to_dict()
            for scene in scenes
            for relationship in self.scene_relationship_overrides.get(scene.scene_id, [])
        ][:16]
        pronoun_plans = {
            f"segment_{sid}": plan.to_dict()
            for sid, plan in self.pronoun_plans.items()
            if sid in nearby
        }
        relevant_names = _relevant_known_names(self.known_names, characters)
        scene_ids = {scene.scene_id for scene in scenes}
        graph_payload = (
            self.character_graph.relevant_payload(
                character_ids=character_ids,
                scene_ids=scene_ids,
            )
            if self.character_graph is not None
            else None
        )
        return {
            "movieSummary": self.movie_summary,
            "currentScenes": [scene.to_dict() for scene in scenes[:3]],
            "characters": characters[:12],
            "relationships": relationship_rows,
            "sceneRelationshipOverrides": scene_relationship_rows,
            "pronounPlans": pronoun_plans,
            "characterGraph": graph_payload,
            "speakerCharacterMapping": {
                speaker: character_id
                for speaker, character_id in self.speaker_character_mapping.items()
                if character_id in character_ids
            },
            "knownNames": relevant_names,
            "translationMemory": translation_memory,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "movie_summary": self.movie_summary,
            "characters": [character.to_dict() for character in self.characters],
            "speaker_character_mapping": dict(self.speaker_character_mapping),
            "known_names": dict(self.known_names),
            "translation_memory": dict(self.translation_memory),
            "scenes": [scene.to_dict() for scene in self.scenes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "scene_relationship_overrides": {
                scene_id: [relationship.to_dict() for relationship in relationships]
                for scene_id, relationships in self.scene_relationship_overrides.items()
            },
            "pronoun_plans": {
                f"segment_{sid}": plan.to_dict()
                for sid, plan in self.pronoun_plans.items()
            },
            "character_graph": (
                self.character_graph.to_dict()
                if self.character_graph is not None
                else None
            ),
        }


@dataclass
class TranslationMemory:
    """Small in-process memory isolated to a single movie/job.

    It is seeded from existing persisted translations, so resuming a
    partial job rebuilds the useful context without sharing state across
    projects.
    """

    translations: dict[int, tuple[str, str]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    pronoun_patterns: list[dict[str, Any]] = field(default_factory=list)
    movie_memory: MovieMemory | None = None

    @classmethod
    def from_segments(cls, segments: list[TranslatedSegment]) -> "TranslationMemory":
        movie_memory = (
            build_movie_memory(segments) if _has_movie_memory_signal(segments) else None
        )
        memory = cls(movie_memory=movie_memory)
        for seg in segments:
            if seg.translation.strip():
                memory.record(seg.id, seg.source_text, seg.translation)
        return memory

    def record(self, segment_id: int, source: str, result: str | TranslationResult) -> None:
        text = ensure_translation_result(result).translation.strip()
        if not text:
            return
        self.translations[segment_id] = (source, text)
        self._record_name_candidates(source, text)
        pair = pronoun_pair(text)
        if pair is not None:
            self.pronoun_patterns.append({"segmentId": segment_id, "pair": pair})
            self.pronoun_patterns = self.pronoun_patterns[-20:]
            if self.movie_memory is not None and self.movie_memory.character_graph is not None:
                plan = self.pronoun_plan_for_segment(segment_id)
                self.movie_memory.character_graph.record_translation_address_pattern(
                    segment_id=segment_id,
                    speaker=plan.speaker if plan is not None else None,
                    listener=plan.listener if plan is not None else None,
                    pair=pair,
                )

    def row_for(self, segment_id: int) -> dict[str, str] | None:
        item = self.translations.get(segment_id)
        if item is None:
            return None
        source, translation = item
        return {"source": source, "translation": translation}

    def prompt_payload(self, nearby_ids: list[int]) -> dict[str, Any]:
        rows = [
            {"id": sid, **row}
            for sid in nearby_ids
            if (row := self.row_for(sid)) is not None
        ]
        translation_memory = {
            "translations": rows[-16:],
            "names": dict(list(self.names.items())[-16:]),
            "pronounPatterns": self.pronoun_patterns[-8:],
        }
        if self.movie_memory is None:
            return translation_memory
        return self.movie_memory.relevant_payload(
            nearby_ids=nearby_ids,
            translation_memory=translation_memory,
        )

    def pronoun_plan_for_segment(self, segment_id: int) -> PronounPlan | None:
        if self.movie_memory is None:
            return None
        return self.movie_memory.pronoun_plans.get(segment_id)

    def address_pattern_for_segment(
        self,
        segment_id: int,
    ) -> VietnameseAddressPattern | None:
        if self.movie_memory is None or self.movie_memory.character_graph is None:
            return None
        plan = self.pronoun_plan_for_segment(segment_id)
        if plan is None:
            return None
        return self.movie_memory.character_graph.address_pattern_for(
            plan.speaker,
            plan.listener,
            plan.scene_id,
        )

    def address_debug_for_segment(self, segment_id: int) -> dict[str, Any]:
        plan = self.pronoun_plan_for_segment(segment_id)
        pattern = self.address_pattern_for_segment(segment_id)
        relationship = None
        if self.movie_memory is not None and plan is not None:
            relationship = _relationship_for_pair(
                self.movie_memory,
                plan.speaker,
                plan.listener,
                plan.scene_id,
            )
        return {
            "speaker": plan.speaker if plan is not None else None,
            "listener": plan.listener if plan is not None else None,
            "relationship": relationship.relationship if relationship is not None else None,
            "relationship_confidence": (
                round(float(relationship.confidence), 3)
                if relationship is not None
                else 0.0
            ),
            "address_pair": (
                [pattern.speaker_self_form, pattern.listener_form]
                if pattern is not None
                else None
            ),
            "address_confidence": (
                round(float(pattern.confidence), 3) if pattern is not None else 0.0
            ),
            "evidence_segments": (
                [
                    f"segment_{sid}"
                    for sid in relationship.evidence_segments[:8]
                ]
                if relationship is not None
                else []
            ),
            "decision": _address_decision(plan, pattern),
            "warnings": [],
        }

    def address_consistency_issues(
        self,
        *,
        segment_id: int,
        source: str,
        translation: str,
    ) -> list[str]:
        pattern = self.address_pattern_for_segment(segment_id)
        if pattern is None or pattern.confidence < PRONOUN_PLAN_ENFORCE_THRESHOLD:
            return []
        if _source_supports_address_shift(source):
            return []
        words = _vi_social_words(translation)
        if not words:
            return []
        expected = {
            item.casefold()
            for item in (pattern.speaker_self_form, pattern.listener_form)
            if item and "/" not in item
        }
        if not expected:
            return []
        unexpected = [word for word in words if word not in expected]
        issues: list[str] = []
        if unexpected:
            issues.append("UNJUSTIFIED_ADDRESS_SHIFT")
            issues.append("ADDRESS_PAIR_INCONSISTENCY")
        expected_pair = (
            pattern.speaker_self_form.casefold()
            if pattern.speaker_self_form
            else None,
            pattern.listener_form.casefold()
            if pattern.listener_form
            else None,
        )
        proposed = pronoun_pair(translation)
        if proposed and expected_pair[0] and expected_pair[1]:
            if proposed == (expected_pair[1], expected_pair[0]):
                issues.append("WRONG_RELATIONSHIP_DIRECTION")
                issues.append("SPEAKER_LISTENER_MISMATCH")
        return list(dict.fromkeys(issues))

    def _record_name_candidates(self, source: str, translation: str) -> None:
        for name in _CJK_NAME_RE.findall(source):
            if name not in self.names:
                # Keep this conservative. The LLM receives the source
                # name and nearest translated line, but we do not invent
                # a global dictionary from weak evidence.
                self.names[name] = translation[:80]


def extract_name_mentions(text: str) -> list[str]:
    """Return conservative person-name/address candidates from CJK dialogue.

    This is intentionally deterministic and offline. It favours explicit
    titled names and repeated forms of address; weak 2-3 character
    candidates are filtered against common non-name dialogue words.
    """
    mentions: list[str] = []

    def add(candidate: str) -> None:
        value = candidate.strip("，。！？、,.!?;:：；“”\"'（）()[] ")
        if not value:
            return
        if value in _COMMON_CJK_TOKENS:
            return
        if value not in mentions:
            mentions.append(value)

    for match in _CJK_TITLED_NAME_RE.findall(text):
        add(match)
    for term in _ADDRESS_TERMS:
        if term in text:
            add(term)
    for sequence in _CJK_SEQUENCE_RE.findall(text):
        candidate = sequence[:2]
        if candidate in _COMMON_CJK_TOKENS:
            continue
        if any(char in _NON_NAME_HINT_CHARS for char in candidate):
            continue
        add(candidate)
    return mentions


def build_movie_memory(segments: list[TranslatedSegment]) -> MovieMemory:
    ordered = sorted(segments, key=lambda seg: (seg.start, seg.end, seg.id))
    known_names = _collect_known_names(ordered)
    speaker_segments = _collect_speaker_segments(ordered)
    characters: list[CharacterMemory] = []
    mapping: dict[str, str] = {}

    for speaker_id, speaker_lines in speaker_segments.items():
        names = _names_near_speaker(ordered, speaker_id)
        confidence = _character_confidence(speaker_lines, names)
        if len(speaker_lines) < 2 and not names:
            continue
        character = CharacterMemory(
            id=f"character_{len(characters) + 1:03d}",
            speaker_ids=[speaker_id],
            source_names=names[:8],
            target_name=names[0] if names else "",
            gender="unknown",
            confidence=confidence,
        )
        characters.append(character)
        if confidence >= 0.38:
            mapping[speaker_id] = character.id

    memory = MovieMemory(
        movie_summary=_summarize_movie(ordered, known_names),
        characters=characters,
        speaker_character_mapping=mapping,
        known_names=known_names,
        translation_memory={},
    )
    memory.scenes = build_scene_memory(ordered, memory)
    _infer_relationship_memory(ordered, memory)
    memory.character_graph = build_character_graph(ordered, memory)
    return memory


def build_scene_memory(
    segments: list[TranslatedSegment],
    movie_memory: MovieMemory,
    *,
    max_gap_seconds: float = 4.0,
    max_segments_per_scene: int = 25,
) -> list[SceneMemory]:
    if not segments:
        return []
    scenes: list[list[TranslatedSegment]] = []
    current: list[TranslatedSegment] = []
    recent_speakers: list[str] = []

    for seg in sorted(segments, key=lambda item: (item.start, item.end, item.id)):
        gap = seg.start - current[-1].end if current else 0.0
        speaker = _stable_speaker(seg.speaker_id)
        if speaker and (not recent_speakers or recent_speakers[-1] != speaker):
            recent_speakers.append(speaker)
            recent_speakers = recent_speakers[-6:]
        should_split = bool(current) and (
            gap > max_gap_seconds
            or len(current) >= max_segments_per_scene
            or len(set(recent_speakers)) >= 4
        )
        if should_split:
            scenes.append(current)
            current = []
            recent_speakers = [speaker] if speaker else []
        current.append(seg)
    if current:
        scenes.append(current)

    return [
        _scene_from_segments(index + 1, scene, movie_memory)
        for index, scene in enumerate(scenes)
    ]


def build_character_graph(
    segments: list[TranslatedSegment],
    movie_memory: MovieMemory,
) -> CharacterGraph:
    source_by_id = {seg.id: seg.source_text for seg in segments}
    graph = CharacterGraph()
    for character in movie_memory.characters:
        aliases = [
            name
            for name in character.source_names
            if name and name != character.target_name
        ]
        graph.characters[character.id] = CharacterGraphNode(
            character_id=character.id,
            names=list(character.source_names),
            aliases=list(dict.fromkeys(character.aliases + aliases)),
            speaker_ids=list(character.speaker_ids),
            gender=None if character.gender == "unknown" else character.gender,
            age=character.age,
            age_group=character.age_group,
            occupation=character.occupation,
            roles=list(character.roles),
            confidence={
                "identity": character.confidence,
                **character.confidence_details,
            },
        )

    relationship_rows: list[RelationshipMemory] = []
    relationship_rows.extend(movie_memory.relationships)
    for rows in movie_memory.scene_relationship_overrides.values():
        relationship_rows.extend(rows)

    for relationship in relationship_rows:
        graph.relationship_facts.append(
            _relationship_fact_from_memory(relationship, source_by_id)
        )
        pattern = _address_pattern_from_relationship(relationship)
        if pattern is not None:
            graph.address_patterns.append(pattern)
        key = _pair_key(relationship.from_character, relationship.to_character)
        timeline_rows = graph.relationship_timeline.setdefault(key, [])
        timeline_rows.append(
            {
                "scene": relationship.scene_id,
                "relationship": relationship.relationship,
                "confidence": round(float(relationship.confidence), 3),
                "evidence_segments": [
                    f"segment_{sid}" for sid in relationship.evidence_segments[:8]
                ],
            }
        )

    graph.contradictions = _relationship_contradictions(graph.relationship_facts)
    return graph


def _relationship_fact_from_memory(
    relationship: RelationshipMemory,
    source_by_id: dict[int, str],
) -> RelationshipFact:
    evidence_type = (
        "explicit_source_statement"
        if relationship.evidence_kind == "explicit"
        else "contextual_source_evidence"
    )
    return RelationshipFact(
        from_character=relationship.from_character,
        to_character=relationship.to_character,
        relationship_type=relationship.relationship,
        domain=relationship.relation_domain,
        confidence=relationship.confidence,
        status=_relationship_status(relationship),
        scene_scope=relationship.scene_id,
        evidence=[
            GraphEvidence(
                segment_id=sid,
                source_text=source_by_id.get(sid, ""),
                evidence_type=evidence_type,
                weight=1.0 if relationship.evidence_kind == "explicit" else 0.72,
            )
            for sid in relationship.evidence_segments
        ],
    )


def _address_pattern_from_relationship(
    relationship: RelationshipMemory,
) -> VietnameseAddressPattern | None:
    self_ref = relationship.addressing.get("from_self")
    target_ref = relationship.addressing.get("from_target")
    if not self_ref and not target_ref:
        return None
    return VietnameseAddressPattern(
        speaker_id=relationship.from_character,
        listener_id=relationship.to_character,
        semantic_relationship_type=relationship.relationship,
        speaker_self_form=self_ref,
        listener_form=target_ref,
        confidence=min(0.96, max(0.0, relationship.confidence)),
        evidence=[
            "explicit_relationship"
            if relationship.evidence_kind == "explicit"
            else "source_context",
            "repeated_direct_address"
            if len(relationship.evidence_segments) > 1
            else "single_direct_address",
        ],
        scene_scope=relationship.scene_id,
        source="relationship_fact",
    )


def _relationship_status(relationship: RelationshipMemory) -> str:
    if relationship.confidence >= 0.90 and relationship.evidence_kind == "explicit":
        return "verified"
    if relationship.confidence >= 0.80:
        return "source_supported"
    if relationship.confidence >= 0.55:
        return "inferred"
    return "unresolved"


def _relationship_contradictions(
    facts: list[RelationshipFact],
) -> list[RelationshipContradiction]:
    by_pair: dict[tuple[str, str], list[RelationshipFact]] = {}
    for fact in facts:
        if fact.confidence < 0.50:
            continue
        by_pair.setdefault((fact.from_character, fact.to_character), []).append(fact)
    contradictions: list[RelationshipContradiction] = []
    for (from_char, to_char), rows in by_pair.items():
        rows = sorted(rows, key=lambda item: item.confidence, reverse=True)
        if not rows:
            continue
        strongest = rows[0]
        for other in rows[1:]:
            if other.relationship_type == strongest.relationship_type:
                continue
            contradictions.append(
                RelationshipContradiction(
                    from_character=from_char,
                    to_character=to_char,
                    existing_relation=strongest.relationship_type,
                    new_relation=other.relationship_type,
                    existing_confidence=strongest.confidence,
                    new_confidence=other.confidence,
                    scene_scope=other.scene_scope,
                )
            )
    return contradictions


def _infer_relationship_memory(
    segments: list[TranslatedSegment],
    movie_memory: MovieMemory,
) -> None:
    evidence: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    for scene in movie_memory.scenes:
        scene_segments = [seg for seg in segments if seg.id in scene.segments]
        for seg in scene_segments:
            speaker = _character_for_segment(seg, movie_memory)
            if speaker is None:
                continue
            listener = _listener_for_segment(seg, scene_segments, movie_memory)
            if listener is None or listener == speaker:
                movie_memory.pronoun_plans[seg.id] = PronounPlan(
                    segment_id=seg.id,
                    speaker=speaker,
                    listener=listener,
                    confidence=0.35 if listener is None else 0.45,
                    scene_id=scene.scene_id,
                    source="unknown_listener",
                )
                continue
            hint = _relationship_hint(seg.source_text)
            if hint is None:
                relationship = _relationship_for_pair(
                    movie_memory, speaker, listener, scene.scene_id
                )
                if relationship is None:
                    movie_memory.pronoun_plans[seg.id] = PronounPlan(
                        segment_id=seg.id,
                        speaker=speaker,
                        listener=listener,
                        confidence=0.48,
                        scene_id=scene.scene_id,
                        source="needs_more_context",
                    )
                    continue
            else:
                forward, reciprocal = hint
                _add_relationship_evidence(
                    evidence,
                    speaker,
                    listener,
                    scene.scene_id,
                    seg.id,
                    forward,
                )
                _add_relationship_evidence(
                    evidence,
                    listener,
                    speaker,
                    scene.scene_id,
                    seg.id,
                    reciprocal,
                    reciprocal=True,
                )
                relationship = _relationship_from_hint(
                    speaker,
                    listener,
                    scene.scene_id,
                    [seg.id],
                    forward,
                    confidence=0.72,
                )
            movie_memory.pronoun_plans[seg.id] = _plan_from_relationship(
                segment_id=seg.id,
                scene_id=scene.scene_id,
                relationship=relationship,
                source="scene_relationship",
            )

    scene_relationships: dict[str, list[RelationshipMemory]] = {}
    global_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (from_char, to_char, rel_type, scene_id), item in evidence.items():
        relationship = _relationship_from_hint(
            from_char,
            to_char,
            scene_id,
            item["segments"],
            item["hint"],
            confidence=_relationship_confidence(item["count"], scene_level=True),
        )
        scene_relationships.setdefault(str(scene_id), []).append(relationship)
        key = (from_char, to_char, rel_type)
        global_item = global_candidates.setdefault(
            key,
            {"count": 0, "segments": [], "hint": item["hint"], "scenes": set()},
        )
        global_item["count"] += int(item["count"])
        global_item["segments"].extend(item["segments"])
        global_item["scenes"].add(scene_id)

    movie_memory.scene_relationship_overrides = scene_relationships
    movie_memory.relationships = []
    for (from_char, to_char, _rel_type), item in global_candidates.items():
        confidence = _relationship_confidence(
            int(item["count"]),
            scene_count=len(item["scenes"]),
            scene_level=False,
        )
        if confidence < 0.80:
            continue
        movie_memory.relationships.append(
            _relationship_from_hint(
                from_char,
                to_char,
                None,
                item["segments"],
                item["hint"],
                confidence=confidence,
            )
        )

    for sid, plan in list(movie_memory.pronoun_plans.items()):
        relationship = _relationship_for_pair(
            movie_memory,
            plan.speaker,
            plan.listener,
            plan.scene_id,
        )
        if relationship is not None and relationship.confidence >= plan.confidence:
            movie_memory.pronoun_plans[sid] = _plan_from_relationship(
                segment_id=sid,
                scene_id=plan.scene_id,
                relationship=relationship,
                source="relationship_memory",
            )


def _relationship_hint(
    text: str,
) -> tuple[
    tuple[str, str | None, str | None],
    tuple[str, str | None, str | None],
] | None:
    for terms, relation, self_ref, target_ref, reverse, reverse_self, reverse_target in _RELATIONSHIP_HINTS:
        if any(term in text for term in terms):
            return (relation, self_ref, target_ref), (reverse, reverse_self, reverse_target)
    return None


def _add_relationship_evidence(
    evidence: dict[tuple[str, str, str, str | None], dict[str, Any]],
    from_char: str,
    to_char: str,
    scene_id: str,
    segment_id: int,
    hint: tuple[str, str | None, str | None],
    *,
    reciprocal: bool = False,
) -> None:
    relationship, _self_ref, _target_ref = hint
    key = (from_char, to_char, relationship, scene_id)
    item = evidence.setdefault(
        key,
        {
            "count": 0,
            "segments": [],
            "hint": hint,
            "reciprocal": reciprocal,
        },
    )
    item["count"] += 1
    if segment_id not in item["segments"]:
        item["segments"].append(segment_id)


def _relationship_from_hint(
    from_char: str,
    to_char: str,
    scene_id: str | None,
    segments: list[int],
    hint: tuple[str, str | None, str | None],
    *,
    confidence: float,
) -> RelationshipMemory:
    relationship, self_ref, target_ref = hint
    return RelationshipMemory(
        from_character=from_char,
        to_character=to_char,
        relationship=relationship,
        confidence=confidence,
        addressing={"from_self": self_ref, "from_target": target_ref},
        scene_id=scene_id,
        evidence_segments=list(dict.fromkeys(segments)),
        relation_domain=_relationship_domain(relationship),
        evidence_source="source_dialogue",
        evidence_kind="explicit" if confidence >= 0.80 else "contextual",
        semantic_category=_relationship_semantic_category(relationship),
    )


def _relationship_confidence(
    count: int,
    *,
    scene_level: bool,
    scene_count: int = 1,
) -> float:
    if scene_level:
        return round(min(0.94, 0.66 + count * 0.12), 3)
    return round(min(0.96, 0.68 + count * 0.08 + max(0, scene_count - 1) * 0.08), 3)


def _relationship_domain(relationship: str) -> str:
    if any(token in relationship for token in ("brother", "sister", "mother", "father", "grand")):
        return "family"
    if "romantic" in relationship or relationship in {"husband", "wife"}:
        return "romantic"
    if relationship in {"employee_to_boss", "boss_to_employee"}:
        return "workplace"
    if relationship in {"student_to_teacher", "teacher_to_student"}:
        return "school"
    if relationship in {"patient_to_doctor", "doctor_to_patient"}:
        return "professional"
    return "unknown"


def _relationship_semantic_category(relationship: str) -> str:
    if any(token in relationship for token in ("brother", "sister", "mother", "father", "grand")):
        return "KINSHIP"
    if "romantic" in relationship:
        return "RELATIONSHIP"
    if relationship in {"employee_to_boss", "boss_to_employee"}:
        return "ORGANIZATIONAL_ROLE"
    if relationship in {
        "student_to_teacher",
        "teacher_to_student",
        "patient_to_doctor",
        "doctor_to_patient",
    }:
        return "PROFESSIONAL_TITLE"
    return "unknown"


def _plan_from_relationship(
    *,
    segment_id: int,
    scene_id: str | None,
    relationship: RelationshipMemory,
    source: str,
) -> PronounPlan:
    return PronounPlan(
        segment_id=segment_id,
        speaker=relationship.from_character,
        listener=relationship.to_character,
        relationship=relationship.relationship,
        self_pronoun=relationship.addressing.get("from_self"),
        target_pronoun=relationship.addressing.get("from_target"),
        confidence=relationship.confidence,
        scene_id=scene_id,
        source=source,
    )


def _relationship_for_pair(
    movie_memory: MovieMemory,
    speaker: str | None,
    listener: str | None,
    scene_id: str | None,
) -> RelationshipMemory | None:
    if speaker is None or listener is None:
        return None
    if scene_id:
        for relationship in movie_memory.scene_relationship_overrides.get(scene_id, []):
            if (
                relationship.from_character == speaker
                and relationship.to_character == listener
            ):
                return relationship
    for relationship in movie_memory.relationships:
        if relationship.from_character == speaker and relationship.to_character == listener:
            return relationship
    return None


def _character_for_segment(
    seg: TranslatedSegment,
    movie_memory: MovieMemory,
) -> str | None:
    speaker = _stable_speaker(seg.speaker_id)
    if speaker is None:
        return None
    return movie_memory.speaker_character_mapping.get(speaker)


def _listener_for_segment(
    seg: TranslatedSegment,
    scene_segments: list[TranslatedSegment],
    movie_memory: MovieMemory,
) -> str | None:
    speaker = _character_for_segment(seg, movie_memory)
    if speaker is None:
        return None
    if any(marker in seg.source_text for marker in ("你们", "大家", "诸位")):
        return None
    explicit = _explicit_listener(seg.source_text, speaker, movie_memory)
    if explicit is not None:
        return explicit
    same_scene = [item.id for item in scene_segments]
    try:
        position = same_scene.index(seg.id)
    except ValueError:
        position = 0
    nearby: list[str] = []
    for near in reversed(scene_segments[max(0, position - 3) : position]):
        character = _character_for_segment(near, movie_memory)
        if character and character != speaker:
            nearby.append(character)
    for near in scene_segments[position + 1 : position + 4]:
        character = _character_for_segment(near, movie_memory)
        if character and character != speaker:
            nearby.append(character)
    unique = list(dict.fromkeys(nearby))
    if len(unique) == 1:
        return unique[0]
    return None


def _explicit_listener(
    text: str,
    speaker: str,
    movie_memory: MovieMemory,
) -> str | None:
    matches: list[str] = []
    for character in movie_memory.characters:
        if character.id == speaker:
            continue
        if any(name and name in text for name in character.source_names):
            matches.append(character.id)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    return None


def _collect_known_names(segments: list[TranslatedSegment]) -> dict[str, dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    for seg in segments:
        for name in extract_name_mentions(seg.source_text):
            item = known.setdefault(
                name,
                {
                    "count": 0,
                    "first_seen": round(float(seg.start), 3),
                    "segments": [],
                },
            )
            item["count"] += 1
            if len(item["segments"]) < 8:
                item["segments"].append(f"segment_{seg.id}")
    return dict(
        sorted(
            known.items(),
            key=lambda item: (-int(item[1]["count"]), float(item[1]["first_seen"])),
        )
    )


def _collect_speaker_segments(
    segments: list[TranslatedSegment],
) -> dict[str, list[TranslatedSegment]]:
    speaker_segments: dict[str, list[TranslatedSegment]] = {}
    for seg in segments:
        speaker = _stable_speaker(seg.speaker_id)
        if speaker is not None:
            speaker_segments.setdefault(speaker, []).append(seg)
    return dict(
        sorted(
            speaker_segments.items(),
            key=lambda item: min(seg.start for seg in item[1]),
        )
    )


def _names_near_speaker(
    ordered: list[TranslatedSegment],
    speaker_id: str,
    *,
    window: int = 2,
) -> list[str]:
    names: dict[str, int] = {}
    for index, seg in enumerate(ordered):
        if _stable_speaker(seg.speaker_id) != speaker_id:
            continue
        start = max(0, index - window)
        end = min(len(ordered), index + window + 1)
        for near in ordered[start:end]:
            for name in extract_name_mentions(near.source_text):
                names[name] = names.get(name, 0) + 1
    return [
        name
        for name, _count in sorted(
            names.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _character_confidence(
    speaker_lines: list[TranslatedSegment],
    names: list[str],
) -> float:
    line_score = min(0.40, len(speaker_lines) * 0.08)
    name_score = min(0.35, len(names) * 0.12)
    speaker_conf = [
        float(seg.speaker_confidence)
        for seg in speaker_lines
        if seg.speaker_confidence is not None
    ]
    diarization_score = 0.0
    if speaker_conf:
        diarization_score = min(0.25, max(0.0, sum(speaker_conf) / len(speaker_conf)) * 0.25)
    return round(min(0.95, line_score + name_score + diarization_score), 3)


def _summarize_movie(
    segments: list[TranslatedSegment],
    known_names: dict[str, dict[str, Any]],
) -> str:
    if not segments:
        return ""
    top_names = list(known_names.keys())[:6]
    first = _short_text(segments[0].source_text)
    last = _short_text(segments[-1].source_text)
    parts = [f"{len(segments)} dialogue segments"]
    if top_names:
        parts.append("recurring names/forms of address: " + ", ".join(top_names))
    if first:
        parts.append(f"opens with: {first}")
    if last and last != first:
        parts.append(f"latest context: {last}")
    return "; ".join(parts)


def _scene_from_segments(
    scene_number: int,
    segments: list[TranslatedSegment],
    movie_memory: MovieMemory,
) -> SceneMemory:
    participants: list[str] = []
    for seg in segments:
        speaker = _stable_speaker(seg.speaker_id)
        character_id = (
            movie_memory.speaker_character_mapping.get(speaker)
            if speaker is not None
            else None
        )
        if character_id and character_id not in participants:
            participants.append(character_id)
    texts = [_short_text(seg.source_text, limit=28) for seg in segments if seg.source_text.strip()]
    summary = " / ".join(texts[:3])
    if len(texts) > 3:
        summary += " / ..."
    return SceneMemory(
        scene_id=f"scene_{scene_number:03d}",
        summary=summary,
        participants=participants,
        segments=[seg.id for seg in segments],
        start=segments[0].start,
        end=segments[-1].end,
    )


def _relevant_known_names(
    known_names: dict[str, dict[str, Any]],
    characters: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    wanted = {
        name
        for character in characters
        for name in character.get("source_names", [])
        if isinstance(name, str)
    }
    rows = [
        (name, payload)
        for name, payload in known_names.items()
        if name in wanted
    ]
    if len(rows) < 12:
        for item in known_names.items():
            if item not in rows:
                rows.append(item)
            if len(rows) >= 12:
                break
    return dict(rows[:12])


def _short_text(text: str, *, limit: int = 48) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _stable_speaker(speaker_id: str | None) -> str | None:
    if not speaker_id:
        return None
    speaker = str(speaker_id)
    if speaker.upper() == "UNKNOWN":
        return None
    return speaker


def _has_movie_memory_signal(segments: list[TranslatedSegment]) -> bool:
    for seg in segments:
        if _stable_speaker(seg.speaker_id) is not None:
            return True
        if extract_name_mentions(seg.source_text):
            return True
    return False


_PRONOUNS = ("anh", "chị", "em", "tôi", "mình", "con", "mẹ", "bố", "ba", "cô", "chú", "bác", "ông", "bà", "sếp", "thầy")
_NEUTRAL_PRONOUNS = {"tôi", "mình", "ta"}
_ADDRESS_SHIFT_SOURCE_MARKERS = (
    "滚",
    "滾",
    "分手",
    "别叫",
    "不要叫",
    "不想再",
    "陌生人",
    "先生",
    "小姐",
    "老板",
    "老師",
    "老师",
    "医生",
    "警官",
)


def _pair_key(speaker: str, listener: str) -> str:
    return f"{speaker}->{listener}"


def _address_decision(
    plan: PronounPlan | None,
    pattern: VietnameseAddressPattern | None,
) -> str:
    if plan is None or plan.listener is None:
        return "omit_or_neutral_unknown_listener"
    if pattern is None:
        return "neutral_or_source_only"
    if pattern.confidence >= PRONOUN_PLAN_ENFORCE_THRESHOLD:
        return "use_established_pair_when_needed"
    return "weak_style_hint_only"


def _source_supports_address_shift(source: str) -> bool:
    if any(marker in source for marker in _ADDRESS_SHIFT_SOURCE_MARKERS):
        return True
    if re.search(r"(不|没|別|别).{0,4}(哥|姐|妈|爸|老板|老师|先生|小姐)", source):
        return True
    return False


def _vi_social_words(text: str) -> list[str]:
    words = [w.casefold() for w in re.findall(r"\w+", text, flags=re.UNICODE)]
    out: list[str] = []
    for word in words:
        if word in _NEUTRAL_PRONOUNS:
            continue
        if word in _PRONOUNS and word not in out:
            out.append(word)
    return out


def pronoun_pair(text: str) -> tuple[str, str] | None:
    words = [w.casefold() for w in re.findall(r"\w+", text, flags=re.UNICODE)]
    found = [w for w in words if w in _PRONOUNS]
    if len(found) < 2:
        return None
    return found[0], found[1]


def suspicious_pronoun_shift(
    *,
    segment_id: int,
    translation: str,
    memory: TranslationMemory,
) -> bool:
    current = pronoun_pair(translation)
    if current is None:
        return False
    recent = [
        item
        for item in memory.pronoun_patterns[-4:]
        if isinstance(item.get("segmentId"), int)
        and abs(int(item["segmentId"]) - segment_id) <= 4
    ]
    if not recent:
        return False
    pairs = {
        tuple(item["pair"])
        for item in recent
        if isinstance(item.get("pair"), (tuple, list)) and len(item["pair"]) == 2
    }
    return bool(pairs and current not in pairs)
