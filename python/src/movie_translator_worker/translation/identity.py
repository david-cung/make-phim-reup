"""Phase 10 multimodal character identity primitives.

The current app has audio diarization but no bundled face tracking model.
This module provides the video-scoped identity layer that can consume face
tracks and active-speaker evidence when available, while degrading to the
existing audio/dialogue pipeline when visual evidence is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from .memory import (
    CharacterContext,
    CharacterContextStore,
    SpeakerProfile,
)


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class FaceObservation:
    timestamp_ms: int
    bbox: BoundingBox
    detection_confidence: float
    visibility_score: float
    mouth_activity_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "bbox": self.bbox.to_dict(),
            "detection_confidence": round(float(self.detection_confidence), 3),
            "visibility_score": round(float(self.visibility_score), 3),
            "mouth_activity_score": (
                round(float(self.mouth_activity_score), 3)
                if self.mouth_activity_score is not None
                else None
            ),
        }


@dataclass
class FaceTrack:
    face_track_id: str
    start_ms: int
    end_ms: int
    observations: list[FaceObservation] = field(default_factory=list)
    embedding: list[float] | None = None
    cluster_id: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_track_id": self.face_track_id,
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "observations": [item.to_dict() for item in self.observations[:24]],
            "embedding_present": self.embedding is not None,
            "cluster_id": self.cluster_id,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class SpeakerIdentity:
    speaker_id: str
    segment_ids: list[str] = field(default_factory=list)
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    voice_embedding: list[float] | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "segment_ids": list(self.segment_ids[:48]),
            "first_seen_ms": int(self.first_seen_ms),
            "last_seen_ms": int(self.last_seen_ms),
            "voice_embedding_present": self.voice_embedding is not None,
            "confidence": round(float(self.confidence), 3),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IdentityEvidence:
    evidence_type: str
    confidence: float
    source_id: str = ""
    target_id: str = ""
    segment_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "confidence": round(float(self.confidence), 3),
            "source_id": self.source_id,
            "target_id": self.target_id,
            "segment_id": self.segment_id,
            "details": _compact_details(self.details),
        }


@dataclass
class ActiveSpeakerEvidence:
    segment_id: str
    speaker_id: str
    face_track_id: str | None
    temporal_overlap_score: float
    mouth_activity_score: float
    visibility_score: float
    audio_confidence: float
    visual_confidence: float
    combined_confidence: float
    evidence_type: str = "weak_candidate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "face_track_id": self.face_track_id,
            "temporal_overlap_score": round(float(self.temporal_overlap_score), 3),
            "mouth_activity_score": round(float(self.mouth_activity_score), 3),
            "visibility_score": round(float(self.visibility_score), 3),
            "audio_confidence": round(float(self.audio_confidence), 3),
            "visual_confidence": round(float(self.visual_confidence), 3),
            "combined_confidence": round(float(self.combined_confidence), 3),
            "evidence_type": self.evidence_type,
        }


@dataclass(frozen=True)
class ActiveSpeakerCandidate:
    face_track_id: str | None
    confidence: float
    evidence: ActiveSpeakerEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_track_id": self.face_track_id,
            "confidence": round(float(self.confidence), 3),
            "evidence": self.evidence.to_dict(),
        }


@dataclass
class CharacterIdentity:
    character_id: str
    speaker_ids: set[str] = field(default_factory=set)
    face_track_ids: set[str] = field(default_factory=set)
    display_name: str | None = None
    gender: str = "unknown"
    gender_confidence: float = 0.0
    identity_confidence: float = 0.0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    evidence: list[IdentityEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "speaker_ids": sorted(self.speaker_ids),
            "face_track_ids": sorted(self.face_track_ids),
            "display_name": self.display_name,
            "gender": self.gender,
            "gender_confidence": round(float(self.gender_confidence), 3),
            "identity_confidence": round(float(self.identity_confidence), 3),
            "first_seen_ms": int(self.first_seen_ms),
            "last_seen_ms": int(self.last_seen_ms),
            "evidence": [item.to_dict() for item in self.evidence[-24:]],
            "metadata": dict(self.metadata),
        }


@dataclass
class SegmentIdentityResolution:
    segment_id: str
    speaker_id: str | None
    character_id: str | None = None
    listener_character_ids: list[str] = field(default_factory=list)
    visible_character_ids: list[str] = field(default_factory=list)
    active_face_track_id: str | None = None
    identity_confidence: float = 0.0
    offscreen_speech: bool = False
    identity_evidence: list[IdentityEvidence] = field(default_factory=list)
    top_candidates: list[ActiveSpeakerCandidate] = field(default_factory=list)
    conflicts: list[IdentityEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "character_id": self.character_id,
            "listener_character_ids": list(self.listener_character_ids),
            "visible_character_ids": list(self.visible_character_ids),
            "active_face_track_id": self.active_face_track_id,
            "identity_confidence": round(float(self.identity_confidence), 3),
            "offscreen_speech": bool(self.offscreen_speech),
            "identity_evidence": [item.to_dict() for item in self.identity_evidence[-8:]],
            "top_candidates": [item.to_dict() for item in self.top_candidates[:4]],
            "conflicts": [item.to_dict() for item in self.conflicts[-8:]],
        }


class FaceIdentityMatcher(Protocol):
    def same_identity_confidence(self, left: FaceTrack, right: FaceTrack) -> float:
        ...


class VoiceIdentityMatcher(Protocol):
    def same_identity_confidence(self, left: SpeakerIdentity, right: SpeakerIdentity) -> float:
        ...


class BasicFaceIdentityMatcher:
    def same_identity_confidence(self, left: FaceTrack, right: FaceTrack) -> float:
        if left.face_track_id == right.face_track_id:
            return 1.0
        if not tracks_can_share_identity(left, right):
            return 0.0
        if left.cluster_id and left.cluster_id == right.cluster_id:
            return min(left.confidence, right.confidence, 0.94)
        if left.embedding is not None and right.embedding is not None:
            return max(0.0, _cosine(left.embedding, right.embedding))
        return 0.0


class BasicVoiceIdentityMatcher:
    def same_identity_confidence(self, left: SpeakerIdentity, right: SpeakerIdentity) -> float:
        if left.speaker_id == right.speaker_id:
            return 1.0
        if left.voice_embedding is not None and right.voice_embedding is not None:
            return max(0.0, _cosine(left.voice_embedding, right.voice_embedding))
        return 0.0


class ActiveSpeakerResolver:
    def rank_candidates(
        self,
        *,
        segment_id: str,
        speaker_id: str,
        start_ms: int,
        end_ms: int,
        visible_face_tracks: Iterable[FaceTrack],
        audio_confidence: float = 1.0,
    ) -> list[ActiveSpeakerCandidate]:
        faces = list(visible_face_tracks)
        candidates = [
            self._candidate_for_face(
                segment_id=segment_id,
                speaker_id=speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                face=face,
                audio_confidence=audio_confidence,
            )
            for face in faces
        ]
        best = max((candidate.confidence for candidate in candidates), default=0.0)
        max_mouth = max(
            (candidate.evidence.mouth_activity_score for candidate in candidates),
            default=0.0,
        )
        offscreen_confidence = 0.0
        if not faces:
            offscreen_confidence = 0.55
        elif best < 0.65 and max_mouth < 0.25:
            offscreen_confidence = 0.72
        if offscreen_confidence:
            evidence = ActiveSpeakerEvidence(
                segment_id=segment_id,
                speaker_id=speaker_id,
                face_track_id=None,
                temporal_overlap_score=0.0,
                mouth_activity_score=0.0,
                visibility_score=0.0,
                audio_confidence=audio_confidence,
                visual_confidence=0.0,
                combined_confidence=offscreen_confidence,
                evidence_type="offscreen_speech",
            )
            candidates.append(
                ActiveSpeakerCandidate(None, offscreen_confidence, evidence)
            )
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    def resolve(
        self,
        *,
        segment_id: str,
        speaker_id: str,
        start_ms: int,
        end_ms: int,
        visible_face_tracks: Iterable[FaceTrack],
        audio_confidence: float = 1.0,
    ) -> ActiveSpeakerEvidence:
        candidates = self.rank_candidates(
            segment_id=segment_id,
            speaker_id=speaker_id,
            start_ms=start_ms,
            end_ms=end_ms,
            visible_face_tracks=visible_face_tracks,
            audio_confidence=audio_confidence,
        )
        if not candidates:
            return ActiveSpeakerEvidence(
                segment_id=segment_id,
                speaker_id=speaker_id,
                face_track_id=None,
                temporal_overlap_score=0.0,
                mouth_activity_score=0.0,
                visibility_score=0.0,
                audio_confidence=audio_confidence,
                visual_confidence=0.0,
                combined_confidence=0.0,
                evidence_type="weak_candidate",
            )
        return candidates[0].evidence

    def _candidate_for_face(
        self,
        *,
        segment_id: str,
        speaker_id: str,
        start_ms: int,
        end_ms: int,
        face: FaceTrack,
        audio_confidence: float,
    ) -> ActiveSpeakerCandidate:
        overlap = _range_overlap(start_ms, end_ms, face.start_ms, face.end_ms)
        duration = max(1, end_ms - start_ms)
        temporal = min(1.0, overlap / duration)
        observations = [
            item
            for item in face.observations
            if start_ms <= item.timestamp_ms <= end_ms
        ]
        visibility = _mean(
            [item.visibility_score * item.detection_confidence for item in observations]
        )
        if not observations and overlap > 0:
            visibility = face.confidence * 0.55
        mouth = _mean(
            [
                item.mouth_activity_score
                for item in observations
                if item.mouth_activity_score is not None
            ]
        )
        visual = min(1.0, 0.55 * mouth + 0.45 * visibility)
        combined = min(
            1.0,
            0.42 * temporal
            + 0.34 * mouth
            + 0.16 * visibility
            + 0.08 * audio_confidence,
        )
        evidence_type = "weak_candidate"
        if combined >= 0.65 and mouth >= 0.50:
            evidence_type = "direct_active_speaker"
        elif temporal > 0:
            evidence_type = "temporal_overlap"
        evidence = ActiveSpeakerEvidence(
            segment_id=segment_id,
            speaker_id=speaker_id,
            face_track_id=face.face_track_id,
            temporal_overlap_score=temporal,
            mouth_activity_score=mouth,
            visibility_score=visibility,
            audio_confidence=audio_confidence,
            visual_confidence=visual,
            combined_confidence=combined,
            evidence_type=evidence_type,
        )
        return ActiveSpeakerCandidate(face.face_track_id, combined, evidence)


class MultimodalIdentityGraph:
    def __init__(self, video_scope_id: str = "current_video") -> None:
        self.video_scope_id = video_scope_id
        self.characters: dict[str, CharacterIdentity] = {}
        self.speaker_links: dict[str, tuple[str, float]] = {}
        self.face_links: dict[str, tuple[str, float]] = {}
        self.segment_resolutions: dict[str, SegmentIdentityResolution] = {}
        self.conflicts: list[IdentityEvidence] = []
        self.same_identity_evidence: dict[tuple[str, str], list[IdentityEvidence]] = {}
        self.different_identity_evidence: dict[tuple[str, str], list[IdentityEvidence]] = {}
        self.merge_decisions: list[dict[str, Any]] = []
        self._next_character = 1

    def new_character(self) -> CharacterIdentity:
        cid = f"CHARACTER_{self._next_character:03d}"
        self._next_character += 1
        character = CharacterIdentity(character_id=cid)
        self.characters[cid] = character
        return character

    def character_for_speaker(self, speaker_id: str | None) -> CharacterIdentity | None:
        if not speaker_id:
            return None
        link = self.speaker_links.get(speaker_id)
        return self.characters.get(link[0]) if link else None

    def character_for_face(self, face_track_id: str | None) -> CharacterIdentity | None:
        if not face_track_id:
            return None
        link = self.face_links.get(face_track_id)
        return self.characters.get(link[0]) if link else None

    def link_speaker(
        self,
        speaker_id: str,
        character: CharacterIdentity,
        confidence: float,
        evidence: IdentityEvidence,
    ) -> None:
        existing = self.speaker_links.get(speaker_id)
        if existing and existing[0] != character.character_id:
            self._handle_conflict(
                observation_id=speaker_id,
                old=existing,
                new_character=character,
                confidence=confidence,
                evidence=evidence,
                kind="speaker_identity_conflict",
                link_table=self.speaker_links,
            )
            if self.speaker_links.get(speaker_id, ("", 0.0))[0] != character.character_id:
                return
            for other in self.characters.values():
                if other.character_id != character.character_id:
                    other.speaker_ids.discard(speaker_id)
        else:
            self.speaker_links[speaker_id] = (
                character.character_id,
                _accumulate(existing[1] if existing else 0.0, confidence),
            )
        character.speaker_ids.add(speaker_id)
        character.evidence.append(evidence)
        character.identity_confidence = max(
            character.identity_confidence,
            self.speaker_links[speaker_id][1],
            confidence,
        )

    def link_face(
        self,
        face_track_id: str,
        character: CharacterIdentity,
        confidence: float,
        evidence: IdentityEvidence,
    ) -> None:
        for existing_face in list(character.face_track_ids):
            if self.different_identity_confidence(existing_face, face_track_id) >= 0.95:
                split = self.new_character()
                split.metadata["split_from_character"] = character.character_id
                character = split
                break
        existing = self.face_links.get(face_track_id)
        if existing and existing[0] != character.character_id:
            self._handle_conflict(
                observation_id=face_track_id,
                old=existing,
                new_character=character,
                confidence=confidence,
                evidence=evidence,
                kind="face_identity_conflict",
                link_table=self.face_links,
            )
            if self.face_links.get(face_track_id, ("", 0.0))[0] != character.character_id:
                return
            for other in self.characters.values():
                if other.character_id != character.character_id:
                    other.face_track_ids.discard(face_track_id)
        else:
            self.face_links[face_track_id] = (
                character.character_id,
                _accumulate(existing[1] if existing else 0.0, confidence),
            )
        character.face_track_ids.add(face_track_id)
        character.evidence.append(evidence)
        character.identity_confidence = max(
            character.identity_confidence,
            self.face_links[face_track_id][1],
            confidence,
        )

    def record_same_identity(self, left: str, right: str, evidence: IdentityEvidence) -> None:
        self.same_identity_evidence.setdefault(_pair_key(left, right), []).append(evidence)

    def record_different_identity(self, left: str, right: str, evidence: IdentityEvidence) -> None:
        self.different_identity_evidence.setdefault(_pair_key(left, right), []).append(evidence)

    def different_identity_confidence(self, left: str, right: str) -> float:
        items = self.different_identity_evidence.get(_pair_key(left, right), [])
        return max((item.confidence for item in items), default=0.0)

    def _handle_conflict(
        self,
        *,
        observation_id: str,
        old: tuple[str, float],
        new_character: CharacterIdentity,
        confidence: float,
        evidence: IdentityEvidence,
        kind: str,
        link_table: dict[str, tuple[str, float]],
    ) -> None:
        conflict = IdentityEvidence(
            evidence_type=kind,
            confidence=min(old[1], confidence),
            source_id=observation_id,
            target_id=new_character.character_id,
            segment_id=evidence.segment_id,
            details={
                "existing_character": old[0],
                "existing_confidence": round(float(old[1]), 3),
                "new_confidence": round(float(confidence), 3),
            },
        )
        self.conflicts.append(conflict)
        self.characters[old[0]].evidence.append(conflict)
        new_character.evidence.append(conflict)
        if old[1] < 0.70 and confidence >= old[1] + 0.15:
            link_table[observation_id] = (new_character.character_id, confidence)
            new_character.identity_confidence = max(
                new_character.identity_confidence,
                confidence,
            )
        elif old[1] >= 0.85 and confidence < old[1]:
            self.characters[old[0]].identity_confidence = min(
                self.characters[old[0]].identity_confidence,
                max(0.50, old[1] - 0.20),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_scope_id": self.video_scope_id,
            "characters": {
                cid: character.to_dict()
                for cid, character in sorted(self.characters.items())
            },
            "speaker_links": {
                speaker: {
                    "character_id": cid,
                    "confidence": round(float(conf), 3),
                }
                for speaker, (cid, conf) in sorted(self.speaker_links.items())
            },
            "face_links": {
                face: {
                    "character_id": cid,
                    "confidence": round(float(conf), 3),
                }
                for face, (cid, conf) in sorted(self.face_links.items())
            },
            "segment_resolutions": {
                sid: resolution.to_dict()
                for sid, resolution in sorted(self.segment_resolutions.items())
            },
            "conflicts": [item.to_dict() for item in self.conflicts[-24:]],
            "same_identity_evidence": {
                f"{left}<->{right}": [item.to_dict() for item in rows[-8:]]
                for (left, right), rows in sorted(self.same_identity_evidence.items())
            },
            "different_identity_evidence": {
                f"{left}<->{right}": [item.to_dict() for item in rows[-8:]]
                for (left, right), rows in sorted(self.different_identity_evidence.items())
            },
            "merge_decisions": list(self.merge_decisions[-48:]),
        }


class CharacterIdentityResolver:
    def __init__(
        self,
        *,
        face_matcher: FaceIdentityMatcher | None = None,
        voice_matcher: VoiceIdentityMatcher | None = None,
        video_scope_id: str = "current_video",
        face_merge_threshold: float = 0.90,
    ) -> None:
        self.face_matcher = face_matcher or BasicFaceIdentityMatcher()
        self.voice_matcher = voice_matcher or BasicVoiceIdentityMatcher()
        self.graph = MultimodalIdentityGraph(video_scope_id=video_scope_id)
        self.face_merge_threshold = face_merge_threshold

    def resolve(
        self,
        *,
        speakers: Iterable[SpeakerIdentity],
        face_tracks: Iterable[FaceTrack],
        active_speaker_evidence: Iterable[ActiveSpeakerEvidence],
    ) -> MultimodalIdentityGraph:
        speakers_by_id = {speaker.speaker_id: speaker for speaker in speakers}
        faces_by_id = {face.face_track_id: face for face in face_tracks}
        self._record_face_negative_evidence(list(faces_by_id.values()))
        self._resolve_voice_reidentification(list(speakers_by_id.values()))
        self._resolve_face_reidentification(list(faces_by_id.values()))
        for speaker in speakers_by_id.values():
            character = self.graph.character_for_speaker(speaker.speaker_id)
            if character is None:
                character = self.graph.new_character()
            self.graph.link_speaker(
                speaker.speaker_id,
                character,
                max(0.35, speaker.confidence),
                IdentityEvidence(
                    "voice_identity",
                    max(0.35, speaker.confidence),
                    source_id=speaker.speaker_id,
                    target_id=character.character_id,
                ),
            )
            _touch_character(character, speaker.first_seen_ms, speaker.last_seen_ms)

        for evidence in active_speaker_evidence:
            self._apply_active_speaker_evidence(evidence, faces_by_id)
        self._ensure_face_characters(list(faces_by_id.values()))
        return self.graph

    def _resolve_voice_reidentification(self, speakers: list[SpeakerIdentity]) -> None:
        for left_index, left in enumerate(speakers):
            for right in speakers[left_index + 1 :]:
                confidence = self.voice_matcher.same_identity_confidence(left, right)
                if confidence < 0.90:
                    continue
                character = (
                    self.graph.character_for_speaker(left.speaker_id)
                    or self.graph.character_for_speaker(right.speaker_id)
                    or self.graph.new_character()
                )
                for speaker in (left, right):
                    self.graph.link_speaker(
                        speaker.speaker_id,
                        character,
                        confidence,
                        IdentityEvidence(
                            "voice_identity",
                            confidence,
                            source_id=speaker.speaker_id,
                            target_id=character.character_id,
                        ),
                    )

    def _resolve_face_reidentification(self, faces: list[FaceTrack]) -> None:
        for left_index, left in enumerate(faces):
            for right in faces[left_index + 1 :]:
                negative = self.graph.different_identity_confidence(
                    left.face_track_id,
                    right.face_track_id,
                )
                confidence = self.face_matcher.same_identity_confidence(left, right)
                quality = min(left.confidence, right.confidence)
                threshold = self.face_merge_threshold
                decision = "reject_merge"
                reason = "below_threshold"
                if negative >= 0.95:
                    reason = "co_occurrence_conflict"
                elif quality < 0.55:
                    reason = "low_track_quality"
                elif confidence >= threshold and quality >= 0.55:
                    decision = "merge"
                    reason = "strong_embedding_no_negative_evidence"
                self.graph.merge_decisions.append(
                    {
                        "track_a": left.face_track_id,
                        "track_b": right.face_track_id,
                        "embedding_similarity": round(float(confidence), 3),
                        "track_quality": round(float(quality), 3),
                        "negative_identity_confidence": round(float(negative), 3),
                        "threshold": threshold,
                        "decision": decision,
                        "reason": reason,
                    }
                )
                if decision != "merge":
                    continue
                self.graph.record_same_identity(
                    left.face_track_id,
                    right.face_track_id,
                    IdentityEvidence(
                        "face_embedding_similarity",
                        confidence,
                        source_id=left.face_track_id,
                        target_id=right.face_track_id,
                        details={"track_quality": round(float(quality), 3)},
                    ),
                )
                character = (
                    self.graph.character_for_face(left.face_track_id)
                    or self.graph.character_for_face(right.face_track_id)
                    or self.graph.new_character()
                )
                for face in (left, right):
                    self.graph.link_face(
                        face.face_track_id,
                        character,
                        confidence,
                        IdentityEvidence(
                            "face_identity",
                            confidence,
                            source_id=face.face_track_id,
                            target_id=character.character_id,
                        ),
                    )
                    _touch_character(character, face.start_ms, face.end_ms)

    def _record_face_negative_evidence(self, faces: list[FaceTrack]) -> None:
        for left_index, left in enumerate(faces):
            for right in faces[left_index + 1 :]:
                if tracks_can_share_identity(left, right):
                    continue
                self.graph.record_different_identity(
                    left.face_track_id,
                    right.face_track_id,
                    IdentityEvidence(
                        "cannot_be_same_character",
                        1.0,
                        source_id=left.face_track_id,
                        target_id=right.face_track_id,
                        details={
                            "reason": "simultaneous_independent_visibility",
                            "temporal_overlap_ms": _range_overlap(
                                left.start_ms,
                                left.end_ms,
                                right.start_ms,
                                right.end_ms,
                            ),
                            "max_observation_iou": round(
                                _max_simultaneous_iou(left, right),
                                3,
                            ),
                        },
                    ),
                )

    def _ensure_face_characters(self, faces: list[FaceTrack]) -> None:
        for face in faces:
            if face.confidence < 0.45:
                continue
            if self.graph.character_for_face(face.face_track_id) is not None:
                continue
            character = self.graph.new_character()
            self.graph.link_face(
                face.face_track_id,
                character,
                max(0.45, face.confidence * 0.75),
                IdentityEvidence(
                    "visible_face_identity",
                    max(0.45, face.confidence * 0.75),
                    source_id=face.face_track_id,
                    target_id=character.character_id,
                    details={"reason": "visible_non_speaking_character"},
                ),
            )
            _touch_character(character, face.start_ms, face.end_ms)

    def _apply_active_speaker_evidence(
        self,
        evidence: ActiveSpeakerEvidence,
        faces_by_id: dict[str, FaceTrack],
    ) -> None:
        speaker_character = self.graph.character_for_speaker(evidence.speaker_id)
        face_character = self.graph.character_for_face(evidence.face_track_id)
        target = self._target_character_for_evidence(
            evidence=evidence,
            speaker_character=speaker_character,
            face_character=face_character,
        )
        if target is None:
            target = self.graph.new_character()
        speaker_evidence = IdentityEvidence(
            evidence.evidence_type,
            evidence.combined_confidence,
            source_id=evidence.speaker_id,
            target_id=target.character_id,
            segment_id=evidence.segment_id,
            details={"face_track_id": evidence.face_track_id},
        )
        self.graph.link_speaker(
            evidence.speaker_id,
            target,
            max(evidence.audio_confidence * 0.65, evidence.combined_confidence),
            speaker_evidence,
        )
        speaker_link = self.graph.speaker_links.get(evidence.speaker_id)
        face_visual_conflict = (
            evidence.face_track_id is not None
            and face_character is None
            and speaker_character is not None
            and bool(speaker_character.face_track_ids)
            and speaker_link is not None
            and speaker_link[1] >= 0.85
            and evidence.combined_confidence < 0.85
        )
        if face_visual_conflict:
            conflict = IdentityEvidence(
                "conflict",
                evidence.combined_confidence,
                source_id=evidence.face_track_id or "",
                target_id=target.character_id,
                segment_id=evidence.segment_id,
                details={
                    "reason": "moderate_visual_candidate_conflicts_with_strong_speaker_history",
                    "speaker_id": evidence.speaker_id,
                },
            )
            self.graph.conflicts.append(conflict)
            target.evidence.append(conflict)
            target.identity_confidence = min(
                target.identity_confidence,
                max(0.50, speaker_link[1] - 0.20),
            )
            return
        if evidence.face_track_id and evidence.combined_confidence >= 0.55:
            face = faces_by_id.get(evidence.face_track_id)
            self.graph.link_face(
                evidence.face_track_id,
                target,
                evidence.combined_confidence,
                IdentityEvidence(
                    evidence.evidence_type,
                    evidence.combined_confidence,
                    source_id=evidence.face_track_id,
                    target_id=target.character_id,
                    segment_id=evidence.segment_id,
                    details={"speaker_id": evidence.speaker_id},
                ),
            )
            if face is not None:
                _touch_character(target, face.start_ms, face.end_ms)

    def _target_character_for_evidence(
        self,
        *,
        evidence: ActiveSpeakerEvidence,
        speaker_character: CharacterIdentity | None,
        face_character: CharacterIdentity | None,
    ) -> CharacterIdentity | None:
        if evidence.face_track_id is None:
            return speaker_character
        speaker_link = self.graph.speaker_links.get(evidence.speaker_id)
        face_link = self.graph.face_links.get(evidence.face_track_id)
        if speaker_character is not None and face_character is not None:
            if speaker_character.character_id == face_character.character_id:
                return speaker_character
            if speaker_link and speaker_link[1] >= 0.85 and evidence.combined_confidence < speaker_link[1]:
                return speaker_character
            if speaker_link and speaker_link[1] < 0.70 and evidence.combined_confidence >= speaker_link[1] + 0.15:
                return face_character
            return None
        if speaker_character is not None:
            if (
                speaker_link
                and speaker_link[1] < 0.70
                and evidence.combined_confidence >= speaker_link[1] + 0.15
            ):
                return self.graph.new_character()
            return speaker_character
        if face_character is not None:
            return face_character
        return None

    def enrich_segments(
        self,
        *,
        segment_ranges: Iterable[tuple[str, str | None, int, int]],
        face_tracks: Iterable[FaceTrack],
        active_candidates: dict[str, list[ActiveSpeakerCandidate]] | None = None,
    ) -> dict[str, SegmentIdentityResolution]:
        faces = list(face_tracks)
        out: dict[str, SegmentIdentityResolution] = {}
        for segment_id, speaker_id, start_ms, end_ms in segment_ranges:
            character = self.graph.character_for_speaker(speaker_id)
            visible_faces = [
                face
                for face in faces
                if _range_overlap(start_ms, end_ms, face.start_ms, face.end_ms) > 0
            ]
            active_face_id = None
            offscreen = False
            evidence: list[IdentityEvidence] = []
            candidates = (active_candidates or {}).get(segment_id, [])
            if candidates:
                active_face_id = candidates[0].face_track_id
                offscreen = active_face_id is None and candidates[0].confidence >= 0.50
                evidence.append(
                    IdentityEvidence(
                        candidates[0].evidence.evidence_type,
                        candidates[0].confidence,
                        source_id=speaker_id or "",
                        target_id=active_face_id or "",
                        segment_id=segment_id,
                    )
                )
            visible_character_ids = [
                found.character_id
                for face in visible_faces
                if (found := self.graph.character_for_face(face.face_track_id)) is not None
            ]
            listener_ids = [
                cid
                for cid in dict.fromkeys(visible_character_ids)
                if character is None or cid != character.character_id
            ][:2]
            resolution = SegmentIdentityResolution(
                segment_id=segment_id,
                speaker_id=speaker_id,
                character_id=character.character_id if character is not None else None,
                listener_character_ids=listener_ids,
                visible_character_ids=list(dict.fromkeys(visible_character_ids)),
                active_face_track_id=active_face_id,
                identity_confidence=(
                    character.identity_confidence if character is not None else 0.0
                ),
                offscreen_speech=offscreen,
                identity_evidence=evidence,
                top_candidates=candidates[:4],
                conflicts=[
                    conflict
                    for conflict in self.graph.conflicts
                    if conflict.segment_id == segment_id
                ],
            )
            self.graph.segment_resolutions[segment_id] = resolution
            out[segment_id] = resolution
        return out


def speaker_identities_from_segments(
    segments: Iterable[Any],
) -> list[SpeakerIdentity]:
    grouped: dict[str, list[Any]] = {}
    for seg in segments:
        speaker_id = getattr(seg, "speaker_id", None)
        if not speaker_id:
            continue
        grouped.setdefault(str(speaker_id), []).append(seg)
    speakers: list[SpeakerIdentity] = []
    for speaker_id, rows in sorted(grouped.items()):
        first = min(float(getattr(seg, "start", 0.0)) for seg in rows)
        last = max(float(getattr(seg, "end", 0.0)) for seg in rows)
        confidences = [
            float(getattr(seg, "speaker_confidence"))
            for seg in rows
            if getattr(seg, "speaker_confidence", None) is not None
        ]
        speakers.append(
            SpeakerIdentity(
                speaker_id=speaker_id,
                segment_ids=[_segment_key(getattr(seg, "id", idx)) for idx, seg in enumerate(rows)],
                first_seen_ms=int(first * 1000),
                last_seen_ms=int(last * 1000),
                confidence=_mean(confidences) if confidences else 0.45,
                metadata={"source": "diarization"},
            )
        )
    return speakers


def integrate_identity_graph_with_context_store(
    store: CharacterContextStore,
    graph: MultimodalIdentityGraph,
) -> CharacterContextStore:
    for speaker_id, (character_id, confidence) in graph.speaker_links.items():
        profile = store.speaker_profiles.get(speaker_id)
        if profile is None:
            store.speaker_profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                character_id=character_id,
                gender_hint="unknown",
                gender_confidence=0.0,
                voice_evidence={
                    "identity_confidence": round(float(confidence), 3),
                    "source": "multimodal_identity_graph",
                },
            )
        else:
            profile.character_id = character_id
            profile.voice_evidence["identity_confidence"] = round(float(confidence), 3)
            profile.voice_evidence["source"] = "multimodal_identity_graph"
    for character_id, character in graph.characters.items():
        context = store.character_contexts.get(character_id)
        if context is None:
            store.character_contexts[character_id] = CharacterContext(
                character_id=character_id,
                associated_speaker_ids=sorted(character.speaker_ids),
                gender_hint=character.gender,
                gender_confidence=character.gender_confidence,
                confidence=character.identity_confidence,
            )
        else:
            context.associated_speaker_ids = sorted(
                set(context.associated_speaker_ids).union(character.speaker_ids)
            )
            context.confidence = max(context.confidence, character.identity_confidence)
    store.multimodal_extension_points["phase10_identity_graph"] = graph.to_dict()
    return store


def _touch_character(character: CharacterIdentity, start_ms: int, end_ms: int) -> None:
    if character.first_seen_ms == 0:
        character.first_seen_ms = int(start_ms)
    else:
        character.first_seen_ms = min(character.first_seen_ms, int(start_ms))
    character.last_seen_ms = max(character.last_seen_ms, int(end_ms))


def _range_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def tracks_can_share_identity(left: FaceTrack, right: FaceTrack) -> bool:
    """Return False for simultaneous independent faces.

    The only co-occurrence exception is a likely duplicate detection of
    the same physical face: same/similar timestamp and very high bbox
    overlap. Embedding similarity alone is deliberately ignored here.
    """
    if _range_overlap(left.start_ms, left.end_ms, right.start_ms, right.end_ms) <= 0:
        return True
    return _max_simultaneous_iou(left, right) >= 0.88


def _max_simultaneous_iou(left: FaceTrack, right: FaceTrack, *, tolerance_ms: int = 80) -> float:
    best = 0.0
    for a in left.observations:
        for b in right.observations:
            if abs(a.timestamp_ms - b.timestamp_ms) <= tolerance_ms:
                best = max(best, _bbox_iou(a.bbox, b.bbox))
    return best


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    lx2 = left.x + left.width
    ly2 = left.y + left.height
    rx2 = right.x + right.width
    ry2 = right.y + right.height
    iw = max(0.0, min(lx2, rx2) - max(left.x, right.x))
    ih = max(0.0, min(ly2, ry2) - max(left.y, right.y))
    inter = iw * ih
    union = left.width * left.height + right.width * right.height - inter
    return inter / union if union > 0 else 0.0


def _pair_key(left: str, right: str) -> tuple[str, str]:
    ordered = sorted((str(left), str(right)))
    return (ordered[0], ordered[1])


def _mean(values: Iterable[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _accumulate(current: float, evidence: float) -> float:
    current = max(0.0, min(1.0, float(current)))
    evidence = max(0.0, min(1.0, float(evidence)))
    if current == 0.0:
        return evidence
    return min(0.99, current + (1.0 - current) * evidence * 0.55)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _segment_key(value: Any) -> str:
    if isinstance(value, str) and value.startswith("segment_"):
        return value
    try:
        return f"segment_{int(value)}"
    except (TypeError, ValueError):
        return str(value)


def _compact_details(details: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, list) and value and all(isinstance(item, float) for item in value):
            out[key] = "<embedding>"
        else:
            out[key] = value
    return out
