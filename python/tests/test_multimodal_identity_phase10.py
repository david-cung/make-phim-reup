from __future__ import annotations

from movie_translator_worker.translation.identity import (
    ActiveSpeakerEvidence,
    ActiveSpeakerResolver,
    BoundingBox,
    CharacterIdentityResolver,
    FaceObservation,
    FaceTrack,
    SpeakerIdentity,
    integrate_identity_graph_with_context_store,
)
from movie_translator_worker.translation.memory import (
    CharacterContextStore,
    TranslationMemory,
)
from movie_translator_worker.translation.models import TranslatedSegment


def _speaker(
    speaker_id: str,
    *,
    first: int = 0,
    last: int = 1000,
    confidence: float = 0.92,
) -> SpeakerIdentity:
    return SpeakerIdentity(
        speaker_id=speaker_id,
        segment_ids=["segment_1"],
        first_seen_ms=first,
        last_seen_ms=last,
        confidence=confidence,
        metadata={"source": "test"},
    )


def _face(
    face_id: str,
    *,
    start: int = 0,
    end: int = 1000,
    mouth: float | None = 0.8,
    visibility: float = 0.9,
    cluster: str | None = None,
    confidence: float = 0.9,
) -> FaceTrack:
    observations = [
        FaceObservation(
            timestamp_ms=start + (end - start) // 2,
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
            detection_confidence=confidence,
            visibility_score=visibility,
            mouth_activity_score=mouth,
        )
    ]
    return FaceTrack(
        face_track_id=face_id,
        start_ms=start,
        end_ms=end,
        observations=observations,
        cluster_id=cluster,
        confidence=confidence,
    )


def _evidence(
    segment_id: str,
    speaker_id: str,
    face_id: str | None,
    *,
    confidence: float,
    kind: str = "direct_active_speaker",
) -> ActiveSpeakerEvidence:
    return ActiveSpeakerEvidence(
        segment_id=segment_id,
        speaker_id=speaker_id,
        face_track_id=face_id,
        temporal_overlap_score=confidence,
        mouth_activity_score=confidence if face_id else 0.0,
        visibility_score=confidence if face_id else 0.0,
        audio_confidence=confidence,
        visual_confidence=confidence if face_id else 0.0,
        combined_confidence=confidence,
        evidence_type=kind,
    )


def _segment(segment_id: int, text: str, speaker_id: str, start: float) -> TranslatedSegment:
    return TranslatedSegment(
        id=segment_id,
        source_text=text,
        translation="",
        start=start,
        end=start + 0.8,
        speaker_id=speaker_id,
        speaker_confidence=0.93,
    )


def test_a_one_speaker_one_active_face_maps_to_same_character() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[_face("FACE_01")],
        active_speaker_evidence=[_evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.92)],
    )

    speaker_character = graph.speaker_links["SPEAKER_01"]["character_id"] if isinstance(graph.speaker_links["SPEAKER_01"], dict) else graph.speaker_links["SPEAKER_01"][0]
    face_character = graph.face_links["FACE_01"][0]
    assert speaker_character == face_character
    assert graph.characters[speaker_character].identity_confidence >= 0.9


def test_b_listener_face_must_not_become_speaker() -> None:
    candidates = ActiveSpeakerResolver().rank_candidates(
        segment_id="segment_1",
        speaker_id="SPEAKER_01",
        start_ms=0,
        end_ms=1000,
        visible_face_tracks=[
            _face("FACE_01", mouth=0.9),
            _face("FACE_02", mouth=0.05),
        ],
        audio_confidence=0.92,
    )

    assert candidates[0].face_track_id == "FACE_01"
    assert candidates[0].face_track_id != "FACE_02"


def test_c_offscreen_speaker_keeps_face_unassigned() -> None:
    evidence = ActiveSpeakerResolver().resolve(
        segment_id="segment_1",
        speaker_id="SPEAKER_01",
        start_ms=0,
        end_ms=1000,
        visible_face_tracks=[_face("FACE_02", mouth=0.02, visibility=0.8)],
        audio_confidence=0.91,
    )

    assert evidence.face_track_id is None
    assert evidence.evidence_type == "offscreen_speech"


def test_d_character_across_camera_cuts_merges_face_tracks() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[
            _face("FACE_01", cluster="person_a"),
            _face("FACE_09", cluster="person_a"),
            _face("FACE_17", cluster="person_a"),
        ],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.91),
            _evidence("segment_2", "SPEAKER_01", "FACE_09", confidence=0.88),
            _evidence("segment_3", "SPEAKER_01", "FACE_17", confidence=0.86),
        ],
    )

    ids = {graph.face_links[face][0] for face in ("FACE_01", "FACE_09", "FACE_17")}
    assert len(ids) == 1


def test_e_speaker_across_multiple_scenes_preserves_identity() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01", first=0, last=90_000)],
        face_tracks=[],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", None, confidence=0.7, kind="offscreen_speech"),
            _evidence("segment_80", "SPEAKER_01", None, confidence=0.7, kind="offscreen_speech"),
        ],
    )

    character_id = graph.speaker_links["SPEAKER_01"][0]
    assert graph.characters[character_id].last_seen_ms == 90_000


def test_f_weak_evidence_does_not_lock_identity() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01", confidence=0.45)],
        face_tracks=[_face("FACE_03"), _face("FACE_09")],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_03", confidence=0.56),
            _evidence("segment_20", "SPEAKER_01", "FACE_09", confidence=0.92),
        ],
    )

    assert graph.speaker_links["SPEAKER_01"][0] == graph.face_links["FACE_09"][0]
    assert graph.face_links["FACE_03"][0] != graph.face_links["FACE_09"][0]
    assert graph.conflicts


def test_g_strong_historical_mapping_handles_offscreen_speech() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_02")],
        face_tracks=[_face("FACE_04")],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_02", "FACE_04", confidence=0.97),
            _evidence("segment_9", "SPEAKER_02", None, confidence=0.72, kind="offscreen_speech"),
        ],
    )
    resolver.enrich_segments(
        segment_ranges=[("segment_9", "SPEAKER_02", 9000, 9800)],
        face_tracks=[],
        active_candidates={
            "segment_9": ActiveSpeakerResolver().rank_candidates(
                segment_id="segment_9",
                speaker_id="SPEAKER_02",
                start_ms=9000,
                end_ms=9800,
                visible_face_tracks=[],
            )
        },
    )

    character_id = graph.speaker_links["SPEAKER_02"][0]
    assert graph.segment_resolutions["segment_9"].character_id == character_id
    assert graph.segment_resolutions["segment_9"].offscreen_speech is True


def test_h_multiple_speakers_do_not_leak_identities() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01"), _speaker("SPEAKER_02")],
        face_tracks=[_face("FACE_01"), _face("FACE_02")],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.93),
            _evidence("segment_2", "SPEAKER_02", "FACE_02", confidence=0.93),
        ],
    )

    assert graph.speaker_links["SPEAKER_01"][0] != graph.speaker_links["SPEAKER_02"][0]
    assert graph.face_links["FACE_01"][0] != graph.face_links["FACE_02"][0]


def test_i_overlapping_dialogue_keeps_independent_speakers() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[
            _speaker("SPEAKER_01", first=12_000, last=15_000),
            _speaker("SPEAKER_02", first=14_200, last=16_000),
        ],
        face_tracks=[
            _face("FACE_01", start=12_000, end=15_000),
            _face("FACE_02", start=14_200, end=16_000),
        ],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.90),
            _evidence("segment_2", "SPEAKER_02", "FACE_02", confidence=0.89),
        ],
    )

    assert graph.speaker_links["SPEAKER_01"][0] != graph.speaker_links["SPEAKER_02"][0]


def test_j_identity_conflict_is_recorded_without_silent_overwrite() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[_face("FACE_01"), _face("FACE_09")],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.95),
            _evidence("segment_2", "SPEAKER_01", "FACE_09", confidence=0.83),
        ],
    )

    assert graph.conflicts
    assert graph.speaker_links["SPEAKER_01"][0] == graph.face_links["FACE_01"][0]


def test_k_backward_identity_propagation_enriches_early_segment() -> None:
    resolver = CharacterIdentityResolver()
    graph = resolver.resolve(
        speakers=[_speaker("SPEAKER_03", first=0, last=200_000)],
        face_tracks=[_face("FACE_07", start=200_000, end=201_000)],
        active_speaker_evidence=[
            _evidence("segment_200", "SPEAKER_03", "FACE_07", confidence=0.98),
        ],
    )
    resolver.enrich_segments(
        segment_ranges=[("segment_10", "SPEAKER_03", 10_000, 10_800)],
        face_tracks=[],
    )

    assert graph.segment_resolutions["segment_10"].character_id == graph.speaker_links["SPEAKER_03"][0]


def test_l_no_cross_video_identity_leakage() -> None:
    graph_a = CharacterIdentityResolver(video_scope_id="video_a").resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[_face("FACE_01")],
        active_speaker_evidence=[_evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.92)],
    )
    graph_b = CharacterIdentityResolver(video_scope_id="video_b").resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[_face("FACE_01")],
        active_speaker_evidence=[],
    )

    assert graph_a.video_scope_id == "video_a"
    assert graph_b.video_scope_id == "video_b"
    assert graph_a is not graph_b
    assert graph_a.to_dict() != graph_b.to_dict()


def test_m_phase9_integration_uses_character_ids_not_face_ids() -> None:
    segments = [
        _segment(1, "哥哥，你听我说。", "SPEAKER_01", 0),
        _segment(2, "我在听。", "SPEAKER_02", 1),
        _segment(3, "哥哥，别走。", "SPEAKER_01", 2),
    ]
    memory = TranslationMemory.from_segments(segments)
    graph = CharacterIdentityResolver().resolve(
        speakers=[_speaker("SPEAKER_01"), _speaker("SPEAKER_02")],
        face_tracks=[_face("FACE_01"), _face("FACE_02")],
        active_speaker_evidence=[
            _evidence("segment_1", "SPEAKER_01", "FACE_01", confidence=0.91),
            _evidence("segment_2", "SPEAKER_02", "FACE_02", confidence=0.91),
        ],
    )
    store = integrate_identity_graph_with_context_store(
        memory.movie_memory.character_context_store,
        graph,
    )

    payload = store.to_dict()
    assert "FACE_01" not in {
        mapping["speaker_character_id"]
        for mapping in payload["pronoun_mappings"]
    }
    assert payload["multimodal_extension_points"]["phase10_identity_graph"]


def test_n_gender_does_not_select_pronouns() -> None:
    store = CharacterContextStore()
    graph = CharacterIdentityResolver().resolve(
        speakers=[_speaker("SPEAKER_01")],
        face_tracks=[],
        active_speaker_evidence=[],
    )
    character = graph.character_for_speaker("SPEAKER_01")
    assert character is not None
    character.gender = "female"
    character.gender_confidence = 0.91
    store = integrate_identity_graph_with_context_store(store, graph)

    assert store.pronoun_mappings == []
    assert store.character_contexts[character.character_id].gender_hint == "female"
