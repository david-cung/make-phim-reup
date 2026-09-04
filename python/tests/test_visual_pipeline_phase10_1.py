from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from movie_translator_worker.translation.handlers import configure, translate_translate
from movie_translator_worker.translation.identity import (
    ActiveSpeakerResolver,
    BasicFaceIdentityMatcher,
    FaceTrack,
)
from movie_translator_worker.translation.memory import TranslationMemory
from movie_translator_worker.translation.models import TranslatedSegment
from movie_translator_worker.visual.analyzer import (
    VideoVisualAnalyzer,
    VisualAnalysisConfig,
    _Detection,
)
from movie_translator_worker.translation.identity import BoundingBox


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


class _FakeDetector:
    name = "fake-runtime-detector"
    model_path = None

    def __init__(self, *, two_faces: bool = False, gap: tuple[int, int] | None = None) -> None:
        self.two_faces = two_faces
        self.gap = gap
        self.calls = 0

    def detect(self, frame: Any, timestamp_ms: int, scale: float) -> list[_Detection]:
        self.calls += 1
        if self.gap and self.gap[0] <= timestamp_ms <= self.gap[1]:
            return []
        x = 20 + min(8, timestamp_ms // 250)
        out = [_Detection(BoundingBox(float(x), 20.0, 42.0, 50.0), 0.9)]
        if self.two_faces:
            out.append(_Detection(BoundingBox(95.0, 20.0, 42.0, 50.0), 0.88, embedding=[1.0, 0.0]))
        return out


class _FakeEmbedder:
    name = "fake-runtime-embedder"
    model_path = None

    def embed(self, frame: Any, detection: _Detection) -> list[float] | None:
        if detection.embedding is not None:
            return detection.embedding
        return [1.0, 0.0] if detection.bbox.x < 80 else [0.0, 1.0]


def _segment(seg_id: int = 1, speaker: str = "speaker_001") -> TranslatedSegment:
    return TranslatedSegment(
        id=seg_id,
        source_text=f"line {seg_id}",
        translation="",
        start=0.0,
        end=1.8,
        speaker_id=speaker,
        speaker_confidence=0.93,
    )


def _video(path: Path, *, mouth_motion: bool = True, head_motion: bool = False) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (160, 100),
    )
    assert writer.isOpened()
    for idx in range(18):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        shift = idx * 2 if head_motion else 0
        x1, y1, x2, y2 = 20 + shift, 20, 62 + shift, 70
        cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 210, 210), -1)
        cv2.circle(frame, (x1 + 12, y1 + 15), 3, (20, 20, 20), -1)
        cv2.circle(frame, (x1 + 30, y1 + 15), 3, (20, 20, 20), -1)
        mouth_y = y1 + 36 + (6 if mouth_motion and idx % 2 else 0)
        cv2.line(frame, (x1 + 12, mouth_y), (x1 + 30, mouth_y), (0, 0, 0), 3)
        if idx % 2 == 0:
            cv2.rectangle(frame, (95, 20), (137, 70), (180, 180, 180), -1)
        writer.write(frame)
    writer.release()
    return path


def test_a_video_visual_analyzer_decodes_frames_and_tracks_one_face(tmp_path: Path) -> None:
    detector = _FakeDetector()
    result = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=detector,
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=4.0, min_face_size_px=10),
    ).analyze(video_path=_video(tmp_path / "one.avi"), segments=[_segment()], cache_dir=tmp_path / "cache")

    assert result.status == "available"
    assert result.metrics["frames_sampled"] > 0
    assert result.metrics["faces_detected"] > 0
    assert result.metrics["face_tracks_created"] == 1
    assert result.face_tracks[0].face_track_id == "FACE_TRACK_0001"
    assert result.face_tracks[0].observations[0].bbox.width > 0


def test_b_two_people_create_independent_tracks(tmp_path: Path) -> None:
    result = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=_FakeDetector(two_faces=True),
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=4.0, min_face_size_px=10),
    ).analyze(video_path=_video(tmp_path / "two.avi"), segments=[_segment()], cache_dir=tmp_path / "cache")

    assert result.metrics["face_tracks_created"] == 2
    assert len({track.face_track_id for track in result.face_tracks}) == 2


def test_c_track_reentry_creates_new_track_but_reidentifies(tmp_path: Path) -> None:
    result = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=_FakeDetector(gap=(650, 1450)),
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=4.0, min_face_size_px=10, track_max_gap_ms=250),
    ).analyze(video_path=_video(tmp_path / "gap.avi"), segments=[_segment()], cache_dir=tmp_path / "cache")

    assert result.metrics["face_tracks_created"] >= 2
    graph = result.identity_graph.to_dict()
    linked = {item["character_id"] for item in graph["face_links"].values()}
    assert len(linked) == 1


def test_d_mouth_motion_scores_above_static_head_motion(tmp_path: Path) -> None:
    moving = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=8.0, min_face_size_px=10),
    ).analyze(video_path=_video(tmp_path / "mouth.avi", mouth_motion=True), segments=[_segment()], cache_dir=tmp_path / "c1")
    head = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=8.0, min_face_size_px=10),
    ).analyze(video_path=_video(tmp_path / "head.avi", mouth_motion=False, head_motion=True), segments=[_segment()], cache_dir=tmp_path / "c2")

    moving_score = max(o.mouth_activity_score or 0.0 for o in moving.face_tracks[0].observations)
    head_score = max(o.mouth_activity_score or 0.0 for o in head.face_tracks[0].observations)
    assert moving_score > head_score


def test_e_active_speaker_prefers_moving_mouth_over_inactive_face(tmp_path: Path) -> None:
    result = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=_FakeDetector(two_faces=True),
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=8.0, min_face_size_px=10),
    ).analyze(video_path=_video(tmp_path / "active.avi"), segments=[_segment()], cache_dir=tmp_path / "cache")

    candidates = ActiveSpeakerResolver().rank_candidates(
        segment_id="segment_1",
        speaker_id="speaker_001",
        start_ms=0,
        end_ms=1800,
        visible_face_tracks=result.face_tracks,
        audio_confidence=0.9,
    )
    assert candidates[0].face_track_id == "FACE_TRACK_0001"


def test_f_offscreen_speech_remains_possible_with_inactive_face() -> None:
    inactive = FaceTrack("FACE_TRACK_0001", 0, 1000, [], confidence=0.8)
    evidence = ActiveSpeakerResolver().resolve(
        segment_id="segment_1",
        speaker_id="speaker_001",
        start_ms=0,
        end_ms=1000,
        visible_face_tracks=[inactive],
        audio_confidence=0.9,
    )
    assert evidence.face_track_id is None
    assert evidence.evidence_type == "offscreen_speech"


def test_g_cooccurring_faces_do_not_merge_despite_similar_embeddings() -> None:
    left = FaceTrack("FACE_TRACK_0001", 0, 1000, embedding=[1.0, 0.0], confidence=0.95)
    right = FaceTrack("FACE_TRACK_0002", 500, 1300, embedding=[0.99, 0.01], confidence=0.95)
    assert BasicFaceIdentityMatcher().same_identity_confidence(left, right) == 0.0


def test_h_visual_cache_hit_reuses_previous_analysis(tmp_path: Path) -> None:
    detector = _FakeDetector()
    video = _video(tmp_path / "cache.avi")
    analyzer = VideoVisualAnalyzer(
        models_root=tmp_path,
        detector=detector,
        embedder=_FakeEmbedder(),
        config=VisualAnalysisConfig(scan_fps=4.0, min_face_size_px=10),
    )
    first = analyzer.analyze(video_path=video, segments=[_segment()], cache_dir=tmp_path / "cache")
    second = analyzer.analyze(video_path=video, segments=[_segment()], cache_dir=tmp_path / "cache")

    assert first.cache_hit is False
    assert second.status == "cached"
    assert second.cache_hit is True


def test_i_visual_failure_falls_back_without_raising(tmp_path: Path) -> None:
    result = VideoVisualAnalyzer(models_root=tmp_path).analyze(
        video_path=tmp_path / "missing.avi",
        segments=[_segment()],
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "unavailable"


def test_j_memory_store_consumes_visual_identity_context() -> None:
    seg = _segment()
    seg.pronoun_context["visualIdentity"] = {
        "speaker_id": "speaker_001",
        "character_id": "CHARACTER_777",
        "identity_confidence": 0.91,
        "visible_character_ids": ["CHARACTER_777", "CHARACTER_778"],
    }
    seg.pronoun_context["visualIdentityStatus"] = {"status": "available"}
    memory = TranslationMemory.from_segments([seg])

    store = memory.movie_memory.character_context_store
    assert store.speaker_profiles["speaker_001"].character_id == "CHARACTER_777"
    assert "CHARACTER_778" in store.character_contexts


def test_k_production_handler_executes_visual_analyzer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Provider:
        name = "fake"

        def translate_chunks(self, chunks, segments_by_id, options, ctx):
            ctx.on_chunk_completed(chunks[0].chunk_index, {})
            assert segments_by_id[1].pronoun_context["visualIdentity"]["character_id"] == "CHARACTER_001"
            return {}

    class _Analyzer:
        def __init__(self, *, models_root):
            self.models_root = models_root

        def analyze(self, *, video_path, segments, cache_dir=None, source_fingerprint=None):
            assert Path(video_path).name == "prod.avi"
            from movie_translator_worker.translation.identity import SegmentIdentityResolution
            return type(
                "Result",
                (),
                {
                    "status": "available",
                    "cache_hit": False,
                    "metrics": {"frames_sampled": 3},
                    "model_info": None,
                    "error": None,
                    "identity_graph": None,
                    "segment_resolutions": {
                        "segment_1": SegmentIdentityResolution(
                            "segment_1",
                            "speaker_001",
                            character_id="CHARACTER_001",
                            identity_confidence=0.9,
                        )
                    },
                },
            )()

    class _Ctx:
        def __init__(self) -> None:
            self.events = []

        def cancelled(self) -> bool:
            return False

        def emit_progress(self, method, params):
            self.events.append((method, params))

    models = tmp_path / "models"
    (models / "translation").mkdir(parents=True)
    (models / "translation" / "m.gguf").write_bytes(b"x")
    configure(models_root=models, provider=_Provider())
    monkeypatch.setattr("movie_translator_worker.translation.handlers.VideoVisualAnalyzer", _Analyzer)

    result = translate_translate(
        {
            "transcriptCacheKey": "t",
            "audioHash": "a",
            "videoPath": str(_video(tmp_path / "prod.avi")),
            "visualCacheDir": str(tmp_path / "visual-cache"),
            "sourceFingerprint": {"hash": "sha256:test"},
            "segments": [{"id": 1, "text": "hello", "start": 0, "end": 1, "speakerId": "speaker_001"}],
            "options": {"model": "m.gguf"},
        },
        _Ctx(),
    )

    assert result["visualIdentityStatus"]["status"] == "available"
