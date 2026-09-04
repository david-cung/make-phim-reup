"""Video-to-Phase-10 visual observation pipeline.

This module owns real video frame decoding, face observation generation,
lightweight tracking, appearance embeddings, mouth-motion estimation, and
cache IO. Identity fusion stays in ``translation.identity``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .. import logging as log
from ..translation.identity import (
    ActiveSpeakerResolver,
    BoundingBox,
    CharacterIdentityResolver,
    FaceObservation,
    FaceTrack,
    integrate_identity_graph_with_context_store,
    speaker_identities_from_segments,
)

VISUAL_ANALYSIS_VERSION = "visual_phase_10_1_v1"


@dataclass(frozen=True)
class VisualAnalysisConfig:
    enabled: bool = True
    scan_fps: float = 3.0
    dialogue_padding_ms: int = 350
    min_face_confidence: float = 0.55
    min_face_size_px: int = 32
    max_analysis_width: int = 960
    track_max_gap_ms: int = 800
    embedding_interval_frames: int = 5
    assignment_threshold: float = 0.35
    reid_threshold: float = 0.82
    identity_merge_threshold: float = 0.90
    debug_overlay_dir: str | None = None
    debug_overlay_max_frames: int = 24
    visual_cache_version: str = VISUAL_ANALYSIS_VERSION

    def signature(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            key: data[key]
            for key in (
                "scan_fps",
                "dialogue_padding_ms",
                "min_face_confidence",
                "min_face_size_px",
                "max_analysis_width",
                "track_max_gap_ms",
                "embedding_interval_frames",
                "assignment_threshold",
                "reid_threshold",
                "identity_merge_threshold",
                "visual_cache_version",
            )
        }


@dataclass(frozen=True)
class VisualModelInfo:
    detector: str
    detector_path: str | None
    embedder: str
    embedder_path: str | None
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualAnalysisResult:
    status: str
    face_tracks: list[FaceTrack] = field(default_factory=list)
    identity_graph: Any | None = None
    segment_resolutions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_info: VisualModelInfo | None = None
    error: str | None = None
    cache_hit: bool = False

    def to_payload(self) -> dict[str, Any]:
        graph_payload = (
            self.identity_graph.to_dict()
            if hasattr(self.identity_graph, "to_dict")
            else self.identity_graph
        )
        return {
            "status": self.status,
            "faceTracks": [_face_track_to_cache(track) for track in self.face_tracks],
            "identityGraph": graph_payload,
            "segmentResolutions": {
                sid: value.to_dict() if hasattr(value, "to_dict") else value
                for sid, value in self.segment_resolutions.items()
            },
            "metrics": dict(self.metrics),
            "modelInfo": self.model_info.to_dict() if self.model_info else None,
            "error": self.error,
            "cacheHit": self.cache_hit,
        }


@dataclass(frozen=True)
class _Detection:
    bbox: BoundingBox
    confidence: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    embedding: list[float] | None = None


class FaceDetector(Protocol):
    name: str
    model_path: str | None

    def detect(self, frame: Any, timestamp_ms: int, scale: float) -> list[_Detection]:
        ...


class FaceEmbedder(Protocol):
    name: str
    model_path: str | None

    def embed(self, frame: Any, detection: _Detection) -> list[float] | None:
        ...


class VideoVisualAnalyzer:
    def __init__(
        self,
        *,
        models_root: Path,
        config: VisualAnalysisConfig | None = None,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
    ) -> None:
        self.models_root = Path(models_root)
        self.config = config or VisualAnalysisConfig()
        self._detector = detector
        self._embedder = embedder

    def analyze(
        self,
        *,
        video_path: str | Path | None,
        segments: Iterable[Any],
        cache_dir: str | Path | None = None,
        source_fingerprint: dict[str, Any] | None = None,
    ) -> VisualAnalysisResult:
        started = time.monotonic()
        if not self.config.enabled:
            return VisualAnalysisResult("disabled", metrics={"cache_hit": False})
        if not video_path:
            return VisualAnalysisResult("unavailable", error="video path not provided")
        path = Path(video_path)
        if not path.is_file():
            return VisualAnalysisResult("unavailable", error=f"video not found: {path}")

        segment_list = list(segments)
        cache_root = Path(cache_dir) if cache_dir else path.parent / "visual"
        fingerprint = source_fingerprint or _fingerprint(path)
        try:
            detector = self._detector or _make_opencv_detector(self.models_root, self.config)
            embedder = self._embedder or _make_opencv_embedder(self.models_root)
        except Exception as e:
            return VisualAnalysisResult("unavailable", error=str(e))
        model_info = VisualModelInfo(
            detector=getattr(detector, "name", "unknown"),
            detector_path=getattr(detector, "model_path", None),
            embedder=getattr(embedder, "name", "none"),
            embedder_path=getattr(embedder, "model_path", None),
            version=self.config.visual_cache_version,
        )
        cache_key = _cache_key(fingerprint, self.config.signature(), model_info.to_dict())
        cached = _load_cache(cache_root, cache_key)
        if cached is not None:
            cached.cache_hit = True
            cached.status = "cached"
            cached.metrics["cache_hit"] = True
            return cached

        try:
            decoded = self._decode_and_track(path, segment_list, detector, embedder)
            resolver = CharacterIdentityResolver(
                video_scope_id=str(cache_key),
                face_merge_threshold=self.config.identity_merge_threshold,
            )
            speakers = speaker_identities_from_segments(segment_list)
            active = _active_evidence_for_segments(
                segments=segment_list,
                face_tracks=decoded.face_tracks,
            )
            graph = resolver.resolve(
                speakers=speakers,
                face_tracks=decoded.face_tracks,
                active_speaker_evidence=active,
            )
            candidates = _active_candidates_for_segments(segment_list, decoded.face_tracks)
            resolutions = resolver.enrich_segments(
                segment_ranges=[
                    (
                        f"segment_{getattr(seg, 'id')}",
                        getattr(seg, "speaker_id", None),
                        int(float(getattr(seg, "start", 0.0)) * 1000),
                        int(float(getattr(seg, "end", 0.0)) * 1000),
                    )
                    for seg in segment_list
                ],
                face_tracks=decoded.face_tracks,
                active_candidates=candidates,
            )
            metrics = dict(decoded.metrics)
            visual_characters = [
                character for character in graph.characters.values() if character.face_track_ids
            ]
            metrics.update(
                {
                    "characters_resolved": len(visual_characters),
                    "identity_graph_characters": len(graph.characters),
                    "track_embedding_quality": _track_embedding_quality(decoded.face_tracks),
                    "identity_merge_decisions": list(graph.merge_decisions[-48:]),
                    "character_clusters": [
                        {
                            "character_id": character.character_id,
                            "face_tracks": sorted(character.face_track_ids),
                            "speaker_ids": sorted(character.speaker_ids),
                            "identity_confidence": round(float(character.identity_confidence), 3),
                        }
                        for character in visual_characters
                    ],
                    "active_speaker_candidates": sum(len(v) for v in candidates.values()),
                    "dialogue_segments_with_visual_evidence": sum(
                        1 for r in resolutions.values() if r.visible_character_ids
                    ),
                    "offscreen_segments": sum(1 for r in resolutions.values() if r.offscreen_speech),
                    "analysis_duration_ms": int((time.monotonic() - started) * 1000),
                    "cache_hit": False,
                }
            )
            overlay_dir = self.config.debug_overlay_dir
            if overlay_dir:
                metrics["debug_overlay_frames"] = _write_debug_overlays(
                    path,
                    decoded.face_tracks,
                    graph.face_links,
                    Path(overlay_dir),
                    max_frames=self.config.debug_overlay_max_frames,
                )
            result = VisualAnalysisResult(
                "available",
                face_tracks=decoded.face_tracks,
                identity_graph=graph,
                segment_resolutions=resolutions,
                metrics=metrics,
                model_info=model_info,
            )
            _save_cache(cache_root, cache_key, fingerprint, self.config.signature(), result)
            return result
        except Exception as e:
            log.warn("visual analysis failed; continuing audio-only", error=str(e))
            return VisualAnalysisResult(
                "failed",
                metrics={"cache_hit": False, "analysis_duration_ms": int((time.monotonic() - started) * 1000)},
                model_info=model_info,
                error=str(e),
            )

    def _decode_and_track(
        self,
        path: Path,
        segments: list[Any],
        detector: FaceDetector,
        embedder: FaceEmbedder,
    ) -> "_DecodeResult":
        cv2 = _cv2()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int(total_frames / fps * 1000) if total_frames > 0 else 0
        windows = _dialogue_windows(segments, self.config.dialogue_padding_ms)
        tracker = _FaceTracker(self.config)
        frame_step = max(1, int(round(fps / max(0.1, self.config.scan_fps))))
        sampled = 0
        corrupt = 0
        detections_total = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step != 0:
                idx += 1
                continue
            timestamp_ms = int(round(idx * 1000.0 / fps))
            idx += 1
            if windows and not _in_windows(timestamp_ms, windows):
                continue
            if frame is None:
                corrupt += 1
                continue
            sampled += 1
            scaled, scale = _downscale(frame, self.config.max_analysis_width)
            detections = [
                det
                for det in detector.detect(scaled, timestamp_ms, scale)
                if det.confidence >= self.config.min_face_confidence
                and min(det.bbox.width, det.bbox.height) >= self.config.min_face_size_px
            ]
            detections = _suppress_duplicates(detections)
            enriched: list[_Detection] = []
            for det_index, det in enumerate(detections):
                emb = det.embedding
                if sampled % max(1, self.config.embedding_interval_frames) == 0 or emb is None:
                    emb = embedder.embed(scaled, det)
                enriched.append(_Detection(det.bbox, det.confidence, det.landmarks, emb))
            detections_total += len(enriched)
            tracker.update(scaled, timestamp_ms, enriched)
        cap.release()
        tracks = tracker.finish()
        metrics = {
            "frames_sampled": sampled,
            "corrupt_frames": corrupt,
            "faces_detected": detections_total,
            "face_tracks_created": len(tracks),
            "embeddings_generated": sum(1 for t in tracks if t.embedding is not None),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "duration_ms": duration_ms,
            "detector": getattr(detector, "name", "unknown"),
            "embedder": getattr(embedder, "name", "unknown"),
        }
        return _DecodeResult(tracks, metrics)


@dataclass
class _DecodeResult:
    face_tracks: list[FaceTrack]
    metrics: dict[str, Any]


class _FaceTracker:
    def __init__(self, config: VisualAnalysisConfig) -> None:
        self.config = config
        self.active: list[_TrackState] = []
        self.done: list[_TrackState] = []
        self.next_track = 1

    def update(self, frame: Any, timestamp_ms: int, detections: list[_Detection]) -> None:
        assignments: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.active):
            if timestamp_ms - track.last_ms > self.config.track_max_gap_ms:
                continue
            for di, det in enumerate(detections):
                score = _assignment_score(track, det, timestamp_ms)
                if score >= self.config.assignment_threshold:
                    assignments.append((score, ti, di))
        assignments.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _score, ti, di in assignments:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            self.active[ti].add(frame, timestamp_ms, detections[di])
        for di, det in enumerate(detections):
            if di not in used_dets:
                self.active.append(_TrackState.new(self.next_track, frame, timestamp_ms, det))
                self.next_track += 1
        still_active: list[_TrackState] = []
        for track in self.active:
            if timestamp_ms - track.last_ms > self.config.track_max_gap_ms:
                self.done.append(track)
            else:
                still_active.append(track)
        self.active = still_active

    def finish(self) -> list[FaceTrack]:
        states = self.done + self.active
        return [state.to_face_track() for state in states if state.observations]


@dataclass
class _TrackState:
    face_track_id: str
    start_ms: int
    last_ms: int
    observations: list[FaceObservation]
    embeddings: list[list[float]]
    last_bbox: BoundingBox
    last_patch: Any | None = None
    last_upper_patch: Any | None = None

    @classmethod
    def new(cls, index: int, frame: Any, timestamp_ms: int, det: _Detection) -> "_TrackState":
        state = cls(
            face_track_id=f"FACE_TRACK_{index:04d}",
            start_ms=timestamp_ms,
            last_ms=timestamp_ms,
            observations=[],
            embeddings=[],
            last_bbox=det.bbox,
        )
        state.add(frame, timestamp_ms, det)
        return state

    def add(self, frame: Any, timestamp_ms: int, det: _Detection) -> None:
        mouth_patch = _roi(frame, det.bbox, lower=True)
        upper_patch = _roi(frame, det.bbox, lower=False)
        mouth = _mouth_motion(self.last_patch, mouth_patch, self.last_upper_patch, upper_patch)
        visibility = _visibility(det.bbox, frame.shape[1], frame.shape[0])
        self.observations.append(
            FaceObservation(
                timestamp_ms=timestamp_ms,
                bbox=det.bbox,
                detection_confidence=det.confidence,
                visibility_score=visibility,
                mouth_activity_score=mouth,
            )
        )
        if det.embedding is not None:
            self.embeddings.append(det.embedding)
        self.last_bbox = det.bbox
        self.last_ms = timestamp_ms
        self.last_patch = mouth_patch
        self.last_upper_patch = upper_patch

    def to_face_track(self) -> FaceTrack:
        embedding = _mean_embedding(self.embeddings)
        confidence = _track_quality(self.observations, embedding)
        return FaceTrack(
            face_track_id=self.face_track_id,
            start_ms=self.start_ms,
            end_ms=self.last_ms,
            observations=self.observations,
            embedding=embedding,
            cluster_id=None,
            confidence=confidence,
        )


class _HaarDetector:
    name = "opencv-haar-frontalface"

    def __init__(self, cascade_path: str) -> None:
        cv2 = _cv2()
        self.model_path = cascade_path
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("OpenCV Haar face cascade is unavailable")

    def detect(self, frame: Any, timestamp_ms: int, scale: float) -> list[_Detection]:
        cv2 = _cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        return [
            _Detection(
                BoundingBox(float(x), float(y), float(w), float(h)),
                0.76,
            )
            for x, y, w, h in faces
        ]


class _YuNetDetector:
    name = "opencv-yunet"

    def __init__(self, model_path: Path, config: VisualAnalysisConfig) -> None:
        cv2 = _cv2()
        self.model_path = str(model_path)
        self._detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (320, 320),
            float(config.min_face_confidence),
            0.3,
            5000,
        )

    def detect(self, frame: Any, timestamp_ms: int, scale: float) -> list[_Detection]:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _retval, faces = self._detector.detect(frame)
        if faces is None:
            return []
        out: list[_Detection] = []
        for row in faces:
            x, y, bw, bh = [float(v) for v in row[:4]]
            landmarks = [(float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)]
            out.append(_Detection(BoundingBox(x, y, bw, bh), float(row[-1]), landmarks))
        return out


class _AppearanceEmbedder:
    name = "opencv-appearance-descriptor"
    model_path = None

    def embed(self, frame: Any, detection: _Detection) -> list[float] | None:
        cv2 = _cv2()
        crop = _crop(frame, detection.bbox)
        if crop is None:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        values = [float(v) / 255.0 for v in resized.flatten()]
        return _l2_normalize(values)


class _SFaceEmbedder:
    name = "opencv-sface"

    def __init__(self, model_path: Path) -> None:
        cv2 = _cv2()
        self.model_path = str(model_path)
        self._recognizer = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, frame: Any, detection: _Detection) -> list[float] | None:
        aligned = _crop(frame, detection.bbox)
        if aligned is None:
            return None
        feature = self._recognizer.feature(aligned)
        return _l2_normalize([float(v) for v in feature.flatten()])


def _make_opencv_detector(models_root: Path, config: VisualAnalysisConfig) -> FaceDetector:
    cv2 = _cv2()
    yunet = _find_model(models_root, ("face_detection_yunet", "yunet"), (".onnx",))
    if yunet is not None and hasattr(cv2, "FaceDetectorYN_create"):
        return _YuNetDetector(yunet, config)
    cascade = getattr(cv2, "data", None)
    cascade_path = None
    if cascade is not None:
        candidate = Path(cascade.haarcascades) / "haarcascade_frontalface_default.xml"
        if candidate.is_file():
            cascade_path = str(candidate)
    if cascade_path:
        return _HaarDetector(cascade_path)
    raise RuntimeError("visual face model unavailable: install YuNet ONNX under models/visual")


def _make_opencv_embedder(models_root: Path) -> FaceEmbedder:
    cv2 = _cv2()
    sface = _find_model(models_root, ("face_recognition_sface", "sface"), (".onnx",))
    if sface is not None and hasattr(cv2, "FaceRecognizerSF_create"):
        return _SFaceEmbedder(sface)
    return _AppearanceEmbedder()


def _active_candidates_for_segments(
    segments: list[Any],
    face_tracks: list[FaceTrack],
) -> dict[str, list[Any]]:
    resolver = ActiveSpeakerResolver()
    out: dict[str, list[Any]] = {}
    for seg in segments:
        speaker = getattr(seg, "speaker_id", None)
        if not speaker:
            continue
        start_ms = int(float(getattr(seg, "start", 0.0)) * 1000)
        end_ms = int(float(getattr(seg, "end", 0.0)) * 1000)
        visible = [track for track in face_tracks if _overlap(start_ms, end_ms, track.start_ms, track.end_ms) > 0]
        out[f"segment_{getattr(seg, 'id')}"] = resolver.rank_candidates(
            segment_id=f"segment_{getattr(seg, 'id')}",
            speaker_id=str(speaker),
            start_ms=start_ms,
            end_ms=end_ms,
            visible_face_tracks=visible,
            audio_confidence=float(getattr(seg, "speaker_confidence", 1.0) or 1.0),
        )
    return out


def _active_evidence_for_segments(segments: list[Any], face_tracks: list[FaceTrack]) -> list[Any]:
    return [
        candidates[0].evidence
        for candidates in _active_candidates_for_segments(segments, face_tracks).values()
        if candidates
    ]


def apply_visual_result_to_segments(segments: list[Any], result: VisualAnalysisResult) -> None:
    status_payload = {
        "status": result.status,
        "metrics": dict(result.metrics),
        "modelInfo": result.model_info.to_dict() if result.model_info else None,
        "error": result.error,
        "cacheHit": result.cache_hit,
    }
    graph_payload = (
        result.identity_graph.to_dict()
        if hasattr(result.identity_graph, "to_dict")
        else result.identity_graph
    )
    for seg in segments:
        sid = f"segment_{getattr(seg, 'id')}"
        ctx = dict(getattr(seg, "pronoun_context", {}) or {})
        resolution = result.segment_resolutions.get(sid)
        ctx["visualIdentityStatus"] = status_payload
        if resolution is not None:
            ctx["visualIdentity"] = (
                resolution.to_dict() if hasattr(resolution, "to_dict") else resolution
            )
        if graph_payload and not ctx.get("visualIdentityGraph"):
            ctx["visualIdentityGraph"] = _compact_graph(graph_payload)
        object.__setattr__(seg, "pronoun_context", ctx)


def integrate_visual_result_with_memory_store(memory: Any, result: VisualAnalysisResult) -> None:
    if memory.movie_memory is None or memory.movie_memory.character_context_store is None:
        return
    if result.identity_graph is None:
        memory.movie_memory.character_context_store.multimodal_extension_points["visual_identity"] = {
            "status": result.status,
            "error": result.error,
            "metrics": dict(result.metrics),
        }
        return
    integrate_identity_graph_with_context_store(
        memory.movie_memory.character_context_store,
        result.identity_graph,
    )


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError("opencv-python is not installed") from e
    return cv2


def _find_model(models_root: Path, stems: tuple[str, ...], suffixes: tuple[str, ...]) -> Path | None:
    roots = [models_root / "visual", models_root / "vision", models_root]
    for root in roots:
        for stem in stems:
            for suffix in suffixes:
                direct = root / f"{stem}{suffix}"
                if direct.is_file():
                    return direct
        if root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in suffixes and any(stem in path.stem.lower() for stem in stems):
                    return path
    return None


def _dialogue_windows(segments: list[Any], padding_ms: int) -> list[tuple[int, int]]:
    windows = [
        (
            max(0, int(float(getattr(seg, "start", 0.0)) * 1000) - padding_ms),
            int(float(getattr(seg, "end", 0.0)) * 1000) + padding_ms,
        )
        for seg in segments
    ]
    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1] + 250:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _in_windows(timestamp_ms: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= timestamp_ms <= end for start, end in windows)


def _downscale(frame: Any, max_width: int) -> tuple[Any, float]:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame, 1.0
    cv2 = _cv2()
    scale = max_width / float(frame.shape[1])
    resized = cv2.resize(frame, (max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _suppress_duplicates(detections: list[_Detection]) -> list[_Detection]:
    kept: list[_Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(_iou(det.bbox, prev.bbox) < 0.55 for prev in kept):
            kept.append(det)
    return kept


def _assignment_score(track: _TrackState, det: _Detection, timestamp_ms: int) -> float:
    iou = _iou(track.last_bbox, det.bbox)
    center = _center_similarity(track.last_bbox, det.bbox)
    emb = _cosine(_mean_embedding(track.embeddings), det.embedding) if det.embedding else 0.0
    gap = max(0.0, 1.0 - (timestamp_ms - track.last_ms) / 1500.0)
    return 0.42 * iou + 0.25 * center + 0.23 * emb + 0.10 * gap


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    inter_w = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0.0, min(ay2, by2) - max(a.y, b.y))
    inter = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _center_similarity(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay = a.x + a.width / 2, a.y + a.height / 2
    bx, by = b.x + b.width / 2, b.y + b.height / 2
    dist = math.hypot(ax - bx, ay - by)
    scale = max(a.width, a.height, b.width, b.height, 1.0)
    return max(0.0, 1.0 - dist / (scale * 2.0))


def _visibility(bbox: BoundingBox, width: int, height: int) -> float:
    size = min(1.0, math.sqrt(max(0.0, bbox.width * bbox.height)) / max(1.0, min(width, height) * 0.22))
    return max(0.0, min(1.0, size))


def _roi(frame: Any, bbox: BoundingBox, *, lower: bool) -> Any | None:
    cv2 = _cv2()
    y1 = bbox.y + bbox.height * (0.56 if lower else 0.12)
    y2 = bbox.y + bbox.height * (0.92 if lower else 0.45)
    x1 = bbox.x + bbox.width * 0.20
    x2 = bbox.x + bbox.width * 0.80
    crop = _crop_xy(frame, x1, y1, x2, y2)
    if crop is None:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA)


def _crop(frame: Any, bbox: BoundingBox) -> Any | None:
    return _crop_xy(frame, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)


def _crop_xy(frame: Any, x1: float, y1: float, x2: float, y2: float) -> Any | None:
    h, w = frame.shape[:2]
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    return frame[iy1:iy2, ix1:ix2].copy()


def _mouth_motion(prev: Any | None, cur: Any | None, prev_upper: Any | None, cur_upper: Any | None) -> float:
    if prev is None or cur is None:
        return 0.0
    mouth = _mean_absdiff(prev, cur)
    upper = _mean_absdiff(prev_upper, cur_upper) if prev_upper is not None and cur_upper is not None else 0.0
    localized = max(0.0, mouth - upper * 0.65)
    return max(0.0, min(1.0, localized / 35.0))


def _mean_absdiff(a: Any | None, b: Any | None) -> float:
    if a is None or b is None:
        return 0.0
    cv2 = _cv2()
    diff = cv2.absdiff(a, b)
    return float(diff.mean())


def _mean_embedding(items: list[list[float]]) -> list[float] | None:
    if not items:
        return None
    dims = min(len(item) for item in items)
    mean = [sum(item[i] for item in items) / len(items) for i in range(dims)]
    return _l2_normalize(mean)


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    return max(0.0, min(1.0, sum(left[i] * right[i] for i in range(n))))


def _track_quality(observations: list[FaceObservation], embedding: list[float] | None) -> float:
    if not observations:
        return 0.0
    det = sum(o.detection_confidence for o in observations) / len(observations)
    vis = sum(o.visibility_score for o in observations) / len(observations)
    length = min(1.0, len(observations) / 8.0)
    emb = 0.15 if embedding is not None else 0.0
    return max(0.0, min(0.98, 0.42 * det + 0.28 * vis + 0.15 * length + emb))


def _track_embedding_quality(tracks: list[FaceTrack]) -> dict[str, dict[str, Any]]:
    return {
        track.face_track_id: {
            "embedding_present": track.embedding is not None,
            "track_quality": round(float(track.confidence), 3),
            "observations": len(track.observations),
            "mean_detection_confidence": round(
                sum(obs.detection_confidence for obs in track.observations)
                / max(1, len(track.observations)),
                3,
            ),
            "mean_visibility": round(
                sum(obs.visibility_score for obs in track.observations)
                / max(1, len(track.observations)),
                3,
            ),
        }
        for track in tracks
    }


def _write_debug_overlays(
    video_path: Path,
    tracks: list[FaceTrack],
    face_links: dict[str, tuple[str, float]],
    output_dir: Path,
    *,
    max_frames: int,
) -> int:
    if max_frames <= 0:
        return 0
    observations_by_ts: dict[int, list[tuple[FaceTrack, FaceObservation]]] = {}
    for track in tracks:
        for observation in track.observations:
            observations_by_ts.setdefault(observation.timestamp_ms, []).append((track, observation))
    if not observations_by_ts:
        return 0

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    target_ts = set(sorted(observations_by_ts)[:max_frames])
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    idx = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(round(idx * 1000.0 / fps))
            idx += 1
            if timestamp_ms not in target_ts:
                continue
            for track, observation in observations_by_ts[timestamp_ms]:
                _draw_debug_observation(frame, track, observation, face_links)
            out = output_dir / f"visual_identity_{timestamp_ms:08d}ms.png"
            cv2.imwrite(str(out), frame)
            written += 1
    finally:
        cap.release()
    return written


def _draw_debug_observation(
    frame: Any,
    track: FaceTrack,
    observation: FaceObservation,
    face_links: dict[str, tuple[str, float]],
) -> None:
    cv2 = _cv2()
    bbox = observation.bbox
    x1, y1 = int(round(bbox.x)), int(round(bbox.y))
    x2, y2 = int(round(bbox.x + bbox.width)), int(round(bbox.y + bbox.height))
    character_id = face_links.get(track.face_track_id, ("UNRESOLVED", 0.0))[0]
    mouth = (
        f"{observation.mouth_activity_score:.2f}"
        if observation.mouth_activity_score is not None
        else "NA"
    )
    label = f"{track.face_track_id} {character_id} mouth={mouth}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (64, 220, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (64, 220, 255),
        1,
        cv2.LINE_AA,
    )


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(fingerprint: dict[str, Any], config: dict[str, Any], model: dict[str, Any]) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "config": config, "model": model},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "visual_identity" / f"{cache_key}.json"


def _load_cache(cache_root: Path, cache_key: str) -> VisualAnalysisResult | None:
    path = _cache_path(cache_root, cache_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tracks = [_face_track_from_cache(item) for item in payload.get("faceTracks") or []]
        result = VisualAnalysisResult(
            status="cached",
            face_tracks=tracks,
            identity_graph=payload.get("identityGraph"),
            segment_resolutions=payload.get("segmentResolutions") or {},
            metrics=payload.get("metrics") or {},
            model_info=VisualModelInfo(**payload["modelInfo"]) if payload.get("modelInfo") else None,
            error=payload.get("error"),
            cache_hit=True,
        )
        return result
    except Exception as e:
        log.warn("visual cache ignored", path=str(path), error=str(e))
        return None


def _save_cache(
    cache_root: Path,
    cache_key: str,
    fingerprint: dict[str, Any],
    config: dict[str, Any],
    result: VisualAnalysisResult,
) -> None:
    path = _cache_path(cache_root, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_payload()
    payload["analysisVersion"] = VISUAL_ANALYSIS_VERSION
    payload["sourceFingerprint"] = fingerprint
    payload["config"] = config
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _face_track_to_cache(track: FaceTrack) -> dict[str, Any]:
    payload = track.to_dict()
    payload["embedding"] = track.embedding
    return payload


def _face_track_from_cache(payload: dict[str, Any]) -> FaceTrack:
    observations = [
        FaceObservation(
            timestamp_ms=int(item.get("timestamp_ms", 0)),
            bbox=BoundingBox(**item.get("bbox", {})),
            detection_confidence=float(item.get("detection_confidence", 0.0)),
            visibility_score=float(item.get("visibility_score", 0.0)),
            mouth_activity_score=(
                float(item["mouth_activity_score"])
                if item.get("mouth_activity_score") is not None
                else None
            ),
        )
        for item in payload.get("observations") or []
    ]
    return FaceTrack(
        face_track_id=str(payload.get("face_track_id")),
        start_ms=int(payload.get("start_ms", 0)),
        end_ms=int(payload.get("end_ms", 0)),
        observations=observations,
        embedding=payload.get("embedding"),
        cluster_id=payload.get("cluster_id"),
        confidence=float(payload.get("confidence", 0.0)),
    )


def _compact_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_scope_id": graph.get("video_scope_id"),
        "speaker_links": graph.get("speaker_links") or {},
        "face_links": graph.get("face_links") or {},
        "conflicts": list(graph.get("conflicts") or [])[-8:],
    }
