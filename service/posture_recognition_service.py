"""
Posture Recognition Service (Zone-aware)
Cross-platform:
- Mac -> Torch CPU
- Jetson -> TensorRT (if available) else Torch CPU

Zone filtering:
- Only triggers fall notification if the person center lies within configured zone polygon
- Zone is provided as normalized vertices [(x,y), ...] in range 0..1
"""

import cv2
import numpy as np
import asyncio
import platform
import os
import time
from typing import Optional, List, Tuple, Dict, Any

from config.constants import ENABLE_UI
from config.posture_config import PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from tracking.person_tracker import PersonTracker
from notification.notification_handler import NotificationHandler
from utils.logger import get_logger

from inference.torch_backend import TorchBackend
try:
    from inference.human_detector import HumanDetector
except Exception:
    HumanDetector = None

try:
    from inference.tensorrt_backend import TensorRTBackend
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


logger = get_logger(__name__)


class PoseRecognition:
    def __init__(
        self,
        video_source: str,
        source_id: str,
        model_path: str = "yolo26n-pose.pt",
        config: Optional[PostureConfig] = None,
        confidence_threshold: float = 0.5,
        infer_every_n_frames: int = 1,
        normalized_zone_vertices: Optional[List[Tuple[float, float]]] = None,
        backend=None,
        backend_name: Optional[str] = None,
        use_backend_track_ids: bool = True,
        reconnect_interval_sec: float = 5.0,
        person_detector: Optional[HumanDetector] = None,
        person_box_overlap_threshold: float = 0.6,
    ):
        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        self.infer_every_n_frames = max(1, infer_every_n_frames)
        self.config = config or PostureConfig()

        # Zone (normalized 0..1)
        self.normalized_zone_vertices = normalized_zone_vertices  # can be None
        self._zone_poly_px = None  # cached per resolution (np.int32 Nx1x2)
        self._zone_last_shape = None  # (h,w)

        # Backend selection or injection
        if backend is None:
            self.backend, self._backend_name = self.create_backend(model_path=model_path, use_tracking=True)
        else:
            self.backend = backend
            self._backend_name = backend_name or type(backend).__name__

        # In shared backend mode, backend-level track IDs are unsafe across streams
        self.use_backend_track_ids = use_backend_track_ids
        self.reconnect_interval_sec = max(1.0, float(reconnect_interval_sec))
        self.person_detector = person_detector
        self.person_box_overlap_threshold = max(0.1, min(1.0, float(person_box_overlap_threshold)))

        # Core components (stream-local)
        self.keypoint_extractor = KeypointExtractor(self.config)
        self.classifier = PostureClassifier(self.config)
        self.frame_annotator = FrameAnnotator()
        self.person_tracker = PersonTracker(lying_threshold=self.config.lying_duration_threshold)
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(video_source)

        # State
        self.frame_index = 0
        self.last_persons = []
        self._running = False
        self._stream_connected = False
        self._last_reconnect_attempt = 0.0
        self._reader_task: Optional[asyncio.Task] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_motion_detected: bool = False
        self._latest_frame_seq: int = 0
        self._last_processed_frame_seq: int = 0

        # Stream-local lightweight ID assignment when backend IDs are unavailable/disabled
        self._next_local_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._track_max_distance_px = 80.0
        self._track_stale_frames = 30
        self._idle_sleep_sec = 0.003
        self._fps_value: float = 0.0
        self._fps_frames: int = 0
        self._fps_window_start: float = time.perf_counter()
        self._last_fps_log_ts: float = 0.0

        logger.info(f"Service initialized for source: {source_id}")
        if self.normalized_zone_vertices:
            logger.info(f"Zone enabled with {len(self.normalized_zone_vertices)} vertices (normalized).")
        else:
            logger.warning("Zone NOT configured; notifications will trigger anywhere in frame.")

    # ==========================================================
    # Backend selection
    # ==========================================================

    @staticmethod
    def _is_jetson() -> bool:
        return os.path.exists("/etc/nv_tegra_release")

    @classmethod
    def create_backend(cls, model_path: str, use_tracking: bool = True):
        # Mac -> CPU torch
        if platform.system() == "Darwin":
            logger.info("Using Torch backend (Mac)")
            return TorchBackend(model_path, use_tracking=use_tracking), "torch"

        # Jetson -> prefer TRT
        if cls._is_jetson() and TRT_AVAILABLE:
            engine_path = "yolo26n-pose_fp16.engine"
            if os.path.exists(engine_path):
                logger.info("Using TensorRT backend (Jetson)")
                return TensorRTBackend(engine_path), "tensorrt"
            logger.warning("TensorRT engine not found, falling back to Torch CPU")

        logger.info("Using Torch backend (fallback)")
        return TorchBackend(model_path, use_tracking=use_tracking), "torch"

    @classmethod
    def create_human_detector(cls, preferred_model: str = "yolo26m.pt"):
        if HumanDetector is None:
            raise RuntimeError("Human detector module not available (inference/human_detector.py missing)")
        candidate_models = []

        if cls._is_jetson():
            candidate_models.extend([
                "yolo26m_fp16.engine",
                "yolo26m.engine",
            ])

        candidate_models.extend([
            preferred_model,
            "yolo26m.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo26m-pose.pt",  # last-resort fallback if detect model is unavailable
        ])

        seen = set()
        ordered_candidates = []
        for m in candidate_models:
            if m not in seen:
                seen.add(m)
                ordered_candidates.append(m)

        for model_path in ordered_candidates:
            if os.path.exists(model_path):
                logger.info(f"Using shared human detector model: {model_path}")
                return HumanDetector(model_path=model_path)

        raise FileNotFoundError(
            "No human detector model found. Tried: " + ", ".join(ordered_candidates)
        )

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def start(self):
        if self._running:
            return

        logger.info(f"Starting posture detection on: {self.video_source}")
        await self.notification_handler.start()
        self._running = True
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        await self._cleanup()

    async def process_once(self) -> bool:
        """
        Process one frame for this stream.
        Returns False when stream ends (caller should stop this service).
        """
        if not self._running:
            await self.start()

        if self._latest_frame_seq == self._last_processed_frame_seq:
            await asyncio.sleep(self._idle_sleep_sec)
            return True

        frame = self._latest_frame
        motion_detected = self._latest_motion_detected
        current_seq = self._latest_frame_seq
        if frame is None:
            await asyncio.sleep(0)
            return True

        frame = frame.copy()

        run_inference = (
            motion_detected
            and (self.frame_index % self.infer_every_n_frames == 0)
        )

        if run_inference:
            self.last_persons = self._infer_and_classify(frame)

        should_continue = await self._postprocess_and_notify(frame)
        self._last_processed_frame_seq = current_seq
        self._update_fps()

        await asyncio.sleep(0)
        self.frame_index = (self.frame_index + 1) % 1_000_000

        return should_continue

    async def run(self):
        try:
            await self.start()
            while True:
                keep_running = await self.process_once()
                if not keep_running:
                    break
        except Exception as e:
            logger.error(f"Fatal error in service: {str(e)}", exc_info=True)
        finally:
            await self.stop()

    async def _try_connect_stream(self):
        self._last_reconnect_attempt = time.time()
        try:
            await asyncio.to_thread(self.stream_handler.start_stream)
            self._stream_connected = True
            logger.info(f"[{self.source_id}] Stream connected")
        except Exception as e:
            self._stream_connected = False
            logger.warning(
                f"[{self.source_id}] Reconnect failed, retrying in {self.reconnect_interval_sec:.1f}s: {e}"
            )

    async def _reader_loop(self):
        try:
            while self._running:
                if not self._stream_connected:
                    now = time.time()
                    if (now - self._last_reconnect_attempt) >= self.reconnect_interval_sec:
                        await self._try_connect_stream()
                    await asyncio.sleep(0.10)
                    continue

                frame, motion_detected = await asyncio.to_thread(self.stream_handler.read_frame)
                if frame is None:
                    logger.warning(f"[{self.source_id}] Stream disconnected, will retry reconnect")
                    self.stream_handler.release_stream()
                    self._stream_connected = False
                    self.last_persons = []
                    await asyncio.sleep(0.10)
                    continue

                # Keep only latest frame (drop older frames under load).
                self._latest_frame = frame
                self._latest_motion_detected = motion_detected
                self._latest_frame_seq += 1
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

    # ==========================================================
    # Zone helpers
    # ==========================================================

    def _zone_polygon_px(self, frame_h: int, frame_w: int):
        """
        Returns polygon in pixel coords shaped (N,1,2) int32, cached per resolution.
        """
        if not self.normalized_zone_vertices:
            return None

        if self._zone_last_shape == (frame_h, frame_w) and self._zone_poly_px is not None:
            return self._zone_poly_px

        pts = []
        for (nx, ny) in self.normalized_zone_vertices:
            x = int(round(nx * frame_w))
            y = int(round(ny * frame_h))
            pts.append([x, y])

        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        self._zone_poly_px = poly
        self._zone_last_shape = (frame_h, frame_w)
        return poly

    def _point_in_zone(self, cx: int, cy: int, poly_px) -> bool:
        """
        True if point lies inside or on polygon edge.
        """
        if poly_px is None:
            return True  # zone not configured => allow
        return cv2.pointPolygonTest(poly_px, (float(cx), float(cy)), False) >= 0

    def _bbox_center_from_keypoints(self, kpts: dict) -> Optional[Tuple[int, int]]:
        """
        Uses available extracted keypoints to estimate bbox center.
        Works on both backends because it relies on KeypointExtractor output.
        """
        coords = []
        for v in kpts.values():
            if isinstance(v, tuple) and len(v) == 2 and v[0] is not None and v[1] is not None:
                coords.append(v)

        if not coords:
            return None

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def _valid_pose_points(self, kpts: Dict[str, Tuple[float, float]]) -> List[Tuple[int, int]]:
        pts = []
        for p in kpts.values():
            if isinstance(p, tuple) and len(p) == 2 and p[0] is not None and p[1] is not None:
                x, y = int(p[0]), int(p[1])
                if x > 0 and y > 0:
                    pts.append((x, y))
        return pts

    def _max_pose_overlap_with_person_boxes(
        self,
        pose_kpts: Dict[str, Tuple[float, float]],
        person_boxes: List[Tuple[int, int, int, int, float]],
    ) -> float:
        pts = self._valid_pose_points(pose_kpts)
        if not pts or not person_boxes:
            return 0.0

        best_ratio = 0.0
        for (x1, y1, x2, y2, _conf) in person_boxes:
            inside = 0
            for x, y in pts:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside += 1
            ratio = inside / max(1, len(pts))
            if ratio > best_ratio:
                best_ratio = ratio
        return best_ratio

    def _update_fps(self):
        now = time.perf_counter()
        self._fps_frames += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps_value = self._fps_frames / max(elapsed, 1e-6)
            self._fps_frames = 0
            self._fps_window_start = now

            if (now - self._last_fps_log_ts) >= 5.0:
                logger.info(f"[{self.source_id}] FPS={self._fps_value:.2f}")
                self._last_fps_log_ts = now

    # ==========================================================
    # Inference + Classification
    # ==========================================================

    def _unpack_infer_output(self, infer_out):
        if infer_out is None:
            return None, None
        if isinstance(infer_out, tuple):
            if len(infer_out) >= 2:
                return infer_out[0], infer_out[1]
            return None, None
        return None, None

    def _center_from_raw_keypoints(self, person_keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        if person_keypoints is None or len(person_keypoints) == 0:
            return None

        xs = []
        ys = []
        for row in person_keypoints:
            x = float(row[0])
            y = float(row[1])
            conf = float(row[2]) if len(row) > 2 else 1.0
            if x > 0 and y > 0 and conf >= self.config.min_keypoint_confidence:
                xs.append(x)
                ys.append(y)

        if not xs:
            return None

        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    def _assign_local_track_ids(self, keypoints_data: np.ndarray) -> List[int]:
        assigned_ids = []
        used_track_ids = set()

        for person_keypoints in keypoints_data:
            center = self._center_from_raw_keypoints(person_keypoints)
            if center is None:
                person_id = self._next_local_id
                self._next_local_id += 1
                self._tracks[person_id] = {"center": None, "last_seen": self.frame_index}
                assigned_ids.append(person_id)
                continue

            best_track_id = None
            best_distance = float("inf")

            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                track_center = track.get("center")
                if track_center is None:
                    continue

                distance = float(np.hypot(center[0] - track_center[0], center[1] - track_center[1]))
                if distance < best_distance and distance <= self._track_max_distance_px:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                person_id = self._next_local_id
                self._next_local_id += 1
            else:
                person_id = best_track_id

            self._tracks[person_id] = {"center": center, "last_seen": self.frame_index}
            used_track_ids.add(person_id)
            assigned_ids.append(person_id)

        min_last_seen = self.frame_index - self._track_stale_frames
        self._tracks = {
            tid: trk for tid, trk in self._tracks.items()
            if trk.get("last_seen", -1) >= min_last_seen
        }

        return assigned_ids

    def _resolve_person_ids(self, keypoints_data: np.ndarray, track_ids) -> List[int]:
        if (
            self.use_backend_track_ids
            and track_ids
            and isinstance(track_ids, list)
            and len(track_ids) == len(keypoints_data)
        ):
            return [int(x) for x in track_ids]

        return self._assign_local_track_ids(keypoints_data)

    def _infer_and_classify(self, frame: np.ndarray):
        persons = []
        try:
            infer_out = self.backend.infer(frame)
            keypoints_data, track_ids = self._unpack_infer_output(infer_out)

            if keypoints_data is None or len(keypoints_data) == 0:
                return persons

            person_ids = self._resolve_person_ids(keypoints_data, track_ids)

            self.classifier.set_frame_dimensions(frame.shape[0], frame.shape[1])

            for idx, person_keypoints in enumerate(keypoints_data):
                person_id = person_ids[idx] if idx < len(person_ids) else idx

                kpt_coords = person_keypoints[:, :2]
                kpt_conf = person_keypoints[:, 2].tolist()

                kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
                label, confidence = self.classifier.classify(kpts, frame.shape[0])

                persons.append({
                    "id": person_id,
                    "label": label,
                    "confidence": float(confidence),
                    "keypoints": kpts
                })

            return persons

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return []

    # ==========================================================
    # Per-frame postprocess
    # ==========================================================

    async def _postprocess_and_notify(self, frame: np.ndarray) -> bool:
        active_ids = set()
        falling_ids = set()
        for p in self.last_persons:
            active_ids.add(p["id"])
            if self.person_tracker.update(p["id"], p["label"]):
                falling_ids.add(p["id"])
        self.person_tracker.cleanup(active_ids)

        h, w = frame.shape[:2]
        zone_poly = self._zone_polygon_px(h, w)

        if ENABLE_UI == "1" and zone_poly is not None:
            cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 0, 255), thickness=1)

        person_boxes = []
        if falling_ids:
            if self.person_detector is not None:
                try:
                    person_boxes = self.person_detector.detect_person_boxes(frame)
                except Exception as e:
                    logger.error(f"[{self.source_id}] Human detector failed: {e}", exc_info=True)
                    person_boxes = []
            else:
                logger.warning(f"[{self.source_id}] Human detector not configured; skipping fall alerts")

        for p in self.last_persons:
            label = p["label"]
            if (p["id"] in falling_ids) or (p["label"] == "Lying"):
                label = "Falling"

            frame = self.frame_annotator.annotate_frame(
                frame,
                posture_label=label,
                confidence=p["confidence"],
                keypoints=p["keypoints"],
                person_id=p["id"],
                region_of_interest=zone_poly
            )

            center = self._bbox_center_from_keypoints(p["keypoints"])
            if center and ENABLE_UI == "1":
                cv2.circle(frame, center, 4, (0, 255, 255), -1)

            if p["id"] in falling_ids:
                if center is None:
                    logger.debug(f"[{self.source_id}] Skip alert: no center for id={p['id']}")
                else:
                    cx, cy = center
                    if self._point_in_zone(cx, cy, zone_poly):
                        overlap = self._max_pose_overlap_with_person_boxes(p["keypoints"], person_boxes)
                        if overlap < self.person_box_overlap_threshold:
                            logger.info(
                                f"[{self.source_id}] Alert blocked by human-box gate: "
                                f"id={p['id']} overlap={overlap:.2f} threshold={self.person_box_overlap_threshold:.2f}"
                            )
                            continue

                        await self.notification_handler.notify_fall(
                            person_id=p["id"],
                            frame=frame.copy(),
                            source_id=self.source_id,
                            confidence=p["confidence"]
                        )
                    else:
                        logger.debug(
                            f"[{self.source_id}] Falling outside zone: id={p['id']} center=({cx},{cy})"
                        )

        if ENABLE_UI == "1":
            cv2.putText(
                frame,
                f"FPS: {self._fps_value:.2f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"Posture Detection - {self.source_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

        return True

    # ==========================================================
    # Cleanup
    # ==========================================================

    async def _cleanup(self):
        logger.info(f"[{self.source_id}] Releasing resources...")
        await self.notification_handler.stop()
        self.stream_handler.release_stream()
        if ENABLE_UI == "1":
            cv2.destroyAllWindows()
