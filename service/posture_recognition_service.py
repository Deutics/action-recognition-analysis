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
from typing import Optional, List, Tuple

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

        # Backend selection
        self.backend = self._select_backend(model_path)

        # Core components
        self.keypoint_extractor = KeypointExtractor(self.config)
        self.classifier = PostureClassifier(self.config)
        self.frame_annotator = FrameAnnotator()
        self.person_tracker = PersonTracker(lying_threshold=self.config.lying_duration_threshold)
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(video_source)

        # State
        self.frame_index = 0
        self.last_persons = []

        logger.info(f"Service initialized for source: {source_id}")
        if self.normalized_zone_vertices:
            logger.info(f"Zone enabled with {len(self.normalized_zone_vertices)} vertices (normalized).")
        else:
            logger.warning("Zone NOT configured; notifications will trigger anywhere in frame.")

    # ==========================================================
    # Backend selection
    # ==========================================================

    def _is_jetson(self) -> bool:
        return os.path.exists("/etc/nv_tegra_release")

    def _select_backend(self, model_path: str):
        # Mac -> CPU torch
        if platform.system() == "Darwin":
            self._backend_name = "torch"
            logger.info("Using Torch backend (Mac)")
            return TorchBackend(model_path)

        # Jetson -> prefer TRT
        if self._is_jetson() and TRT_AVAILABLE:
            self._backend_name = "tensorrt"
            engine_path = "yolo26n-pose_fp16.engine"
            if os.path.exists(engine_path):
                logger.info("Using TensorRT backend (Jetson)")
                return TensorRTBackend(engine_path)
            logger.warning("TensorRT engine not found, falling back to Torch CPU")

        self._backend_name = "torch"
        logger.info("Using Torch backend (fallback)")
        return TorchBackend(model_path)

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
        # pointPolygonTest expects contour and (x,y)
        return cv2.pointPolygonTest(poly_px, (float(cx), float(cy)), False) >= 0

    def _bbox_center_from_keypoints(self, kpts: dict) -> Optional[Tuple[int, int]]:
        """
        Uses available extracted keypoints to estimate bbox center.
        Works on both backends because it relies on your KeypointExtractor output.
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

    # ==========================================================
    # Main loop
    # ==========================================================

    async def run(self):
        try:
            logger.info(f"Starting posture detection on: {self.video_source}")
            self.stream_handler.start_stream()
            await self.notification_handler.start()

            while True:
                frame, motion_detected = await asyncio.to_thread(self.stream_handler.read_frame)
                if frame is None:
                    logger.warning("Stream ended")
                    break

                frame = frame.copy()

                run_inference = (
                    motion_detected
                    and (self.frame_index % self.infer_every_n_frames == 0)
                )

                # for load testing on jetson
                # run_inference = True

                if run_inference:
                    self.last_persons = self._infer_and_classify(frame)

                # Falling tracking
                active_ids = set()
                falling_ids = set()
                for p in self.last_persons:
                    active_ids.add(p["id"])
                    if self.person_tracker.update(p["id"], p["label"]):
                        falling_ids.add(p["id"])
                self.person_tracker.cleanup(active_ids)

                # Zone polygon (pixel)
                h, w = frame.shape[:2]
                zone_poly = self._zone_polygon_px(h, w)

                # Optional: draw zone in UI
                if ENABLE_UI == "1" and zone_poly is not None:
                    cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 0, 255), thickness=1)

                # Annotation + notification with zone filter
                for p in self.last_persons:
                    label = p["label"]
                    if p["id"] in falling_ids:
                        label = "Falling"

                    frame = self.frame_annotator.annotate_frame(
                        frame,
                        posture_label=label,
                        confidence=p["confidence"],
                        keypoints=p["keypoints"],
                        person_id=p["id"],
                        region_of_interest=zone_poly
                    )

                    # Center point from keypoints
                    center = self._bbox_center_from_keypoints(p["keypoints"])
                    if center and ENABLE_UI == "1":
                        cv2.circle(frame, center, 4, (0, 255, 255), -1)

                    # Only notify if falling + inside zone
                    if p["id"] in falling_ids:
                        if center is None:
                            # No reliable center => be conservative (skip)
                            logger.debug(f"[{self.source_id}] Skip alert: no center for id={p['id']}")
                        else:
                            cx, cy = center
                            if self._point_in_zone(cx, cy, zone_poly):
                                frame = self.frame_annotator.draw_region_of_interest(frame, zone_poly)
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
                    cv2.imshow(f"Posture Detection - {self.source_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                await asyncio.sleep(0)
                self.frame_index = (self.frame_index + 1) % 1_000_000

        except Exception as e:
            logger.error(f"Fatal error in service: {str(e)}", exc_info=True)
        finally:
            await self._cleanup()

    # ==========================================================
    # Inference + Classification
    # ==========================================================

    def _infer_and_classify(self, frame: np.ndarray):
        persons = []
        try:
            keypoints_data, track_ids = self.backend.infer(frame)

            if keypoints_data is None or len(keypoints_data) == 0:
                return persons

            for idx, person_keypoints in enumerate(keypoints_data):
                person_id = track_ids[idx] if track_ids and idx < len(track_ids) else idx

                kpt_coords = person_keypoints[:, :2]
                kpt_conf = person_keypoints[:, 2].tolist()

                kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)

                self.classifier.set_frame_dimensions(frame.shape[0], frame.shape[1])
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
    # Cleanup
    # ==========================================================

    async def _cleanup(self):
        logger.info("Releasing resources...")
        await self.notification_handler.stop()
        self.stream_handler.release_stream()
        if ENABLE_UI == "1":
            cv2.destroyAllWindows()