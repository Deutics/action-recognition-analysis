"""
Posture Recognition Service
Cross-platform:
- Mac -> Torch CPU
- Jetson -> TensorRT (if available) else Torch CPU
"""

import cv2
import numpy as np
import asyncio
import time
import platform
import os

from typing import Optional

from config.constants import ENABLE_UI
from config.posture_config import PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from tracking.person_tracker import PersonTracker
from notification.notification_handler import NotificationHandler
from utils.logger import get_logger

# Backends
from inference.torch_backend import TorchBackend

try:
    from inference.tensorrt_backend import TensorRTBackend
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


logger = get_logger(__name__)


class PoseRecognition:

    def __init__(self,
                 video_source: str,
                 source_id: str,
                 model_path: str = "yolo26n-pose.pt",
                 config: Optional[PostureConfig] = None,
                 confidence_threshold: float = 0.5,
                 infer_every_n_frames: int = 1):

        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        self.infer_every_n_frames = max(1, infer_every_n_frames)

        self.config = config or PostureConfig()

        # --------------------------------------------------
        # Backend Selection
        # --------------------------------------------------

        self.backend = self._select_backend(model_path)

        # Core components (UNCHANGED)
        self.keypoint_extractor = KeypointExtractor(self.config)
        self.classifier = PostureClassifier(self.config)
        self.frame_annotator = FrameAnnotator()
        self.person_tracker = PersonTracker(
            lying_threshold=self.config.lying_duration_threshold
        )
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(video_source)

        # State
        self.frame_index = 0
        self.last_persons = []

        logger.info(f"Service initialized for source: {source_id}")

    # ==========================================================
    # Backend selection
    # ==========================================================

    def _is_jetson(self):
        return os.path.exists("/etc/nv_tegra_release")

    def _select_backend(self, model_path):

        # Mac -> always CPU Torch
        if platform.system() == "Darwin":
            self._backend_name = "torch"
            logger.info("Using Torch backend (Mac)")
            return TorchBackend(model_path)

        # Jetson -> prefer TensorRT
        if self._is_jetson() and TRT_AVAILABLE:
            self._backend_name = "tensorrt"
            engine_path = "yolo26n-pose_fp16.engine"
            if os.path.exists(engine_path):
                logger.info("Using TensorRT backend (Jetson)")
                return TensorRTBackend(engine_path)
            else:
                logger.warning("TensorRT engine not found, falling back to Torch CPU")

        # Fallback
        logger.info("Using Torch CPU backend (fallback)")
        return TorchBackend(model_path)

    # ==========================================================
    # Main loop
    # ==========================================================

    async def run(self):

        try:
            logger.info(f"Starting posture detection on: {self.video_source}")
            self.stream_handler.start_stream()

            await self.notification_handler.start()

            while True:

                frame, motion_detected = await asyncio.to_thread(
                    self.stream_handler.read_frame
                )

                if frame is None:
                    logger.warning("Stream ended")
                    break

                # IMPORTANT: GStreamer frames can be readonly
                frame = frame.copy()

                run_inference = (
                    motion_detected and
                    (self.frame_index % self.infer_every_n_frames == 0)
                )

                if run_inference:
                    # self.last_persons = await asyncio.to_thread(
                    #     self._infer_and_classify, frame
                    # )
                    self.last_persons = self._infer_and_classify(frame)

                # -------------------------
                # Falling tracking
                # -------------------------
                active_ids = set()
                falling_ids = set()

                for p in self.last_persons:
                    active_ids.add(p["id"])
                    if self.person_tracker.update(p["id"], p["label"]):
                        falling_ids.add(p["id"])

                self.person_tracker.cleanup(active_ids)

                # -------------------------
                # Annotation
                # -------------------------
                for p in self.last_persons:

                    label = p["label"]
                    if p["id"] in falling_ids:
                        label = "Falling"

                    frame = self.frame_annotator.annotate_frame(
                        frame,
                        posture_label=label,
                        confidence=p["confidence"],
                        keypoints=p["keypoints"],
                        person_id=p["id"]
                    )

                    if p["id"] in falling_ids:
                        await self.notification_handler.notify_fall(
                            person_id=p["id"],
                            frame=frame.copy(),
                            source_id=self.source_id,
                            confidence=p["confidence"]
                        )

                if self.frame_index % 100 == 0:
                    logger.info(f"Frame count: {self.frame_index}")

                # Generating notifications for testing on jetson
                if (self.frame_index + 1) % 1000 == 0:
                    await self.notification_handler.notify_fall(
                        person_id="123456",
                        frame=frame.copy(),
                        source_id=self.source_id,
                        confidence=100
                    )

                if ENABLE_UI == "1":
                    cv2.imshow(f"Posture Detection - {self.source_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
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
            # ----------------------------------------
            # 1) TensorRT Backend (Jetson)
            # ----------------------------------------
            if self._backend_name == "tensorrt":

                # Send RAW frame only (H,W,3)
                keypoints_data, track_ids = self.backend.infer(frame)

                if keypoints_data is None or len(keypoints_data) == 0:
                    return persons

            # ----------------------------------------
            # 2) PyTorch Backend (Mac / CPU)
            # ----------------------------------------
            else:
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    persist=True,
                    verbose=False
                )

                if not results:
                    return persons

                result = results[0]
                if result.keypoints is None:
                    return persons

                keypoints_data = result.keypoints.data.cpu().numpy()

                track_ids = []
                if result.boxes is not None and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

            # ----------------------------------------
            # 3) Post-processing (common)
            # ----------------------------------------
            for idx, person_keypoints in enumerate(keypoints_data):
                person_id = (
                    track_ids[idx] if track_ids is not None and idx < len(track_ids)
                    else idx
                )

                kpt_coords = person_keypoints[:, :2]
                kpt_conf = person_keypoints[:, 2].tolist()

                kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)

                self.classifier.set_frame_dimensions(frame.shape[0], frame.shape[1])
                label, confidence = self.classifier.classify(
                    kpts,
                    frame.shape[0]
                )

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