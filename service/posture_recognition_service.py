"""
Posture Recognition Service - LOW LATENCY VERSION
Zero-buffer, latest-frame-only processing
"""

import cv2
from typing import Optional
from ultralytics import YOLO
import numpy as np
import torch
import time

from config.posture_config import PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from tracking.person_tracker import PersonTracker
from notification.notification_handler import NotificationHandler
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseRecognition:
    """
    Low-latency posture recognition service.
    Designed for real-time display with minimum delay.
    """

    def __init__(self,
                 video_source: str,
                 source_id: str,
                 model_path: str = "yolo11n-pose.pt",
                 config: Optional[PostureConfig] = None,
                 confidence_threshold: float = 0.5,
                 infer_every_n_frames: int = 1):

        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        self.infer_every_n_frames = max(1, infer_every_n_frames)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.config = config or PostureConfig()

        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

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

        logger.info(f"Low-latency service initialized for source: {source_id}")

    # ------------------------------------------------------------------ #

    def run(self):
        """Main low-latency loop"""

        try:
            logger.info(f"Starting posture detection on source: {self.video_source}")
            self.stream_handler.start_stream()

            cap = self.stream_handler.capture
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Stream opened - FPS: {fps}, Resolution: {width}x{height}")

            while True:
                frame, motion_detected = self.stream_handler.read_frame()
                if frame is None:
                    logger.warning("End of stream reached")
                    break

                # ---------------- INFERENCE CONTROL ---------------- #
                run_inference = (
                    motion_detected and
                    (self.frame_index % self.infer_every_n_frames == 0)
                )

                if run_inference:
                    self.last_persons = self._infer_and_classify(frame)

                # ---------------- FALLING TRACKING ---------------- #
                active_ids = set()
                falling_ids = set()

                for p in self.last_persons:
                    active_ids.add(p["id"])
                    if self.person_tracker.update(p["id"], p["label"]):
                        falling_ids.add(p["id"])

                self.person_tracker.cleanup(active_ids)

                # ---------------- ANNOTATION ---------------- #
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
                        self.notification_handler.save_falling_notification(
                            person_id=p["id"],
                            frame=frame.copy(),
                            source_id=self.source_id
                        )

                cv2.imshow(f"Posture Detection - Source {self.source_id}", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break

                self.frame_index = (self.frame_index + 1) % 1_000_000

        except Exception as e:
            logger.error(f"Fatal error in service: {str(e)}", exc_info=True)

        finally:
            self._cleanup()

    # ------------------------------------------------------------------ #

    def _infer_and_classify(self, frame: np.ndarray):
        """Run YOLO pose + posture classification on a single frame"""

        persons = []

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

        frame_h = frame.shape[0]

        for idx, person_keypoints in enumerate(keypoints_data):
            person_id = track_ids[idx] if idx < len(track_ids) else idx

            kpt_coords = person_keypoints[:, :2]
            kpt_conf = person_keypoints[:, 2].tolist()

            kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
            label, confidence = self.classifier.classify(kpts, frame_h)

            persons.append({
                "id": person_id,
                "label": label,
                "confidence": confidence,
                "keypoints": kpts
            })

        return persons

    # ------------------------------------------------------------------ #

    def _cleanup(self):
        logger.info("Releasing resources...")
        self.stream_handler.release_stream()
        cv2.destroyAllWindows()
