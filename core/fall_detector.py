"""Sequential fall detection processor with GPU/CPU validation"""

import cv2
import time
import numpy as np
import torch
from typing import Dict
from ultralytics import YOLO
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from tracking.person_tracker import PersonTracker
from notification.notification_handler import NotificationHandler
from utils.frame_annotator import FrameAnnotator
from utils.logger import get_logger

logger = get_logger(__name__)


class FallDetectionProcessor:
    def __init__(self, model_path: str = "yolo11n-pose.pt", lying_threshold: float = 3.0, confidence_threshold: float = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Processor using device: {self.device}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.keypoint_extractor = KeypointExtractor()
        self.classifier = PostureClassifier()
        self.frame_annotator = FrameAnnotator()
        self.notification_handler = NotificationHandler()
        self.confidence_threshold = confidence_threshold
        self.trackers = {}
        self.lying_threshold = lying_threshold
        # self.last_persons = {}
        logger.info(f"FallDetectionProcessor initialized with model={model_path}")
        
    def _get_tracker(self, stream_id: str) -> PersonTracker:
        if stream_id not in self.trackers:
            self.trackers[stream_id] = PersonTracker(
                lying_threshold=self.lying_threshold,
                cooldown_period=10.0,
                person_timeout=3.0
            )
        return self.trackers[stream_id]

    def process_frame(self, frame, stream_id: str):
        notifications_sent = 0
        persons = []

        h, w = frame.shape[:2]
        self.classifier.set_frame_dimensions(h, w)

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        annotated_frame = frame.copy()

        if not results or results[0].boxes is None or results[0].keypoints is None:
            return annotated_frame, 0

        result = results[0]
        boxes = result.boxes
        keypoints_data = result.keypoints.data.cpu().numpy()
        num_boxes = len(boxes)
        num_kpts = len(keypoints_data)
        num_persons = min(num_boxes, num_kpts)

        for idx in range(num_persons):
            person_keypoints = keypoints_data[idx]
            box = boxes[idx]

            person_id = idx  # frame-local, safe

            kpt_coords = person_keypoints[:, :2]

            kpt_conf = None
            if person_keypoints.shape[1] >= 3:
                kpt_conf = person_keypoints[:, 2].tolist()

            kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
            posture, confidence = self.classifier.classify(kpts, h)

            print(f"Stream {stream_id} | Person {person_id} → {posture}")

            # Alert logic (frame-based, no cooldown)
            if posture == "Lying":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                notification_frame = frame.copy()
                cv2.rectangle(notification_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    notification_frame,
                    f"FALL DETECTED",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                self.notification_handler.save_falling_notification(
                    person_id,
                    notification_frame,
                    stream_id
                )

                notifications_sent += 1
                logger.warning(
                    f"FALL DETECTED | Stream={stream_id} | Person={person_id}"
                )

            persons.append({
                "id": person_id,
                "label": "Falling" if posture == "Lying" else posture,
                "confidence": confidence,
                "keypoints": kpts
            })

        # self.last_persons[stream_id] = persons

        # Annotation pass
        for p in persons:
            annotated_frame = self.frame_annotator.annotate_frame(
                annotated_frame,
                posture_label=p["label"],
                confidence=p["confidence"],
                keypoints=p["keypoints"],
                person_id=p["id"]
            )

        return annotated_frame, notifications_sent
