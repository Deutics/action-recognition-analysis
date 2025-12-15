from utils.logger import get_logger
import numpy as np
from ultralytics import YOLO

logger = get_logger(__name__)


class YOLOPoseDetector:
    """Wrapper for YOLO Pose inference with padding/truncation."""

    def __init__(self, model_path="yolo11n-pose.pt", max_persons=2, num_keypoints=17):
        self.model = YOLO(model_path)
        self.max_persons = max_persons
        self.num_keypoints = num_keypoints

    def infer(self, frame):
        results = self.model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints) == 0:
            logger.debug("No keypoints detected, returning zero-padded arrays.")
            keypoints_xy = np.zeros((self.max_persons, self.num_keypoints, 2))
            keypoints_conf = np.zeros((self.max_persons, self.num_keypoints))
            return keypoints_xy, keypoints_conf

        keypoints_xy = results.keypoints.xy.cpu().numpy()      # (M, V, 2)
        keypoints_conf = results.keypoints.conf.cpu().numpy()  # (M, V)

        num_detected = keypoints_xy.shape[0]
        if num_detected < self.max_persons:
            pad_xy = np.zeros((self.max_persons - num_detected, self.num_keypoints, 2))
            pad_conf = np.zeros((self.max_persons - num_detected, self.num_keypoints))
            keypoints_xy = np.vstack([keypoints_xy, pad_xy])
            keypoints_conf = np.vstack([keypoints_conf, pad_conf])
        elif num_detected > self.max_persons:
            keypoints_xy = keypoints_xy[:self.max_persons]
            keypoints_conf = keypoints_conf[:self.max_persons]

        return keypoints_xy, keypoints_conf