import os
import platform
from typing import Dict, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger(__name__)


class HumanDetector:
    """
    Human detector wrapper.
    - Prefers TensorRT engine on Jetson when model_path points to .engine
    - Falls back to Torch inference otherwise
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_det: int = 30,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Human detector model not found: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        self.model = YOLO(model_path, task="detect")
        self.device = self._resolve_device(model_path)
        logger.info(f"HumanDetector loaded: model={model_path}, device={self.device}")

    @staticmethod
    def _resolve_device(model_path: str):
        if model_path.lower().endswith(".engine"):
            # Let Ultralytics use the TensorRT engine backend directly without forcing
            # a CUDA device through torch's device selection.
            return None

        device = "cpu"
        device = "cuda" if torch.cuda.is_available() else device
        device = "mps" if (platform.system() == "Darwin" and torch.mps.is_available()) else device
        return device

    def detect_person_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, conf) for class 0 (person).
        """
        if frame is None:
            return []

        predict_kwargs: Dict[str, object] = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": [0],
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(frame, **predict_kwargs)

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),))

        h, w = frame.shape[:2]
        boxes = []
        for i, b in enumerate(xyxy):
            x1 = int(max(0, min(w - 1, b[0])))
            y1 = int(max(0, min(h - 1, b[1])))
            x2 = int(max(0, min(w - 1, b[2])))
            y2 = int(max(0, min(h - 1, b[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2, float(confs[i])))

        return boxes
