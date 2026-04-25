# inference/torch_backend.py

import numpy as np
import torch
from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger(__name__)


class TorchBackend:
    """
    Torch CPU backend (Mac / fallback)
    """

    def __init__(self, model_path: str, use_tracking: bool = True):
        logger.info(f"Loading YOLO model (Torch CPU): {model_path}")
        self.model = YOLO(model_path)
        self.use_tracking = use_tracking
        device = "cpu"
        device = "cuda" if torch.cuda.is_available() else device
        device = "mps" if torch.mps.is_available() else device

        self.model.to(device)

    def infer(self, frame):
        """
        Returns:
            keypoints_data (np.ndarray)
            track_ids (list[int])
        """

        if self.use_tracking:
            results = self.model.track(
                frame,
                conf=0.5,
                persist=True,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=0.5,
                verbose=False
            )

        if not results:
            return None, None

        result = results[0]

        if result.keypoints is None:
            return None, None

        keypoints_data = result.keypoints.data.cpu().numpy()

        track_ids = []
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int).tolist()

        return keypoints_data, track_ids
