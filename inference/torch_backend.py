from prompt_toolkit.contrib.telnet.log import logger
from sympy.printing.pytorch import torch
from torch.xpu import device
from ultralytics import YOLO
import numpy as np
from .backend import PoseBackend
from utils.logger import get_logger


logger = get_logger(__name__)


class TorchBackend(PoseBackend):

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        device = 'cpu'
        device = 'cuda' if torch.cuda.is_available() else device
        device = 'mps' if torch.mps.is_available() else device
        self.model.to(device)

    def infer(self, frame: np.ndarray):

        results = self.model.track(
            frame,
            conf=0.5,
            persist=True,
            verbose=False
        )

        if not results:
            return None, []

        result = results[0]

        if result.keypoints is None:
            return None, []

        keypoints = result.keypoints.data.cpu().numpy()

        track_ids = []
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        return keypoints, track_ids