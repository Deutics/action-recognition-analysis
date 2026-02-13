from abc import ABC, abstractmethod
from venv import logger

import numpy as np
from utils.logger import get_logger


logger = get_logger(__name__)


class PoseBackend(ABC):

    @abstractmethod
    def infer(self, frame: np.ndarray):
        """
        Returns:
            keypoints (numpy array)
            track_ids (list[int])
        """
        pass