# preprocessing/slowfast_preprocessor.py
import cv2
import torch
import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


class SlowFastPreprocessor:
    """
    Preprocess frames for SlowFast models.

    Includes:
    - Color conversion (BGR → RGB)
    - Resize to model input size
    - Normalization using custom mean & std
    - Splitting frames into slow and fast pathways
    - Conversion to 5D tensor format (1, C, frame_window_size, H, W)
    """

    def __init__(self, fast_T: int, img_size: int, alpha: int = 4):
        """
        Initialize the preprocessor.

        Args:
            fast_T (int): Number of frames for the fast pathway.
            img_size (int): Target height/width for each frame.
            alpha (int): Temporal stride ratio between fast and slow pathways.
        """
        self.fast_T = fast_T
        self.img_size = img_size
        self.alpha = alpha
        self.slow_T = fast_T // alpha
        logger.info(f"SlowFastPreprocessor initialized with fast_T={fast_T}, img_size={img_size}, alpha={alpha}")

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.

        Args:
            frame (np.ndarray): Raw frame from OpenCV (BGR format).

        Returns:
            np.ndarray: Normalized frame in (H, W, C) format, float32.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size))
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - MEAN) / STD
        logger.debug("Frame preprocessed for SlowFast pathway")
        return norm

    def make_tensor(self, frames):
        """
        Convert a buffer of frames into SlowFast pathway tensors.

        Args:
            frames (list[np.ndarray]): Sequence of frames (frame_window_size frames).

        Returns:
            list[torch.Tensor]: [slow_tensor, fast_tensor], each shaped (1, C, frame_window_size, H, W).
        """
        frames = frames[-self.fast_T:]

        if len(frames) < self.fast_T:
            last_frame = frames[-1]
            frames = frames + [last_frame] * (self.fast_T - len(frames))
            logger.warning(f"Frame buffer too short, duplicated last frame to reach fast_T={self.fast_T}")

        idxs = np.linspace(0, self.fast_T - 1, self.slow_T).astype(int)
        slow_path = [frames[i] for i in idxs]
        fast_path = frames

        slow_tensor = self._to_tensor(slow_path)
        fast_tensor = self._to_tensor(fast_path)

        logger.debug(f"Generated SlowFast tensors: slow_shape={slow_tensor.shape}, fast_shape={fast_tensor.shape}")
        return [slow_tensor, fast_tensor]

    def _to_tensor(self, frames):
        """
        Convert a list of frames into a 5D tensor.

        Args:
            frames (list[np.ndarray]): Preprocessed frames.

        Returns:
            torch.Tensor: Tensor of shape (1, C, frame_window_size, H, W).
        """
        arr = np.stack(frames, axis=0)     # (frame_window_size,H,W,C)
        arr = arr.transpose(3,0,1,2)       # (C,frame_window_size,H,W)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)
        logger.debug(f"Converted {len(frames)} frames to tensor of shape {tensor.shape}")
        return tensor
