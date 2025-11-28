# preprocessing/frame_preprocessor.py
import cv2
import torch
import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)

# Standard ImageNet normalization mean & std (RGB order)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FramePreprocessor:
    """
    Handles frame-level preprocessing for action-recognition models.

    Includes:
    - Color conversion (BGR → RGB)
    - Resize to model input size
    - Normalization using ImageNet mean & std
    - Stack frames into 5D tensor format (1, C, frame_window_size, H, W)
    """

    def __init__(self, img_size: int):
        """
        Initialize the preprocessor.

        Args:
            img_size (int): The height/width that each frame must be resized to.
        """
        self.img_size = img_size
        logger.info(f"FramePreprocessor initialized with img_size={self.img_size}")

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.

        Steps:
        - Convert BGR → RGB
        - Resize to (img_size, img_size)
        - Scale pixel values to [0, 1]
        - Normalize using ImageNet mean/std

        Args:
            frame (np.ndarray): Raw frame from OpenCV (BGR format).

        Returns:
            np.ndarray: Normalized frame in (H, W, C) format, float32.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (self.img_size, self.img_size))
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        normalized_frame = (normalized_frame - MEAN) / STD
        logger.debug("Frame preprocessed")
        return normalized_frame

    def make_tensor(self, frame_buffer):
        """
        Stack a sequence of preprocessed frames into a 5D tensor.
        Expected by PyTorchVideo models.

        Args:
            frame_buffer (list[np.ndarray]):
                List of frames (frame_window_size frames), each shaped (H, W, C).

        Returns:
            torch.Tensor:
                Tensor of shape (1, C, frame_window_size, H, W)
        """
        stacked_frames = np.stack(frame_buffer, axis=0)
        stacked_frames = stacked_frames.transpose(3, 0, 1, 2)
        tensor = torch.from_numpy(stacked_frames).unsqueeze(0)
        logger.debug(f"Stacked {len(frame_buffer)} frames into tensor of shape {tensor.shape}")
        return tensor
