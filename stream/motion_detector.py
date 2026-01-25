import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class MotionDetector:
    """
    Robust motion detector using running-average background subtraction.

    Key fixes vs previous version:
    - Resets background model if frame size changes (prevents accumulateWeighted assertion crash)
    - Ensures background model is float32 and same shape as grayscale input
    - Safer handling of occasional bad frames
    """

    def __init__(self, min_area: int = 300, alpha: float = 0.02, threshold: int = 20):
        self.min_area = min_area
        self.alpha = float(alpha)
        self.threshold = int(threshold)

        self.background_model: np.ndarray | None = None
        self._bg_shape: tuple | None = None  # shape of grayscale frame
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        logger.info(
            f"MotionDetector initialized (min_area={min_area}, alpha={alpha}, threshold={threshold})"
        )

    def _init_background(self, gray_frame: np.ndarray):
        """
        Initialize (or re-initialize) the background model.
        accumulateWeighted expects dst to be float, same size/channels.
        """
        self.background_model = gray_frame.astype(np.float32)
        self._bg_shape = gray_frame.shape
        logger.info(f"Background model initialized/reset. shape={self._bg_shape}")

    def detect_motion(self, frame: np.ndarray) -> bool:
        """Return True if motion is detected, else False."""
        if frame is None:
            logger.debug("detect_motion: received None frame")
            return False

        # Some decoders can occasionally deliver empty frames
        if not hasattr(frame, "shape") or frame.size == 0:
            logger.debug("detect_motion: received empty/invalid frame")
            return False

        # Convert & blur (grayscale => single channel)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            # If frame is already gray or has unexpected channels
            if len(frame.shape) == 2:
                gray = frame
            else:
                logger.warning("detect_motion: unsupported frame format, skipping")
                return False

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Initialize background model
        if self.background_model is None:
            self._init_background(gray)
            return False

        # ✅ Critical fix: if stream renegotiates resolution, reset background
        if self._bg_shape != gray.shape:
            logger.warning(f"Frame size changed {self._bg_shape} -> {gray.shape}. Resetting background model.")
            self._init_background(gray)
            return False

        # Ensure float32 background (safety)
        if self.background_model.dtype != np.float32:
            self.background_model = self.background_model.astype(np.float32)

        # Update background model
        # (gray is uint8, bg is float32, shapes match => safe)
        cv2.accumulateWeighted(gray, self.background_model, self.alpha)

        # Foreground mask
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background_model))
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.dilate(mask, self.morph_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                return True

        return False

    def reset(self):
        """Reset the background model (e.g., after reconnect/camera change)."""
        self.background_model = None
        self._bg_shape = None
        logger.info("MotionDetector background model reset.")