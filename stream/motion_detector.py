import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class MotionDetector:
    """Optimized motion detector using running-average background subtraction."""

    def __init__(self, min_area: int = 300, alpha: float = 0.02, threshold: int = 20):
        """
        Args:
            min_area (int): Minimum contour area to treat as motion.
            alpha (float): Weight for the running average background model.
            threshold (int): Binary threshold for frame differencing.
        """
        self.min_area = min_area
        self.alpha = alpha
        self.threshold = threshold

        self.background_model = None
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        logger.info(
            f"MotionDetector initialized (min_area={min_area}, alpha={alpha}, threshold={threshold})"
        )

    # ------------------------------------------------------------
    # Motion Detection
    # ------------------------------------------------------------
    def detect_motion(self, frame: np.ndarray) -> bool:
        """Return True if motion is detected, else False."""

        if frame is None:
            logger.warning("Received empty frame in detect_motion().")
            return False

        # Convert & blur
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        # Initialize background model
        if self.background_model is None:
            self.background_model = gray_frame.astype("float")
            logger.debug("Background model initialized.")
            return False

        # Update background model
        cv2.accumulateWeighted(gray_frame, self.background_model, self.alpha)

        frame_difference = cv2.absdiff(
            gray_frame, cv2.convertScaleAbs(self.background_model)
        )
        _, threshold_mask = cv2.threshold(
            frame_difference, self.threshold, 255, cv2.THRESH_BINARY
        )

        threshold_mask = cv2.dilate(threshold_mask, self.morph_kernel, iterations=2)

        contours, _ = cv2.findContours(
            threshold_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Detect meaningful motion
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                logger.debug("Motion detected (contour exceeded min_area).")
                return True

        return False

    # ------------------------------------------------------------
    # Reset Background
    # ------------------------------------------------------------
    def reset(self):
        """Reset the background model (e.g., after camera change)."""
        self.background_model = None
        logger.info("MotionDetector background model reset.")
