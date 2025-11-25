# motion_detector.py
import cv2
import numpy as np
from collections import deque

class MotionDetection:
    """Optimized motion detector using running average background subtraction."""

    def __init__(self, min_area: int = 300, alpha: float = 0.02, threshold: int = 20):
        self.min_area = min_area
        self.alpha = alpha  # weight for running average
        self.threshold = threshold
        self.bg_model = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.bg_model is None:
            self.bg_model = gray.astype("float")
            return False

        cv2.accumulateWeighted(gray, self.bg_model, self.alpha)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg_model))
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, self.kernel, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= self.min_area:
                return True
        return False

    def reset(self):
        self.bg_model = None
