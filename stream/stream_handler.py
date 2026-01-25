# stream/stream_handler.py
import cv2
import time
from .motion_detector import MotionDetector
from utils.logger import get_logger
logger = get_logger(__name__)


class StreamHandler:
    """
    Handles video capture from webcam, RTSP, or file.
    Includes motion detection and auto-reconnect on failure.
    """

    def __init__(self, source: str, max_retries: int = 5, retry_delay: float = 2.0):
        self.source = source
        self.capture = None
        self.motion_detector = MotionDetector()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0

    def start_stream(self):
        """Initialize video capture with retries."""
        self.capture = cv2.VideoCapture(self.source)
        self.motion_detector.reset()
        retries = 0

        while not self.capture.isOpened() and retries < self.max_retries:
            logger.warning(
                f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s (Attempt {retries + 1})"
            )
            time.sleep(self.retry_delay)
            self.capture.release()
            self.capture = cv2.VideoCapture(self.source)
            self.motion_detector.reset()
            retries += 1

        if not self.capture.isOpened():
            logger.error(f"Failed to open video source after {retries} attempts: {self.source}")
            raise ValueError(f"Cannot open video source: {self.source}")

        logger.info(f"Stream successfully opened: {self.source}")

    def read_frame(self):
        """
        Read a frame from the video source.
        Returns:
            tuple: (frame, motion_detected)
        """
        if self.capture is None:
            logger.error("Attempted to read frame before starting the stream.")
            raise ValueError("Capture not started. Call 'start_stream()' first.")

        ret, frame = self.capture.read()
        if not ret:
            logger.warning(f"Failed to read frame from '{self.source}', attempting reconnect...")
            self.capture.release()
            time.sleep(self.retry_delay)
            self.capture = cv2.VideoCapture(self.source)
            self.motion_detector.reset()
            return None, False

        motion_detected = self.motion_detector.detect_motion(frame)
        frame = cv2.resize(frame, (640, 480))
        return frame, motion_detected

    def release_stream(self):
        """Release the video capture resource."""
        if self.capture:
            self.capture.release()
            logger.info(f"Stream released: {self.source}")
