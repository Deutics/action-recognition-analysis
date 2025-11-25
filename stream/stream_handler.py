# stream/stream_handler.py
import cv2
import time
from .motion_detector import MotionDetection

class StreamHandler:
    """Handles video capture from webcam, RTSP, or file, with motion filtering and auto-reconnect."""

    def __init__(self, source, max_retries=5, retry_delay=2.0):
        self.source = source
        self.cap = None
        self.motion_detector = MotionDetection()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        retries = 0
        while not self.cap.isOpened() and retries < self.max_retries:
            print(f"[WARN] Cannot open stream {self.source}, retrying in {self.retry_delay}s")
            time.sleep(self.retry_delay)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            retries += 1
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")
        print(f"[INFO] Stream opened: {self.source}")

    def read_frame(self):
        if self.cap is None:
            raise ValueError("Capture not started")

        ret, frame = self.cap.read()
        if not ret:
            # Auto-reconnect logic
            self.cap.release()
            time.sleep(self.retry_delay)
            self.cap = cv2.VideoCapture(self.source)
            return None

        # Detect motion, but always return frame
        motion_flag = self.motion_detector.detect(frame)
        return frame, motion_flag

    def release(self):
        if self.cap:
            self.cap.release()
