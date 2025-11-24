import cv2

class StreamHandler:
    """Handles video capture from webcam, RTSP, or file."""

    def __init__(self, source):
        self.source = source
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

    def read_frame(self):
        if self.cap is None:
            raise ValueError("Capture not started")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
