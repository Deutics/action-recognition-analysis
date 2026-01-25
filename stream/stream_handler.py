# stream/stream_handler.py
import os
import cv2
import time
import platform
from .motion_detector import MotionDetector
from utils.logger import get_logger

logger = get_logger(__name__)

class StreamHandler:
    def __init__(self, source: str, max_retries: int = 5, retry_delay: float = 2.0, out_size=(640, 480)):
        self.source = source
        self.capture = None
        self.motion_detector = MotionDetector()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.out_size = out_size

    def _is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def _is_jetson(self) -> bool:
        # quick heuristic
        return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model")

    def _gst_rtsp_jetson_hevc(self, url: str) -> str:
        # Quote URL; keep & unescaped inside quotes
        w, h = self.out_size
        return (
            f'rtspsrc location="{url}" protocols=tcp latency=200 drop-on-latency=true ! '
            f'rtph265depay ! h265parse ! nvv4l2decoder ! '
            f'nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

    def _gst_rtsp_generic(self, url: str) -> str:
        # Cross-platform-ish pipeline (no Jetson-specific elements)
        # Works on Linux with good plugin coverage; on Windows/mac this usually won’t exist unless GStreamer is installed.
        w, h = self.out_size
        return (
            f'rtspsrc location="{url}" protocols=tcp latency=200 drop-on-latency=true ! '
            f'decodebin ! videoconvert ! videoscale ! video/x-raw,format=BGR,width={w},height={h} ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

    def _open_capture(self) -> cv2.VideoCapture:
        # Try best option first
        if self._is_rtsp():
            if self._is_jetson():
                gst = self._gst_rtsp_jetson_hevc(self.source)
                logger.info("Opening RTSP via explicit Jetson GStreamer HEVC pipeline")
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap
                logger.warning("Jetson HEVC GStreamer pipeline failed to open")

            # Generic GStreamer as second attempt (Linux only typically)
            gst = self._gst_rtsp_generic(self.source)
            logger.info("Opening RTSP via generic GStreamer pipeline")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            logger.warning("Generic GStreamer pipeline failed to open")

        # Fallback: OpenCV default backend (FFmpeg on Linux)
        logger.info("Opening source via OpenCV default backend (likely FFmpeg)")
        return cv2.VideoCapture(self.source)

    def start_stream(self):
        self.motion_detector.reset()
        retries = 0

        while retries < self.max_retries:
            self.capture = self._open_capture()
            if self.capture.isOpened():
                logger.info(f"Stream successfully opened: {self.source}")
                return

            logger.warning(
                f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s "
                f"(Attempt {retries + 1}/{self.max_retries})"
            )
            try:
                self.capture.release()
            except Exception:
                pass
            time.sleep(self.retry_delay)
            self.motion_detector.reset()
            retries += 1

        logger.error(f"Failed to open video source after {retries} attempts: {self.source}")
        raise ValueError(f"Cannot open video source: {self.source}")

    def read_frame(self):
        if self.capture is None:
            raise ValueError("Capture not started. Call 'start_stream()' first.")

        ret, frame = self.capture.read()
        if not ret or frame is None:
            logger.warning(f"Failed to read frame from '{self.source}', attempting reconnect...")
            try:
                self.capture.release()
            except Exception:
                pass
            time.sleep(self.retry_delay)
            self.motion_detector.reset()
            self.capture = self._open_capture()
            return None, False

        # If GStreamer is doing the scaling, this is already (640x480).
        # If FFmpeg fallback, we still normalize size here.
        if self.out_size is not None:
            w, h = self.out_size
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        motion_detected = self.motion_detector.detect_motion(frame)
        return frame, motion_detected

    def release_stream(self):
        if self.capture:
            self.capture.release()
            logger.info(f"Stream released: {self.source}")