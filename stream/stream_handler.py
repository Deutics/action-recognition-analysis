# stream/stream_handler.py
import os
import cv2
import time
import platform
from typing import Optional, Tuple

from .motion_detector import MotionDetector
from utils.logger import get_logger

logger = get_logger(__name__)


def _is_jetson() -> bool:
    # Simple Jetson detection
    if platform.system().lower() != "linux":
        return False
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/usr/sbin/nvpmodel")


def _is_rtsp(src: str) -> bool:
    return isinstance(src, str) and src.lower().startswith("rtsp://")


class StreamHandler:
    """
    Handles video capture from webcam, RTSP, or file.
    - Jetson: prefers GStreamer (nvv4l2decoder) for RTSP
    - Others: uses OpenCV default backend
    Includes motion detection and auto-reconnect on failure.
    """

    def __init__(
        self,
        source,
        max_retries: int = 10,
        retry_delay: float = 2.0,
        target_size: Optional[Tuple[int, int]] = (640, 480),  # set None to disable resize
        prefer_gstreamer_on_jetson: bool = True,
        rtsp_latency_ms: int = 200,
    ):
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self.motion_detector = MotionDetector()

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.target_size = target_size

        self.prefer_gstreamer_on_jetson = prefer_gstreamer_on_jetson
        self.rtsp_latency_ms = rtsp_latency_ms

        self._opened_backend = "none"  # "gst-hevc", "gst-h264", "opencv"

    # ----------------------------
    # GStreamer pipelines (Jetson)
    # ----------------------------
    def _gst_url(self, url: str) -> str:
        # Escape '&' for GStreamer property parsing
        return url.replace("&", "%26")

    def _gst_pipeline_jetson_hevc(self, url: str) -> str:
        url_gst = self._gst_url(url)
        return (
            f'rtspsrc location="{url_gst}" protocols=tcp latency={self.rtsp_latency_ms} drop-on-latency=true ! '
            f'queue ! rtph265depay ! queue ! h265parse ! queue ! nvv4l2decoder ! '
            f'queue ! nvvidconv ! video/x-raw,format=BGRx ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

    def _gst_pipeline_jetson_h264(self, url: str) -> str:
        url_gst = self._gst_url(url)
        return (
            f'rtspsrc location="{url_gst}" protocols=tcp latency={self.rtsp_latency_ms} drop-on-latency=true ! '
            f'queue ! rtph264depay ! queue ! h264parse ! queue ! nvv4l2decoder ! '
            f'queue ! nvvidconv ! video/x-raw,format=BGRx ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

    # ----------------------------
    # Open helpers
    # ----------------------------
    def _open_with_gstreamer(self, url: str) -> bool:
        # Try HEVC then H264 (safe even if stream is only one of them)
        gst_hevc = self._gst_pipeline_jetson_hevc(url)
        logger.info("Opening RTSP via GStreamer on Jetson (try=hevc)")
        cap = cv2.VideoCapture(gst_hevc, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.capture = cap
            self._opened_backend = "gst-hevc"
            return True
        cap.release()

        gst_h264 = self._gst_pipeline_jetson_h264(url)
        logger.info("Opening RTSP via GStreamer on Jetson (try=h264)")
        cap = cv2.VideoCapture(gst_h264, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.capture = cap
            self._opened_backend = "gst-h264"
            return True
        cap.release()
        return False

    def _open_with_opencv(self) -> bool:
        logger.info("Opening source via OpenCV default backend")
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            self.capture = cap
            self._opened_backend = "opencv"
            return True
        cap.release()
        return False

    # ----------------------------
    # Public API
    # ----------------------------
    def start_stream(self):
        """Initialize video capture with retries."""
        self.release_stream()
        self.motion_detector.reset()

        retries = 0
        last_err = "unknown"

        while retries < self.max_retries:
            try:
                opened = False

                if (
                    self.prefer_gstreamer_on_jetson
                    and _is_jetson()
                    and _is_rtsp(self.source)
                ):
                    opened = self._open_with_gstreamer(self.source)
                    if not opened:
                        logger.warning("GStreamer open failed; falling back to OpenCV/FFmpeg")

                if not opened:
                    opened = self._open_with_opencv()

                if not opened or self.capture is None or not self.capture.isOpened():
                    last_err = "VideoCapture not opened"
                    raise RuntimeError(last_err)

                logger.info(f"Stream successfully opened ({self._opened_backend}): {self.source}")
                # Reset background after successful open
                self.motion_detector.reset()
                return

            except Exception as e:
                last_err = str(e)
                retries += 1
                logger.warning(
                    f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s "
                    f"(Attempt {retries}/{self.max_retries}) | err={last_err}"
                )
                self.release_stream()
                self.motion_detector.reset()
                time.sleep(self.retry_delay)

        logger.error(
            f"Failed to open video source after {self.max_retries} attempts: {self.source} | last={last_err}"
        )
        raise ValueError(f"Cannot open video source: {self.source}")

    def _resize_if_needed(self, frame):
        if self.target_size is None:
            return frame
        w, h = self.target_size
        # simple resize (fast). If you want letterbox, tell me.
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    def read_frame(self):
        """
        Read a frame from the video source.
        Returns:
            tuple: (frame, motion_detected)
        """
        if self.capture is None:
            raise ValueError("Capture not started. Call 'start_stream()' first.")

        ret, frame = self.capture.read()

        if not ret or frame is None:
            logger.warning(f"Failed to read frame from '{self.source}', reconnecting...")
            self.release_stream()
            time.sleep(self.retry_delay)
            self.start_stream()
            return None, False

        # Detect motion on the *original* frame size for consistency
        motion_detected = self.motion_detector.detect_motion(frame)

        # Now resize for downstream model/UI if needed
        frame = self._resize_if_needed(frame)
        return frame, motion_detected

    def release_stream(self):
        """Release the video capture resource."""
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
            logger.info(f"Stream released: {self.source}")