# stream/stream_handler.py

import os
import cv2
import time
from urllib.parse import urlparse
import sys

from .motion_detector import MotionDetector
from utils.logger import get_logger


logger = get_logger(__name__)


def _can_use_gstreamer_rtsp():
    return sys.platform.startswith("linux")  # Jetson/Linux

def _is_rtsp(src) -> bool:
    if not isinstance(src, str):
        return False
    return src.strip().lower().startswith("rtsp://")


def _is_file(src) -> bool:
    # basic heuristic
    return isinstance(src, str) and (src.endswith(".mp4") or src.endswith(".avi") or src.endswith(".mkv"))


def _build_gst_rtsp_pipeline(url: str, use_hw: bool = True, width: int = 1280, height: int = 720, latency: int = 200):
    """
    Jetson-friendly RTSP pipeline.
    - appsink drop=true max-buffers=1: latest-frame-only
    - sync=false: low latency (don’t wait on timestamps)
    - nv* elements are Jetson hardware decode path (if available)
    """
    if use_hw:
        # Hardware decode path (best on Jetson)
        return (
            f"rtspsrc location={url} latency={latency} protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, width={w}, height={h}, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        ).format(w=width, h=height)
    else:
        # Software decode fallback
        return (
            f"rtspsrc location={url} latency={latency} protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "avdec_h264 ! videoconvert ! "
            f"video/x-raw, width={width}, height={height}, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )


class StreamHandler:
    """
    Handles video capture from webcam, RTSP, or file.
    Includes motion detection and auto-reconnect on failure.
    """

    def __init__(
        self,
        source,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        target_size=(640, 480),          # final size you want downstream
        prefer_gstreamer: bool = True,    # enable on Jetson
        gst_use_hw: bool = True,          # try Jetson HW decode
        gst_latency: int = 200,
        read_warmup_frames: int = 5,      # discard a few frames after connect
        open_timeout_sec: float = 15.0,
    ):
        self.source = source
        self.capture = None
        self.motion_detector = MotionDetector()

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.target_size = target_size
        self.prefer_gstreamer = prefer_gstreamer
        self.gst_use_hw = gst_use_hw
        self.gst_latency = gst_latency
        self.read_warmup_frames = read_warmup_frames
        self.open_timeout_sec = open_timeout_sec

    # ----------------------- internal helpers -----------------------

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Opens cv2.VideoCapture for source.
        Uses GStreamer pipeline for RTSP when enabled.
        """
        src = self.source

        # Webcam integer
        if isinstance(src, int):
            cap = cv2.VideoCapture(src)
            return cap

        # RTSP
        if _is_rtsp(src) and self.prefer_gstreamer and _can_use_gstreamer_rtsp():
            pipeline = _build_gst_rtsp_pipeline(
                src,
                use_hw=self.gst_use_hw,
                width=self.target_size[0],
                height=self.target_size[1],
                latency=self.gst_latency,
            )
            logger.info(f"Opening RTSP via GStreamer (hw={self.gst_use_hw})")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            return cap

        # File / HTTP / others
        cap = cv2.VideoCapture(src)
        return cap

    def _wait_for_first_frame(self, cap: cv2.VideoCapture) -> bool:
        """
        Some RTSP connections "open" but don't deliver frames immediately.
        This waits a bit for frames to arrive.
        """
        deadline = time.time() + self.open_timeout_sec
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                # warmup done; optionally discard a few frames
                for _ in range(max(0, self.read_warmup_frames - 1)):
                    cap.read()
                return True
            time.sleep(0.1)
        return False

    def _reconnect(self) -> None:
        self.release_stream()
        self.motion_detector.reset()
        time.sleep(self.retry_delay)
        self.start_stream()

    # ----------------------- public API -----------------------

    def start_stream(self):
        """Initialize video capture with retries + wait for frames."""
        self.motion_detector.reset()

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                cap = self._open_capture()

                if not cap.isOpened():
                    raise RuntimeError("VideoCapture not opened")

                # Wait for a real frame (important for RTSP)
                if not self._wait_for_first_frame(cap):
                    cap.release()
                    raise RuntimeError("Stream opened but no frames received (timeout)")

                self.capture = cap
                logger.info(f"Stream successfully opened: {self.source}")
                return

            except Exception as e:
                last_err = e
                logger.warning(
                    f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s "
                    f"(Attempt {attempt}/{self.max_retries}) | err={e}"
                )
                try:
                    if self.capture:
                        self.capture.release()
                except Exception:
                    pass
                time.sleep(self.retry_delay)

        logger.error(f"Failed to open video source after {self.max_retries} attempts: {self.source} | last={last_err}")
        raise ValueError(f"Cannot open video source: {self.source}")

    def read_frame(self):
        """
        Read a frame from the video source.
        Returns:
            tuple: (frame_resized, motion_detected)
        """
        if self.capture is None:
            raise ValueError("Capture not started. Call 'start_stream()' first.")

        ret, frame = self.capture.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning(f"Failed to read frame from '{self.source}', reconnecting...")
            self._reconnect()
            return None, False

        # ✅ Resize FIRST, so motion detector always sees consistent size
        if self.target_size:
            frame = cv2.resize(frame, self.target_size)

        motion_detected = self.motion_detector.detect_motion(frame)
        return frame, motion_detected

    def release_stream(self):
        """Release the video capture resource."""
        try:
            if self.capture:
                self.capture.release()
        finally:
            self.capture = None
            logger.info(f"Stream released: {self.source}")