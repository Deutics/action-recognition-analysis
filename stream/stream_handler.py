# stream/stream_handler.py
import os
import time
import platform
import subprocess
from typing import Optional, Tuple, Union

import cv2

from .motion_detector import MotionDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class StreamHandler:
    """
    Handles video capture from webcam, RTSP, or file.
    - Cross-platform:
        * macOS/Windows: uses OpenCV default/FFmpeg backend
        * Jetson (Linux aarch64): prefers GStreamer pipeline for RTSP (HW decode)
    - Motion detector is applied on the *same sized* frames (to avoid accumulateWeighted assertion).
    - Resizes AFTER motion detection (or you can choose to resize before, consistently).
    """

    def __init__(
        self,
        source: Union[str, int],
        max_retries: int = 5,
        retry_delay: float = 2.0,
        resize_to: Optional[Tuple[int, int]] = (640, 480),  # (w, h) or None
        prefer_gstreamer_on_jetson: bool = True,
        rtsp_latency_ms: int = 200,
        rtsp_transport: str = "tcp",  # "tcp" or "udp"
    ):
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self.motion_detector = MotionDetector()

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.resize_to = resize_to
        self.prefer_gstreamer_on_jetson = prefer_gstreamer_on_jetson
        self.rtsp_latency_ms = rtsp_latency_ms
        self.rtsp_transport = rtsp_transport

        # Internal
        self._last_open_mode = "unknown"  # "gstreamer" | "opencv"
        self._last_error = ""

    # -------------------------
    # Platform helpers
    # -------------------------
    def _is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def _is_jetson(self) -> bool:
        # Jetson is typically Linux + aarch64, and has nv* gstreamer plugins
        if platform.system().lower() != "linux":
            return False
        if platform.machine().lower() not in ("aarch64", "arm64"):
            return False

        # Optional: check if nvv4l2decoder exists (best signal)
        try:
            p = subprocess.run(
                ["gst-inspect-1.0", "nvv4l2decoder"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return p.returncode == 0
        except Exception:
            # If gst-inspect not available, still consider it Jetson by arch
            return True

    def _url_escape_for_gst(self, url: str) -> str:
        # GStreamer sometimes treats '&' as separator in property strings.
        # Percent-encode it so rtspsrc location parses correctly.
        return url.replace("&", "%26")

    def _probe_rtsp_codec(self, url: str) -> Optional[str]:
        """
        Best-effort probe to identify codec (h264/hevc).
        Uses ffprobe if available. If it fails, returns None (we'll still try).
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-rtsp_transport", self.rtsp_transport,
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nk=1:nw=1",
                url,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
            codec = out.decode("utf-8", errors="ignore").strip().lower()
            if codec:
                return codec  # e.g. "h264" or "hevc"
        except Exception:
            return None
        return None

    # -------------------------
    # GStreamer pipelines (Jetson)
    # -------------------------
    def _gst_pipeline_jetson(self, url: str, codec: Optional[str]) -> str:
        """
        Explicit Jetson RTSP pipeline for OpenCV appsink output BGR.
        Supports HEVC/H264. If codec unknown, default to decodebin (less ideal).
        """
        url_gst = self._url_escape_for_gst(url)
        protocols = "tcp" if self.rtsp_transport.lower() == "tcp" else "udp"
        latency = int(self.rtsp_latency_ms)

        # Prefer explicit depay/parse/decoder. This is what tends to work reliably on Jetson.
        if codec == "hevc":
            depay_parse_decode = "rtph265depay ! h265parse ! nvv4l2decoder"
        elif codec == "h264":
            depay_parse_decode = "rtph264depay ! h264parse ! nvv4l2decoder"
        else:
            # Unknown codec: try decodebin (may fail on some HEVC streams, but better than nothing)
            depay_parse_decode = "decodebin"

        # nvvidconv: NVMM -> CPU accessible; convert to BGR for OpenCV
        pipeline = (
            f'rtspsrc location="{url_gst}" protocols={protocols} latency={latency} drop-on-latency=true ! '
            f'{depay_parse_decode} ! '
            f'nvvidconv ! video/x-raw,format=BGRx ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )
        return pipeline

    # -------------------------
    # Open/close
    # -------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        """
        Decide best backend and open capture.
        """
        # 1) Jetson RTSP via GStreamer (preferred)
        if (
            self.prefer_gstreamer_on_jetson
            and self._is_rtsp()
            and self._is_jetson()
        ):
            codec = self._probe_rtsp_codec(self.source)  # "hevc" / "h264" / None
            gst = self._gst_pipeline_jetson(self.source, codec)
            logger.info(f"Opening RTSP via GStreamer on Jetson (codec={codec or 'unknown'})")
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            self._last_open_mode = "gstreamer"
            if cap.isOpened():
                return cap
            self._last_error = "VideoCapture not opened (gstreamer)"
            cap.release()

            # fallback to OpenCV default below
            logger.warning("GStreamer open failed; falling back to OpenCV/FFmpeg")

        # 2) Default OpenCV/FFmpeg path (mac/windows/linux)
        logger.info("Opening source via OpenCV default backend")
        cap = cv2.VideoCapture(self.source)
        self._last_open_mode = "opencv"
        if not cap.isOpened():
            self._last_error = "VideoCapture not opened (opencv)"
        return cap

    def start_stream(self):
        """Initialize video capture with retries."""
        self.motion_detector.reset()

        retries = 0
        while retries < self.max_retries:
            self.capture = self._open_capture()
            if self.capture is not None and self.capture.isOpened():
                logger.info(f"Stream successfully opened ({self._last_open_mode}): {self.source}")
                # IMPORTANT: reset motion detector again after open to avoid stale sizes
                self.motion_detector.reset()
                return

            retries += 1
            logger.warning(
                f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s "
                f"(Attempt {retries}/{self.max_retries}) | err={self._last_error}"
            )
            try:
                if self.capture:
                    self.capture.release()
            except Exception:
                pass
            time.sleep(self.retry_delay)
            self.motion_detector.reset()

        logger.error(
            f"Failed to open video source after {self.max_retries} attempts: {self.source} | last={self._last_error}"
        )
        raise ValueError(f"Cannot open video source: {self.source}")

    # -------------------------
    # Read frames
    # -------------------------
    def read_frame(self):
        """
        Read a frame from the video source.
        Returns:
            (frame, motion_detected)
        Notes:
            - Motion detection runs on the *original frame size* consistently.
            - Resize is applied AFTER motion detection so background model size stays stable.
        """
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
            # Let caller handle None and keep looping
            return None, False

        # Motion detection BEFORE resize to keep sameSize(src,dst) stable
        motion_detected = self.motion_detector.detect_motion(frame)

        # Resize after motion detection (if configured)
        if self.resize_to is not None:
            w, h = self.resize_to
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        return frame, motion_detected

    # -------------------------
    # Release
    # -------------------------
    def release_stream(self):
        """Release the video capture resource."""
        if self.capture:
            try:
                self.capture.release()
            finally:
                self.capture = None
                logger.info(f"Stream released: {self.source}")