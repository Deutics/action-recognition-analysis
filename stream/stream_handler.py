# stream/stream_handler.py

import os
import cv2
import time
import numpy as np
import platform
from .motion_detector import MotionDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class StreamHandler:
    def __init__(
        self,
        source: str,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        out_size=(640, 480),
    ):
        self.source = source
        self.capture = None  # used for OpenCV fallback
        self.pipeline = None  # used for Jetson GStreamer
        self.appsink = None
        self.motion_detector = MotionDetector()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.out_size = out_size

        self.use_gst_hw = False

    # ---------------------------------------------------------

    def _is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def _is_jetson(self) -> bool:
        return os.path.exists("/etc/nv_tegra_release")

    # ---------------------------------------------------------
    # JETSON HARDWARE PIPELINE (H265)
    # ---------------------------------------------------------

    def _build_jetson_pipeline(self, url: str) -> str:
        w, h = self.out_size
        return (
            f'rtspsrc location="{url}" latency=200 protocols=tcp ! '
            f'rtph265depay ! h265parse ! '
            f'nvv4l2decoder enable-max-performance=1 ! '
            f'nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink name=appsink emit-signals=true '
            f'drop=true max-buffers=1 sync=false'
        )

    def _open_gst_hw(self):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)

            pipeline_str = self._build_jetson_pipeline(self.source)
            logger.info("Opening RTSP via Jetson HW GStreamer pipeline")

            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("appsink")

            if self.appsink is None:
                logger.error("appsink not found in pipeline")
                return False

            self.pipeline.set_state(Gst.State.PLAYING)
            self.use_gst_hw = True
            return True

        except Exception as e:
            logger.warning(f"GStreamer HW init failed: {e}")
            return False

    # ---------------------------------------------------------
    # OPENCV FALLBACK (MAC / OTHER OS)
    # ---------------------------------------------------------

    def _open_opencv(self):
        logger.info("Opening source via OpenCV backend")
        self.capture = cv2.VideoCapture(self.source)
        self.use_gst_hw = False
        return self.capture.isOpened()

    # ---------------------------------------------------------

    def start_stream(self):
        self.motion_detector.reset()
        retries = 0

        while retries < self.max_retries:

            # Try Jetson HW first
            if self._is_rtsp() and self._is_jetson():
                if self._open_gst_hw():
                    logger.info(f"Stream opened with Jetson HW: {self.source}")
                    return

            # Fallback to OpenCV
            if self._open_opencv():
                logger.info(f"Stream opened with OpenCV: {self.source}")
                return

            logger.warning(
                f"Cannot open stream '{self.source}', retrying "
                f"({retries + 1}/{self.max_retries})"
            )

            time.sleep(self.retry_delay)
            retries += 1

        raise ValueError(f"Cannot open video source: {self.source}")

    # ---------------------------------------------------------

    def read_frame(self):

        # ------------------------------
        # JETSON HARDWARE MODE
        # ------------------------------
        if self.use_gst_hw:
            from gi.repository import Gst

            sample = self.appsink.emit(
                "try-pull-sample", 200 * 1000 * 1000
            )  # 200ms timeout

            if sample is None:
                return None, False

            buf = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)

            width = s.get_value("width")
            height = s.get_value("height")

            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                return None, False

            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = arr.reshape((height, width, 3))
            finally:
                buf.unmap(mapinfo)

        # ------------------------------
        # OPENCV MODE
        # ------------------------------
        else:
            if self.capture is None:
                return None, False

            ret, frame = self.capture.read()
            if not ret or frame is None:
                return None, False

            if self.out_size:
                w, h = self.out_size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        motion_detected = self.motion_detector.detect_motion(frame)
        return frame, motion_detected

    # ---------------------------------------------------------

    def release_stream(self):
        if self.use_gst_hw and self.pipeline:
            from gi.repository import Gst
            self.pipeline.set_state(Gst.State.NULL)
            logger.info("Jetson HW stream released")

        if self.capture:
            self.capture.release()
            logger.info("OpenCV stream released")