# stream/stream_handler.py
import os
import time
import platform
import numpy as np
import cv2

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
        use_hw_decode: bool = True,
        rtsp_latency_ms: int = 200,
    ):
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.out_size = out_size
        self.use_hw_decode = use_hw_decode
        self.rtsp_latency_ms = rtsp_latency_ms

        self.capture = None  # OpenCV fallback capture
        self.motion_detector = MotionDetector()

        # GI/GStreamer members (only used when available)
        self._gst_enabled = False
        self._gst_pipeline = None
        self._gst_sink = None
        self._gst_sample = None

    def _is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    def _is_jetson(self) -> bool:
        return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model")

    def _is_linux(self) -> bool:
        return platform.system().lower() == "linux"

    # ---------------- GI / GStreamer HW Decode ---------------- #

    def _try_enable_gst(self) -> bool:
        """Import GI + Gst once; return True if available."""
        if self._gst_enabled:
            return True
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # noqa
            Gst.init(None)
            self._gst_enabled = True
            return True
        except Exception as e:
            logger.warning(f"GI/GStreamer not available: {e}")
            self._gst_enabled = False
            return False

    def _gst_rtsp_pipeline_jetson_hevc(self, url: str) -> str:
        w, h = self.out_size
        # Output BGR for OpenCV-like processing
        # Note: Use rtph265depay for HEVC/H265 streams (your camera is H265)
        return (
            f'rtspsrc location="{url}" latency={self.rtsp_latency_ms} protocols=tcp ! '
            f'rtph265depay ! h265parse ! '
            f'nvv4l2decoder enable-max-performance=1 ! '
            f'nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true'
        )

    def _gst_rtsp_pipeline_generic(self, url: str) -> str:
        """Non-Jetson generic pipeline (CPU decode). Still useful on Linux with GStreamer."""
        w, h = self.out_size
        return (
            f'rtspsrc location="{url}" latency={self.rtsp_latency_ms} protocols=tcp ! '
            f'decodebin ! videoconvert ! videoscale ! '
            f'video/x-raw,format=BGR,width={w},height={h} ! '
            f'appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true'
        )

    def _gst_open(self) -> bool:
        """Open a GStreamer pipeline via GI and prepare appsink pulling."""
        if not self._try_enable_gst():
            return False

        from gi.repository import Gst  # type: ignore

        if not self._is_rtsp():
            return False

        # Prefer Jetson HW decode pipeline when on Jetson + enabled
        if self._is_linux() and self._is_jetson() and self.use_hw_decode:
            pipeline_str = self._gst_rtsp_pipeline_jetson_hevc(self.source)
            logger.info("Opening RTSP via GI GStreamer Jetson HW HEVC pipeline")
        elif self._is_linux():
            pipeline_str = self._gst_rtsp_pipeline_generic(self.source)
            logger.info("Opening RTSP via GI GStreamer generic pipeline")
        else:
            # On mac/windows, GI may exist but GStreamer RTSP elements often not present
            return False

        try:
            self._gst_pipeline = Gst.parse_launch(pipeline_str)
            self._gst_sink = self._gst_pipeline.get_by_name("appsink")
            if self._gst_sink is None:
                logger.warning("GStreamer appsink not found in pipeline.")
                return False

            ret = self._gst_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning("GStreamer pipeline failed to go to PLAYING.")
                return False

            return True
        except Exception as e:
            logger.warning(f"GStreamer open failed: {e}")
            self._gst_pipeline = None
            self._gst_sink = None
            return False

    def _gst_read_frame(self, timeout_sec: float = 2.0):
        """Pull a frame from appsink; returns (frame, ok). Compatible with GI bindings."""
        from gi.repository import Gst  # type: ignore

        if self._gst_sink is None:
            return None, False

        timeout_ns = int(timeout_sec * 1e9)

        sample = None

        # Preferred: try-pull-sample with timeout (available in many GI builds)
        try:
            sample = self._gst_sink.emit("try-pull-sample", timeout_ns)
        except Exception:
            sample = None

        # Fallback: pull-sample (blocking). We'll keep it safe by limiting how long we wait
        # with a simple time-bounded loop using non-blocking try-pull if unavailable.
        if sample is None:
            t0 = time.time()
            while (time.time() - t0) < timeout_sec:
                try:
                    # Some builds allow pull-sample without blocking too long if max-buffers=1 drop=true
                    sample = self._gst_sink.emit("pull-sample")
                except Exception:
                    sample = None

                if sample is not None:
                    break
                time.sleep(0.01)

        if sample is None:
            return None, False

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None, False

        try:
            data = mapinfo.data
            frame = np.frombuffer(data, dtype=np.uint8).copy().reshape((height, width, 3))
            return frame, True
        finally:
            buf.unmap(mapinfo)

    def _gst_close(self):
        if not self._gst_enabled or self._gst_pipeline is None:
            return
        try:
            from gi.repository import Gst  # type: ignore
            self._gst_pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        self._gst_pipeline = None
        self._gst_sink = None

    # ---------------- OpenCV Fallback ---------------- #

    def _cv_open(self) -> bool:
        logger.info("Opening source via OpenCV default backend (likely FFmpeg)")
        self.capture = cv2.VideoCapture(self.source)
        return bool(self.capture.isOpened())

    # ---------------- Public API ---------------- #

    def start_stream(self):
        self.motion_detector.reset()
        retries = 0

        while retries < self.max_retries:
            opened = False

            # Try GI/GStreamer first on Jetson/Linux RTSP
            if self._is_rtsp() and self._is_linux():
                opened = self._gst_open()

            # Fallback to OpenCV if needed
            if not opened:
                opened = self._cv_open()

            if opened:
                logger.info(f"Stream successfully opened: {self.source}")
                return

            logger.warning(
                f"Cannot open stream '{self.source}', retrying in {self.retry_delay}s "
                f"(Attempt {retries + 1}/{self.max_retries})"
            )
            self.release_stream()
            time.sleep(self.retry_delay)
            self.motion_detector.reset()
            retries += 1

        logger.error(f"Failed to open video source after {retries} attempts: {self.source}")
        raise ValueError(f"Cannot open video source: {self.source}")

    def read_frame(self):
        # Prefer GStreamer if active
        if self._gst_pipeline is not None and self._gst_sink is not None:
            frame, ok = self._gst_read_frame(timeout_sec=2.0)
            if not ok or frame is None:
                logger.warning(f"GStreamer frame pull failed from '{self.source}', reconnecting...")
                self._gst_close()
                time.sleep(self.retry_delay)
                self.motion_detector.reset()
                # Try reopen
                if not self._gst_open():
                    return None, False
                frame, ok = self._gst_read_frame(timeout_sec=2.0)
                if not ok:
                    return None, False
        else:
            # OpenCV path
            if self.capture is None:
                raise ValueError("Capture not started. Call 'start_stream()' first.")
            ok, frame = self.capture.read()
            if not ok or frame is None:
                logger.warning(f"Failed to read frame from '{self.source}', reconnecting...")
                try:
                    self.capture.release()
                except Exception:
                    pass
                time.sleep(self.retry_delay)
                self.motion_detector.reset()
                if not self._cv_open():
                    return None, False
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    return None, False

            # Normalize size on OpenCV path
            if self.out_size is not None:
                w, h = self.out_size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        motion_detected = self.motion_detector.detect_motion(frame)
        return frame, motion_detected

    def release_stream(self):
        self._gst_close()
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
        logger.info(f"Stream released: {self.source}")