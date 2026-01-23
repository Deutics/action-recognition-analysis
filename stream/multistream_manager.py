"""Multistream frame buffer for parallel frame capture with display"""

import threading
import time
import cv2
from typing import Dict, Optional, Tuple
from stream.stream_handler import StreamHandler
from utils.logger import get_logger

logger = get_logger(__name__)


class StreamFrameBuffer:
    def __init__(self, source: str, stream_id: str):
        self.source = source
        self.stream_id = stream_id
        self.stream_handler = StreamHandler(source)
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        self.frame_count = 0
        
    def start(self):
        self.stream_handler.start_stream()
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Stream {self.stream_id} started capturing from {self.source}")
        
    def _capture_loop(self):
        while self.running:
            try:
                frame, _ = self.stream_handler.read_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
            except Exception as e:
                logger.error(f"Stream {self.stream_id} capture error: {e}")
                time.sleep(1)
                
    def get_latest_frame(self) -> Optional[Tuple]:
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.stream_id
            return None, self.stream_id
            
    def update_annotated_frame(self, annotated_frame):
        with self.frame_lock:
            self.latest_annotated_frame = annotated_frame.copy()
            
    def get_display_frame(self):
        with self.frame_lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            elif self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
            
    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        self.stream_handler.release_stream()
        logger.info(f"Stream {self.stream_id} stopped")


class MultiStreamManager:
    def __init__(self, stream_sources: Dict[str, str], enable_display: bool = True):
        self.stream_buffers = {}
        self.enable_display = enable_display
        for stream_id, source in stream_sources.items():
            self.stream_buffers[stream_id] = StreamFrameBuffer(source, stream_id)
            
    def start_all(self):
        for stream_id, buffer in self.stream_buffers.items():
            buffer.start()
        logger.info(f"Started {len(self.stream_buffers)} streams")
        
    def get_all_latest_frames(self) -> Dict[str, any]:
        frames = {}
        for stream_id, buffer in self.stream_buffers.items():
            frame, _ = buffer.get_latest_frame()
            if frame is not None:
                frames[stream_id] = frame
        return frames
        
    def update_annotated_frame(self, stream_id: str, annotated_frame):
        if stream_id in self.stream_buffers:
            self.stream_buffers[stream_id].update_annotated_frame(annotated_frame)
            
    def display_all_streams(self):
        if not self.enable_display:
            return False
            
        for stream_id, buffer in self.stream_buffers.items():
            frame = buffer.get_display_frame()
            if frame is not None:
                # fps_text = f"Stream: {stream_id} | Frames: {buffer.frame_count}"
                # cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f"Stream {stream_id}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
        
    def stop_all(self):
        for buffer in self.stream_buffers.values():
            buffer.stop()
        if self.enable_display:
            cv2.destroyAllWindows()
        logger.info("All streams stopped")
