from typing import List, Any
from utils.logger import get_logger
logger = get_logger(__name__)

class FrameBuffer:
    """
    Maintains a fixed-size buffer of frames for processing.
    Oldest frames are discarded when the buffer exceeds the required size.
    """

    def __init__(self, max_frames: int):
        self.max_frames: int = max_frames
        self._buffer: List[Any] = []
        logger.info(f"FrameBuffer initialized with capacity: {self.max_frames}")

    def add_frame(self, frame: Any):
        """Add a frame to the buffer and discard the oldest if buffer exceeds capacity."""
        self._buffer.append(frame)
        if len(self._buffer) > self.max_frames:
            removed_frame = self._buffer.pop(0)
            logger.debug("Oldest frame removed from buffer to maintain size limit")
        logger.debug(f"Frame added to buffer. Current size: {len(self._buffer)}")

    def is_full(self) -> bool:
        """Check if the buffer has reached the required number of frames."""
        return len(self._buffer) == self.max_frames

    def get_frames(self) -> List[Any]:
        """Retrieve all frames in the buffer."""
        return self._buffer
