import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseFormatter:
    """Maintains a sliding buffer of keypoints and builds PoseC3D sample dictionaries."""

    def __init__(self, window_size=24, max_persons=2, num_keypoints=17,
                 frame_width=640, frame_height=360):

        self.window_size = window_size
        self.max_persons = max_persons
        self.num_keypoints = num_keypoints
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Buffers to store keypoints across frames
        self.keypoints_xy_buffer = []      # list of (max_persons, num_keypoints, 2)
        self.keypoints_conf_buffer = []    # list of (max_persons, num_keypoints)

        logger.info(
            f"PoseFormatter initialized (window_size={window_size}, "
            f"max_persons={max_persons}, num_keypoints={num_keypoints})"
        )

    def update_buffer(self, keypoints_xy, keypoints_conf):
        """Append new frame keypoints to buffer."""
        self.keypoints_xy_buffer.append(keypoints_xy)
        self.keypoints_conf_buffer.append(keypoints_conf)

        if len(self.keypoints_xy_buffer) > self.window_size:
            self.keypoints_xy_buffer.pop(0)
            self.keypoints_conf_buffer.pop(0)
            logger.debug("Buffer exceeded window size, oldest frame removed.")

        logger.debug(f"Buffer updated: {len(self.keypoints_xy_buffer)}/{self.window_size} frames stored.")

    def is_ready(self):
        """Check if buffer has enough frames for inference."""
        ready = len(self.keypoints_xy_buffer) == self.window_size
        if ready:
            logger.debug("Buffer is ready for PoseC3D inference.")
        return ready

    def reset(self):
        self.keypoints_xy_buffer.clear()
        self.keypoints_conf_buffer.clear()

    def build_sample(self):
        """Build PoseC3D sample dict from buffered frames."""
        xy = np.stack(self.keypoints_xy_buffer, axis=0)        # (T, M, V, 2)
        conf = np.stack(self.keypoints_conf_buffer, axis=0)    # (T, M, V)

        # Transpose to match PoseC3D expected format
        keypoints = np.transpose(xy, (1, 0, 2, 3))             # (M, T, V, 2)
        keypoints_conf = np.transpose(conf, (1, 0, 2))         # (M, T, V)

        logger.debug("PoseC3D sample built successfully.")

        sample = {
            "modality": "Pose",
            "frame_dir": "live",
            "label": -1,
            "img_shape": (self.frame_height, self.frame_width),
            "original_shape": (self.frame_height, self.frame_width),
            "keypoint": keypoints.astype(np.float32),
            "keypoint_score": keypoints_conf.astype(np.float32),
            "total_frames": keypoints.shape[1],
            "num_clips": 1,
            "clip_len": self.window_size,
        }
        return sample