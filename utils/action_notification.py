# notifications/action_notification.py
import cv2
import os
from datetime import datetime
from utils.logger import get_logger
logger = get_logger(__name__)


class ActionNotification:
    """
    Utility to register and save notification frames.

    Creates a timestamped JPEG file under a camera-specific folder.
    """

    def __init__(self, camera_id, frame, save_root="notifications"):
        """
        Initialize the notification handler.

        Args:
            camera_id (int | str): Identifier for the camera source.
            frame (np.ndarray): Frame image to be saved.
            save_root (str): Root directory for storing notifications.
        """
        self.camera_id = camera_id
        self.frame = frame
        self.save_root = save_root

    def register(self):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        camera_folder = os.path.join(self.save_root, f"camera_{self.camera_id}")
        os.makedirs(camera_folder, exist_ok=True)

        filepath = os.path.join(camera_folder, f"{timestamp}.jpg")
        cv2.imwrite(filepath, self.frame)

        logger.info(f"[NOTIFICATION] Saved {filepath}")
