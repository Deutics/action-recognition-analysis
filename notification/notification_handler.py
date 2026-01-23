"""
Notification Handler
Handles falling detection notifications and saves annotated frames
"""

import cv2
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class NotificationHandler:
    """Handle falling detection notifications"""

    def __init__(self, output_dir: str = "output/falling_detections"):
        """
        Initialize notification handler

        Args:
            output_dir: Directory to save notification frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Notification handler initialized. Output dir: {self.output_dir}")

    def save_falling_notification(self,
                                  person_id: int,
                                  frame: np.ndarray,
                                  source_id: str) -> str:
        """
        Save falling detection notification
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"falling_person{person_id}_{source_id}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        try:
            # Save frame
            cv2.imwrite(str(filepath), frame)

            logger.warning(
                f"FALLING NOTIFICATION SAVED: "
                f"Person ID={person_id}, Source={source_id}, File={filename}"
            )

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save notification: {e}", exc_info=True)
            return None