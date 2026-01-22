"""
Notification Handler
Handles falling and force-intent detection notifications with frame saving
"""

import cv2
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class NotificationHandler:
    """Handle detection notifications (falling, force-intent, etc.)"""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize notification handler

        Args:
            output_dir: Base output directory
        """
        self.base_output_dir = Path(output_dir)
        self.falling_dir = self.base_output_dir / "falling_detections"
        self.force_intent_dir = self.base_output_dir / "force_intent_detections"
        
        self.falling_dir.mkdir(parents=True, exist_ok=True)
        self.force_intent_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Notification handler initialized.")
        logger.info(f"  Falling output dir: {self.falling_dir}")
        logger.info(f"  Force-intent output dir: {self.force_intent_dir}")

    def save_falling_notification(self,
                                  person_id: int,
                                  frame: np.ndarray,
                                  source_id: str) -> str:
        """
        Save falling detection notification

        Args:
            person_id: Tracked person ID
            frame: Annotated frame with detection
            source_id: Video source identifier

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"falling_person{person_id}_{source_id}_{timestamp}.jpg"
        filepath = self.falling_dir / filename

        try:
            cv2.imwrite(str(filepath), frame)
            logger.warning(
                f"FALLING NOTIFICATION SAVED: "
                f"Person ID={person_id}, Source={source_id}, File={filename}"
            )
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save falling notification: {e}", exc_info=True)
            return None

    def save_force_intent_notification(self,
                                       person_id: int,
                                       frame: np.ndarray,
                                       source_id: str,
                                       signal_breakdown: Optional[Dict[str, Any]] = None) -> str:
        """
        Save force-intent door break-in detection notification as annotated frame (simple image format)

        Args:
            person_id: Tracked person ID
            frame: Annotated frame with detection
            source_id: Video source identifier
            signal_breakdown: Biomechanical signal analysis breakdown (for logging only)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename_base = f"force_intent_person{person_id}_{source_id}_{timestamp}"
        frame_path = self.force_intent_dir / f"{filename_base}.jpg"

        try:
            # Save annotated frame only (simple format, no JSON)
            cv2.imwrite(str(frame_path), frame)

            logger.warning(
                f"FORCE-INTENT ALARM SAVED: "
                f"Person ID={person_id}, Source={source_id}, File={filename_base}.jpg"
            )
            
            # Log signal breakdown for debugging
            if signal_breakdown and 'signals' in signal_breakdown:
                active = signal_breakdown['signals']
                active_count = signal_breakdown.get('active_count', 0)
                logger.warning(
                    f"  Active signals: {active_count}/{signal_breakdown.get('required_count', 2)}"
                )
                for sig_name, sig_status in active.items():
                    status_str = "[YES]" if sig_status else "[NO]"
                    logger.warning(f"    {status_str} {sig_name}")

            return str(frame_path)

        except Exception as e:
            logger.error(f"Failed to save force-intent notification: {e}", exc_info=True)
            return None