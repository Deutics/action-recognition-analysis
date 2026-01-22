"""Door ROI visualization."""
import cv2
import numpy as np
from typing import Tuple
from config.detection_config import DoorROIConfig


class DoorROIRenderer:
    """Renders door ROI overlay on frames."""
    
    @staticmethod
    def render(frame: np.ndarray, roi_config: DoorROIConfig, 
               frame_width: int, frame_height: int) -> np.ndarray:
        """Draw door ROI overlay on frame."""
        if roi_config is None:
            return frame
        
        x_center = int(roi_config.x_center_ratio * frame_width)
        y_center = int(roi_config.y_center_ratio * frame_height)
        width = int(roi_config.width_ratio * frame_width)
        height = int(roi_config.height_ratio * frame_height)
        
        x1 = max(0, x_center - width // 2)
        y1 = max(0, y_center - height // 2)
        x2 = min(frame_width, x_center + width // 2)
        y2 = min(frame_height, y_center + height // 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(frame, (x_center, y_center), 8, (0, 0, 255), -1)
        
        label = "DOOR ROI"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1 - 5, y1 - label_size[1] - 15),
                     (x1 + label_size[0] + 5, y1 - 5), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    @staticmethod
    def calculate_door_center(roi_config: DoorROIConfig,
                             frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Calculate door center coordinates."""
        return (
            int(roi_config.x_center_ratio * frame_width),
            int(roi_config.y_center_ratio * frame_height)
        )
