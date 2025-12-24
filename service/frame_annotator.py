"""
Frame Annotator
Annotate video frames with posture detection results
"""

import cv2
import numpy as np
from typing import Tuple, Dict


class FrameAnnotator:
    """Annotate frames with posture detection results"""
    
    # Color mapping for postures
    POSTURE_COLORS = {
        "Standing": (0, 255, 0),      # Green
        "Sitting": (255, 0, 0),       # Blue
        "Bending": (0, 165, 255),     # Orange
        "Squatting": (255, 0, 255),   # Magenta
        "Lying": (0, 255, 255),       # Yellow
        "Unknown": (128, 128, 128),   # Gray
        "Insufficient_Keypoints": (192, 192, 192),  # Light Gray
    }
    
    @staticmethod
    def annotate_frame(frame: np.ndarray, 
                       posture_label: str, 
                       confidence: float,
                       keypoints: Dict[str, Tuple[float, float]] = None) -> np.ndarray:
        """
        Annotate frame with posture label above person's bounding box
        
        Args:
            frame: Input video frame (BGR format)
            posture_label: Detected posture label
            confidence: Confidence score (0-1)
            keypoints: Keypoints to determine person's bounding box location
            
        Returns:
            Annotated frame
        """
        # Get color for posture
        color = FrameAnnotator.POSTURE_COLORS.get(posture_label, (128, 128, 128))
        
        # Create label text
        text = f"{posture_label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get bounding box from keypoints
        if keypoints:
            valid_kpts = [kpt for kpt in keypoints.values() if kpt and kpt[0] > 0 and kpt[1] > 0]
            if valid_kpts:
                xs = [kpt[0] for kpt in valid_kpts]
                ys = [kpt[1] for kpt in valid_kpts]
                
                x_min = int(min(xs))
                y_min = int(min(ys))
                
                # Draw label at top-left of bounding box
                label_x = max(x_min, 10)
                label_y = max(y_min - 10, 25)
                
                cv2.putText(frame, text, (label_x, label_y), font, font_scale, color, thickness)
            FrameAnnotator._draw_keypoints(frame, keypoints)
        return frame
    
    @staticmethod
    def _draw_keypoints(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float]]) -> None:
        """Draw keypoints on frame"""
        
        # Keypoint radius and color
        radius = 5
        color = (0, 255, 0)
        thickness = -1
        
        # Draw midpoints (larger)
        midpoints = ['shoulder_mid', 'hip_mid', 'knee_mid', 'ankle_mid']
        for kpt_name in midpoints:
            kpt = keypoints.get(kpt_name)
            if kpt and kpt[0] > 0 and kpt[1] > 0:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius + 2, (0, 200, 200), thickness)
        
        # Draw skeleton connections
        connections = [
            ('shoulder_mid', 'hip_mid'),
            ('hip_mid', 'knee_mid'),
            ('knee_mid', 'ankle_mid'),
        ]
        
        for start_kpt, end_kpt in connections:
            start = keypoints.get(start_kpt)
            end = keypoints.get(end_kpt)
            if start and end and start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
                cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)
        
        # Draw individual keypoints
        individual_kpts = [
            'shoulder_left', 'shoulder_right',
            'hip_left', 'hip_right',
            'knee_left', 'knee_right',
            'ankle_left', 'ankle_right'
        ]
        
        for kpt_name in individual_kpts:
            kpt = keypoints.get(kpt_name)
            if kpt and kpt[0] > 0 and kpt[1] > 0:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius, (100, 200, 100), thickness)
