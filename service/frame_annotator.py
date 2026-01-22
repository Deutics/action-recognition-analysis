"""
Frame Annotator
Annotate video frames with posture detection results
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


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
        "Insufficient_Keypoints": (192, 192, 192), # Light Gray
        "Falling": (0, 0, 255),
    }
    
    @staticmethod
    def annotate_frame(frame: np.ndarray, 
                       posture_label: str, 
                       confidence: float,
                       person_id: Optional[int] = None,
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
        text = f"ID:{person_id} {posture_label} ({confidence:.2f})"
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
        midpoints = ['shoulder_mid', 'elbow_mid', 'hip_mid', 'knee_mid', 'ankle_mid']
        for kpt_name in midpoints:
            kpt = keypoints.get(kpt_name)
            if kpt and kpt[0] > 0 and kpt[1] > 0:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius + 2, (0, 200, 200), thickness)
        
        # Draw skeleton connections (including arms)
        connections = [
            ('shoulder_mid', 'elbow_mid'),  # Upper arms
            ('elbow_mid', 'hip_mid'),        # Torso
            ('hip_mid', 'knee_mid'),         # Thighs
            ('knee_mid', 'ankle_mid'),       # Shins
        ]
        
        for start_kpt, end_kpt in connections:
            start = keypoints.get(start_kpt)
            end = keypoints.get(end_kpt)
            if start and end and start[0] > 0 and start[1] > 0 and end[0] > 0 and end[1] > 0:
                cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)
        
        # Draw individual elbows (distinct color for visibility)
        for elbow_name in ['elbow_left', 'elbow_right']:
            elbow = keypoints.get(elbow_name)
            if elbow and elbow[0] > 0 and elbow[1] > 0:
                cv2.circle(frame, (int(elbow[0]), int(elbow[1])), radius, (255, 0, 0), thickness)
        
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
    
    @staticmethod
    def annotate_door_breakin_alert(frame: np.ndarray, person_id: int, force_score: float, max_score: float = 10.0) -> np.ndarray:
        """
        Add prominent 'DOOR BREAKIN' alert label to frame.
        
        Args:
            frame: Input video frame
            person_id: ID of person triggering alert
            force_score: Current accumulated force score
            max_score: Maximum force score for display
            
        Returns:
            Annotated frame with alert
        """
        h, w = frame.shape[:2]
        
        # Draw red border around entire frame to indicate alert
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        
        # Draw large alert text at top-center
        alert_text = f"*** DOOR BREAKIN DETECTED ***"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = 40
        
        # Draw background box for text
        cv2.rectangle(frame, (x - 10, y - 30), (x + text_size[0] + 10, y + 10), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (x, y), font, font_scale, (0, 255, 255), thickness)
        
        # Draw person ID and force score below alert
        detail_text = f"Person ID: {person_id} | Force Score: {force_score:.2f}/{max_score}"
        font_scale_detail = 0.9
        thickness_detail = 2
        text_size_detail = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, thickness_detail)[0]
        x_detail = (w - text_size_detail[0]) // 2
        y_detail = y + 50
        
        cv2.rectangle(frame, (x_detail - 10, y_detail - 25), (x_detail + text_size_detail[0] + 10, y_detail + 5), (0, 0, 255), -1)
        cv2.putText(frame, detail_text, (x_detail, y_detail), cv2.FONT_HERSHEY_SIMPLEX, font_scale_detail, (0, 255, 255), thickness_detail)
        
        return frame

