"""
Interactive ROI Selector Module
Handles all door ROI selection logic.
Decoupled from main.py for better modularity.
"""

import cv2
from config.detection_config import DoorROIConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Constants for ROI selection
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
BLACK_FRAME_THRESHOLD = 10
MAX_FRAME_SEARCH = 300


class ROISelector:
    """Interactive ROI selector for door barriers"""
    
    @staticmethod
    def select_door_roi(video_source: str, source_id: str) -> DoorROIConfig:
        """
        Interactive door ROI selector.
        
        Returns ROI configured at DISPLAY resolution (640x480), 
        which matches the actual processing resolution.
        
        Args:
            video_source: Video source path or camera ID
            source_id: Source identifier
            
        Returns:
            DoorROIConfig normalized to 640x480 space
        """
        logger.info(f"[ROI Selector] Opening {source_id}")
        logger.info("Instructions: Left-click to set corners | Right-click to reset | SPACE to confirm")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"[ROI Selector] Failed to open {video_source}")
            return DoorROIConfig()
        
        # Find first non-black frame
        frame = ROISelector._find_good_frame(cap)
        if frame is None:
            logger.error("[ROI Selector] Could not find non-black frame")
            cap.release()
            return DoorROIConfig()
        
        cap.release()
        
        # Get user clicks
        points = ROISelector._get_roi_points(frame, source_id)
        if not points or len(points) < 2:
            logger.warning("[ROI Selector] Cancelled - using default ROI")
            return DoorROIConfig()
        
        # Convert to normalized coordinates
        return ROISelector._convert_to_roi_config(points)
    
    @staticmethod
    def _find_good_frame(cap):
        """Find first non-black frame"""
        for attempt in range(MAX_FRAME_SEARCH):
            ret, raw_frame = cap.read()
            if not ret:
                return None
            
            # Resize to display resolution
            frame = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Check if frame has content (not black)
            if cv2.mean(frame)[0] > BLACK_FRAME_THRESHOLD:
                logger.info(f"[ROI Selector] Found good frame at attempt {attempt + 1}")
                return frame
            
            if attempt % 30 == 0:
                logger.info(f"[ROI Selector] Searching for frame... ({attempt})")
        
        return None
    
    @staticmethod
    def _get_roi_points(frame, source_id: str):
        """Get ROI points via mouse clicks"""
        points = []
        display_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                logger.info(f"[ROI Selector] Point {len(points)}: ({x}, {y})")
                
                # Redraw
                display_frame = frame.copy()
                for i, pt in enumerate(points):
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                points = []
                display_frame = frame.copy()
                logger.info("[ROI Selector] Points reset")
        
        cv2.namedWindow(f"ROI Selector - {source_id}", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(f"ROI Selector - {source_id}", mouse_callback)
        
        logger.info("Waiting for door ROI definition...")
        
        while True:
            cv2.imshow(f"ROI Selector - {source_id}", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE
                if len(points) >= 2:
                    logger.info(f"[ROI Selector] Confirmed {len(points)} points")
                    break
                else:
                    logger.warning("[ROI Selector] Need at least 2 points!")
            elif key == 27:  # ESC
                logger.warning("[ROI Selector] Cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return points
    
    @staticmethod
    def _convert_to_roi_config(points) -> DoorROIConfig:
        """Convert pixel points to normalized ROI config"""
        # Normalize to [0, 1] range at 640x480 resolution
        norm_points = [(x / DISPLAY_WIDTH, y / DISPLAY_HEIGHT) for x, y in points]
        
        xs = [p[0] for p in norm_points]
        ys = [p[1] for p in norm_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Convert to center-based format
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        config = DoorROIConfig(
            x_center_ratio=x_center,
            y_center_ratio=y_center,
            width_ratio=width,
            height_ratio=height
        )
        
        logger.info(f"[ROI Selector] ROI Config: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}x{height:.3f})")
        
        return config
