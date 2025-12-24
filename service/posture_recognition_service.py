"""
Posture Recognition Service
Main service for stream-based posture detection and annotation
"""

import cv2
from typing import Optional
from ultralytics import YOLO
import numpy as np

from config.posture_config import PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from utils.logger import get_logger
logger = get_logger(__name__)


class PoseRecognition:
    """Main posture recognition service for video streams"""
    
    def __init__(self, 
                 video_source: str,
                 source_id: str,
                 model_path: str = "yolov8l-pose.pt",
                 config: Optional[PostureConfig] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize posture recognition service
        
        Args:
            video_source: Video source (file path or RTSP stream)
            source_id: Unique identifier for this stream
            model_path: Path to YOLO pose model
            config: PostureConfig instance
            confidence_threshold: Confidence threshold for detections
        """
        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.config = config or PostureConfig()
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize processors
        self.keypoint_extractor = KeypointExtractor(self.config)
        self.classifier = PostureClassifier(self.config)
        self.frame_annotator = FrameAnnotator()
        
        # Stream handler (includes motion detection)
        self.stream_handler = StreamHandler(video_source)
        
        logger.info(f"Service initialized for source: {source_id}")
    
    def run(self):
        """Main service loop - process video stream continuously"""
        try:
            logger.info(f"Starting posture detection on source: {self.video_source}")
            
            # Start stream
            self.stream_handler.start_stream()
            
            # Get video properties
            fps = int(self.stream_handler.capture.get(cv2.CAP_PROP_FPS))
            width = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Stream opened - FPS: {fps}, Resolution: {width}x{height}")

            while True:
                # Read frame and motion detection from stream handler
                frame, motion_detected = self.stream_handler.read_frame()
                # (h, w) = frame.shape[:2]
                # center = (w // 2, h // 2)
                # M = cv2.getRotationMatrix2D(center, 10, 1.0)
                # frame = cv2.warpAffine(frame, M, (w, h))

                if frame is None:
                    logger.warning("End of stream reached")
                    break
                # Perform inference only on motion detected
                if motion_detected:
                    frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow(f"Posture Detection - Source {self.source_id}", frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.stream_handler.release_stream()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error in service loop: {str(e)}", exc_info=True)
        finally:
            cv2.destroyAllWindows()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame through entire pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame
        """
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Process each detected person
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    total_persons = len(keypoints_data)
                    
                    for idx, person_keypoints in enumerate(keypoints_data):
                        # Extract keypoints (YOLO returns [x, y, confidence] for each keypoint)
                        kpt_coords = person_keypoints[:, :2]
                        kpt_conf = person_keypoints[:, 2].tolist()  # Convert to list for proper indexing
                        
                        # Extract relevant keypoints
                        kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
                        
                        # Classify posture
                        label, confidence = self.classifier.classify(kpts)
                        
                        # Annotate frame
                        frame = self.frame_annotator.annotate_frame(
                            frame,
                            posture_label=label,
                            confidence=confidence,
                            keypoints=kpts
                        )
                        
                        logger.debug(f"Person {idx+1}/{total_persons}: {label} (confidence: {confidence:.2f})")
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return frame