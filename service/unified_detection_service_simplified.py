"""
Simplified Unified Detection Service
Clean separation of concerns without changing functionality
"""
import cv2
from typing import Optional, List, Dict, Tuple
from ultralytics import YOLO
import numpy as np
import torch
import time

from config.detection_config import ForceIntentConfig, PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.posture.posture_classifier import PostureClassifier
from core.breakin import BreakinClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from tracking.person_tracker import PersonTracker as  PersonTracker
from notification.notification_handler import NotificationHandler
from visualization.roi_renderer import DoorROIRenderer
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedDetectionService:
    """Main detection service orchestrating posture and force detection."""

    def __init__(self, video_source: str, source_id: str, model_path: str = "yolo26n-pose.pt",
                 posture_config: Optional[PostureConfig] = None,
                 force_intent_config: Optional[ForceIntentConfig] = None,
                 confidence_threshold: float = 0.5, infer_every_n_frames: int = 1,
                 enable_posture: bool = True, enable_force_intent: bool = True):

        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        self.infer_every_n_frames = max(1, infer_every_n_frames)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Configs
        self.posture_config = posture_config or PostureConfig()
        self.force_config = force_intent_config or ForceIntentConfig()
        self.enable_posture = enable_posture
        self.enable_force = enable_force_intent and self.force_config.enabled

        # Initialize components
        self._init_components(model_path)
        
        # State
        self.frame_index = 0
        self.last_persons = []
        self.door_center = None
        self.frame_dims = None

    def _init_components(self, model_path: str):
        """Initialize all detection components."""
        # YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Keypoint extraction
        self.keypoint_extractor = KeypointExtractor(self.posture_config)
        
        # Classifiers
        self.posture_classifier = PostureClassifier(self.posture_config)
        self.breakin_classifier = BreakinClassifier(self.force_config, max_distance_to_door=600.0)
        
        # Tracker
        self.tracker = PersonTracker(
            lying_threshold=self.posture_config.lying_duration_threshold,
            force_intent_cooldown=self.force_config.thresholds.cooldown_period,
            energy_threshold=self.force_config.thresholds.energy_threshold,
            min_force_frames=self.force_config.thresholds.min_force_frames,
            decay_per_frame=self.force_config.thresholds.decay_per_frame,
            temporal_window=self.force_config.thresholds.temporal_window,
            frame_score_threshold=self.force_config.thresholds.frame_threshold
        )
        
        # Handlers
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(self.video_source)
        self.frame_annotator = FrameAnnotator()
        self.roi_renderer = DoorROIRenderer()
    
    def run(self):
        """Main detection loop."""
        try:
            # Start stream and setup
            self.stream_handler.start_stream()
            video_writer = self._setup_video_and_door()
            
            # Main loop
            while True:
                frame, motion_detected = self.stream_handler.read_frame()
                if frame is None:
                    break
                
                # Extract keypoints on inference frames
                if motion_detected and (self.frame_index % self.infer_every_n_frames == 0):
                    self.last_persons = self._detect_and_extract(frame)
                
                # Classify all detections
                if self.last_persons:
                    self._classify_all_persons(frame.shape[0])
                
                # Process detections and annotate frame
                frame = self._process_and_annotate(frame)
                
                # Add door ROI visualization
                if self.enable_force and self.door_center:
                    frame = self.roi_renderer.render(
                        frame, self.force_config.door_roi, 
                        self.frame_dims[0], self.frame_dims[1]
                    )
                
                # Display
                # video_writer.write(frame)
                self._display_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                self.frame_index += 1
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}", exc_info=True)
        finally:
            video_writer.release()
            self._cleanup()
    
    def _setup_video_and_door(self) -> cv2.VideoWriter:
        """Setup video writer and calculate door center (separated for clarity)."""
        # Get video properties
        frame_width = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.stream_handler.capture.get(cv2.CAP_PROP_FPS))
        self.frame_dims = (frame_width, frame_height)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            'processed_output.mp4', fourcc, fps, (frame_width, frame_height)
        )
        
        # Calculate door center if force detection enabled
        if self.enable_force and self.force_config.door_roi:
            self.door_center = self.roi_renderer.calculate_door_center(
                self.force_config.door_roi, frame_width, frame_height
            )
            logger.info(f"Door center: {self.door_center}")
        
        return video_writer
    
    def _detect_and_extract(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons and extract keypoints."""
        persons = []
        
        # Run YOLO tracking
        results = self.model.track(
            frame, conf=self.confidence_threshold, persist=True, verbose=False
        )
        
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return persons
        
        # Extract keypoints for each detected person
        result = results[0]
        keypoints_data = result.keypoints.data.cpu().numpy()
        track_ids = (result.boxes.id.cpu().numpy().astype(int).tolist()
                    if result.boxes.id is not None else [])
        
        for idx, person_keypoints in enumerate(keypoints_data):
            if person_keypoints.ndim != 2 or person_keypoints.shape[1] < 2:
                continue
            
            person_id = track_ids[idx] if idx < len(track_ids) else idx
            kpt_coords = person_keypoints[:, :2]
            kpt_conf = (person_keypoints[:, 2].tolist()
                       if person_keypoints.shape[1] >= 3 else None)
            
            kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
            if kpts is None:
                continue
            
            persons.append({
                'id': person_id,
                'keypoints': kpts,
                'timestamp': time.time()
            })
        
        return persons
    
    def _classify_all_persons(self, frame_height: int):
        """Classify posture and force intent for all detected persons."""
        # Posture classification
        if self.enable_posture:
            for person in self.last_persons:
                label, confidence = self.posture_classifier.classify(
                    person['keypoints'], frame_height
                )
                person['posture_label'] = label
                person['posture_confidence'] = confidence
        
        # Force intent classification
        if self.enable_force and self.door_center:
            for person in self.last_persons:
                signal_histories = self.tracker.get_signal_histories(person['id'])
                label, score = self.breakin_classifier.classify(
                    person['keypoints'], self.door_center, signal_histories
                )
                person['breakin_label'] = label
                person['breakin_score'] = score
    
    def _process_and_annotate(self, frame: np.ndarray) -> np.ndarray:
        """Process detections, trigger alerts, and annotate frame."""
        active_ids = set()
        falling_ids = set()
        
        # Process posture tracking
        if self.enable_posture:
            for person in self.last_persons:
                pid = person['id']
                active_ids.add(pid)
                if self.tracker.update(pid, person.get('posture_label', 'Unknown')):
                    falling_ids.add(pid)
        
        # Process force tracking
        if self.enable_force and self.door_center:
            for person in self.last_persons:
                pid = person['id']
                classification = (
                    person.get('breakin_label', 'NORMAL'),
                    person.get('breakin_score', 0.0)
                )
                
                if self.tracker.update_force(pid, classification):
                    self._handle_force_alert(pid, person, frame)
        
        # Cleanup stale persons
        self.tracker.cleanup(active_ids)
        
        # Annotate frame with posture labels
        for person in self.last_persons:
            pid = person['id']
            label = 'Falling' if pid in falling_ids else person.get('posture_label', 'Unknown')
            frame = self.frame_annotator.annotate_frame(
                frame, 
                posture_label=label,
                confidence=person.get('posture_confidence', 0.0),
                keypoints=person.get('keypoints'),
                person_id=pid
            )
        
        # Trigger falling alerts
        for pid in falling_ids:
            logger.warning(f"FALLING ALERT: Person {pid}")
            self.notification_handler.save_falling_notification(pid, frame, self.source_id)
        
        return frame
    
    def _handle_force_alert(self, person_id: int, person: Dict, frame: np.ndarray):
        """Handle force intent alert (separated for clarity)."""
        logger.warning(f"FORCE ALERT: Person {person_id}")
        
        frame = self.frame_annotator.annotate_door_breakin_alert(
            frame, person_id, person['breakin_score'], max_score=10.0
        )
        
        self.notification_handler.save_force_intent_notification(
            person_id, frame, self.source_id,
            {'label': 'FORCE', 'confidence': person['breakin_score']}
        )
    
    def _display_frame(self, frame: np.ndarray):
        """Display frame in window."""
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow(f"Detection - {self.source_id}", display_frame)
    
    def _cleanup(self):
        """Release all resources."""
        try:
            self.stream_handler.release_stream()
            cv2.destroyAllWindows()
            if hasattr(self.model, 'to'):
                self.model.cpu()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
