"""Unified detection service orchestrating posture and force detection."""
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
# from tracking.person_tracker import PersonTracker
from tracking.person_tracker import PersonTracker
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

        self.posture_config = posture_config or PostureConfig()
        self.force_config = force_intent_config or ForceIntentConfig()
        self.enable_posture = enable_posture
        self.enable_force = enable_force_intent and self.force_config.enabled

        self._init_components(model_path)
        self.frame_index = 0
        self.last_persons = []

    def _init_components(self, model_path: str):
        """Initialize all detection components."""
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.keypoint_extractor = KeypointExtractor(self.posture_config)
        self.posture_classifier = PostureClassifier(self.posture_config)
        self.breakin_classifier = BreakinClassifier(
            self.force_config, 
            max_distance_to_door=600.0  # CRITICAL: Distance gating threshold
        )
        self.tracker = PersonTracker(
            lying_threshold=self.posture_config.lying_duration_threshold,
            force_intent_cooldown=self.force_config.thresholds.cooldown_period,
            energy_threshold=self.force_config.thresholds.energy_threshold,
            min_force_frames=self.force_config.thresholds.min_force_frames,
            decay_per_frame=self.force_config.thresholds.decay_per_frame,
            temporal_window=self.force_config.thresholds.temporal_window,
            frame_score_threshold=self.force_config.thresholds.frame_threshold  # Use same threshold
        )
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(self.video_source)
        self.frame_annotator = FrameAnnotator()
        self.roi_renderer = DoorROIRenderer()
    
    def run(self):
        """Main detection loop."""
        try:
            self.stream_handler.start_stream()
            video_writer, frame_dims, door_center = self._setup_video_output()
            
            while True:
                frame, motion_detected = self.stream_handler.read_frame()
                if frame is None:
                    break
                
                if motion_detected and (self.frame_index % self.infer_every_n_frames == 0):
                    self.last_persons = self._extract_keypoints(frame)
                
                # Classify all detections
                if self.last_persons:
                    self._classify_detections(frame.shape[0], door_center)
                
                # Process detections
                frame = self._process_detections(frame, door_center)
                
                if self.enable_force and door_center:
                    frame = self.roi_renderer.render(
                        frame, self.force_config.door_roi, frame_dims[0], frame_dims[1])
                
                # video_writer.write(frame)
                display_frame = cv2.resize(frame, (640, 480))
                cv2.imshow(f"Detection - {self.source_id}", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                self.frame_index += 1
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}", exc_info=True)
        finally:
            video_writer.release()
            self._cleanup()
    
    def _setup_video_output(self):
        """Setup video writer and calculate door center."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.stream_handler.capture.get(cv2.CAP_PROP_FPS))
        
        writer = cv2.VideoWriter('processed_output.mp4', fourcc, fps, 
                                (frame_width, frame_height))
        
        door_center = None
        if self.enable_force and self.force_config.door_roi:
            door_center = self.roi_renderer.calculate_door_center(
                self.force_config.door_roi, frame_width, frame_height)
            logger.info(f"Door center: {door_center}")
        
        return writer, (frame_width, frame_height), door_center
    
    def _extract_keypoints(self, frame: np.ndarray) -> List[Dict]:
        """Extract keypoints only - classification done separately."""
        persons = []
        results = self.model.track(frame, conf=self.confidence_threshold,
                                  persist=True, verbose=False)

        if not results or results[0].keypoints is None or results[0].boxes is None:
            return persons

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

            # Only store keypoints - classification happens separately
            persons.append({
                'id': person_id,
                'keypoints': kpts,
                'timestamp': time.time()
            })
        
        return persons
    
    def _classify_detections(self, frame_height: int, door_center: Optional[Tuple]) -> None:
        """Classify both posture and break-in for all detected persons."""
        
        # 1. Posture classification
        for person in self.last_persons:
            label, confidence = self.posture_classifier.classify(
                person['keypoints'], frame_height
            )
            person['posture_label'] = label
            person['posture_confidence'] = confidence
        
        # 2. Break-in classification (only if force enabled and door_center available)
        if self.enable_force and door_center:
            for person in self.last_persons:
                # Get signal histories from tracker
                signal_histories = self.tracker.get_signal_histories(person['id'])
                
                # Classify force intent (includes distance gating)
                label, score = self.breakin_classifier.classify(
                    person['keypoints'], door_center, signal_histories
                )
                person['breakin_label'] = label
                person['breakin_score'] = score
    
    def _process_detections(self, frame: np.ndarray, door_center) -> np.ndarray:
        """Process both posture and force detections."""
        
        # Step 1: Process posture tracking
        active_ids = set()
        falling_ids = set()

        if self.enable_posture:
            for person in self.last_persons:
                pid = person['id']
                active_ids.add(pid)
                
                # Update tracker with classification result
                if self.tracker.update(pid, person.get('posture_label', 'Unknown')):
                    falling_ids.add(pid)
        
        # Step 2: Process force tracking
        if self.enable_force and door_center:
            for person in self.last_persons:
                pid = person['id']
                
                # Update tracker with classification result
                classification_result = (
                    person.get('breakin_label', 'NORMAL'),
                    person.get('breakin_score', 0.0)
                )
                
                if self.tracker.update_force(pid, classification_result):
                    # Trigger force alert
                    logger.warning(f"FORCE ALERT: Person {pid}")
                    frame = self.frame_annotator.annotate_door_breakin_alert(
                        frame, pid, person['breakin_score'], max_score=10.0
                    )
                    self.notification_handler.save_force_intent_notification(
                        pid, frame, self.source_id,
                        {'label': 'FORCE', 'confidence': person['breakin_score']}
                    )
        
        # Step 3: Cleanup stale persons
        self.tracker.cleanup(active_ids)
        
        # Step 4: Annotate frame with posture
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
        
        # Step 5: Trigger falling alerts
        for pid in falling_ids:
            logger.warning(f"FALLING ALERT: Person {pid}")
            self.notification_handler.save_falling_notification(
                pid, frame, self.source_id
            )
        
        return frame

    def _cleanup(self):
        """Release all resources."""
        try:
            self.stream_handler.release_stream()
            cv2.destroyAllWindows()
            if hasattr(self.model, 'to'):
                self.model.cpu()
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
