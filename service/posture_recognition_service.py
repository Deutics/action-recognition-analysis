"""
Posture Recognition Service - Streamlined
"""

import cv2
from typing import Optional
from ultralytics import YOLO
import numpy as np
import torch
from config.posture_config import PostureConfig
from preprocessing.keypoint_extractor import KeypointExtractor
from core.classifier import PostureClassifier
from service.frame_annotator import FrameAnnotator
from stream.stream_handler import StreamHandler
from tracking.person_tracker import PersonTracker
from notification.notification_handler import NotificationHandler
from utils.logger import get_logger
import threading
import queue

logger = get_logger(__name__)


class PoseRecognition:
    """Posture recognition service with threading"""

    def __init__(self,
                 video_source: str,
                 source_id: str,
                 model_path: str = "yolo11n-pose.pt",
                 config: Optional[PostureConfig] = None,
                 confidence_threshold: float = 0.5,
                 frame_buffer_size: int = 3,
                 result_buffer_size: int = 3):

        self.video_source = video_source
        self.source_id = source_id
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or PostureConfig()

        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Initialize components
        self.keypoint_extractor = KeypointExtractor(self.config)
        self.classifier = PostureClassifier(self.config)
        self.frame_annotator = FrameAnnotator()
        self.person_tracker = PersonTracker(lying_threshold=self.config.lying_duration_threshold)
        self.notification_handler = NotificationHandler()
        self.stream_handler = StreamHandler(video_source)

        # Threading
        self.frame_queue = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue = queue.Queue(maxsize=result_buffer_size)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.processor_thread = None

        logger.info(f"Service initialized for source: {source_id}")

    def _frame_reader_worker(self):
        """Read frames from stream"""
        try:
            logger.info("Frame reader thread started")
            while not self.stop_event.is_set():
                frame, motion_detected = self.stream_handler.read_frame()

                if frame is None:
                    logger.warning("End of stream reached in reader")
                    self.stop_event.set()
                    break

                try:
                    self.frame_queue.put((frame, motion_detected), timeout=0.1)
                except queue.Full:
                    logger.debug("Frame queue full, dropping frame")

        except Exception as e:
            logger.error(f"Error in frame reader thread: {str(e)}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("Frame reader thread stopped")

    def _frame_processor_worker(self):
        """Process frames"""
        try:
            logger.info("Frame processor thread started")
            while not self.stop_event.is_set():
                try:
                    frame, motion_detected = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                processed_frame = self._process_frame(frame) if motion_detected else frame

                try:
                    self.result_queue.put(processed_frame, timeout=0.01)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(processed_frame, timeout=0.01)
                    except:
                        pass

                self.frame_queue.task_done()

        except Exception as e:
            logger.error(f"Error in frame processor thread: {str(e)}", exc_info=True)
            self.stop_event.set()
        finally:
            logger.info("Frame processor thread stopped")

    def run(self):
        """Main service loop"""
        try:
            logger.info(f"Starting posture detection on source: {self.video_source}")
            self.stream_handler.start_stream()

            fps = int(self.stream_handler.capture.get(cv2.CAP_PROP_FPS))
            width = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.stream_handler.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Stream opened - FPS: {fps}, Resolution: {width}x{height}")

            # Start threads
            self.reader_thread = threading.Thread(target=self._frame_reader_worker, daemon=True)
            self.processor_thread = threading.Thread(target=self._frame_processor_worker, daemon=True)
            self.reader_thread.start()
            self.processor_thread.start()

            while not self.stop_event.is_set():
                try:
                    frame = self.result_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                frame = cv2.resize(frame, (640, 480))
                cv2.imshow(f"Posture Detection - Source {self.source_id}", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    break

            self._cleanup()

        except Exception as e:
            logger.error(f"Error in service loop: {str(e)}", exc_info=True)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up threads and resources"""
        logger.info("Cleaning up resources...")
        self.stop_event.set()

        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)

        self.stream_handler.release_stream()
        cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame - EXACT same logic as original"""
        try:
            raw_frame = frame.copy()
            display_frame = frame.copy()
            frame_h = frame.shape[0]

            # Run YOLO tracking
            results = self.model.track(frame, conf=self.confidence_threshold, persist=True, verbose=False)

            if not results or len(results) == 0:
                return display_frame

            result = results[0]
            if result.keypoints is None:
                return display_frame

            keypoints_data = result.keypoints.data.cpu().numpy()

            # Extract tracking IDs (EXACTLY as original)
            track_ids = []
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)

            active_person_ids = set(track_ids)

            # Process each person
            persons = []
            for idx, person_keypoints in enumerate(keypoints_data):
                person_id = track_ids[idx] if idx < len(track_ids) else idx

                kpt_coords = person_keypoints[:, :2]
                kpt_conf = person_keypoints[:, 2].tolist()

                kpts = self.keypoint_extractor.extract(kpt_coords, kpt_conf)
                label, confidence = self.classifier.classify(kpts, frame_h)

                persons.append({
                    "id": person_id,
                    "label": label,
                    "confidence": confidence,
                    "keypoints": kpts
                })

            # Track falling events
            falling_person_ids = set()
            for p in persons:
                if self.person_tracker.update(p["id"], p["label"]):
                    falling_person_ids.add(p["id"])

            # Annotate all persons on display frame
            for p in persons:
                display_frame = self.frame_annotator.annotate_frame(
                    display_frame,
                    posture_label=p["label"],
                    confidence=p["confidence"],
                    keypoints=p["keypoints"],
                    person_id=p["id"]
                )

            # Save falling notifications
            if falling_person_ids:
                for p in persons:
                    if p["id"] in falling_person_ids:
                        event_frame = raw_frame.copy()
                        event_frame = self.frame_annotator.annotate_frame(
                            event_frame,
                            posture_label="Falling",
                            confidence=p["confidence"],
                            keypoints=p["keypoints"],
                            person_id=p["id"]
                        )
                        self.notification_handler.save_falling_notification(
                            person_id=p["id"],
                            frame=event_frame,
                            source_id=self.source_id
                        )

            # Cleanup inactive persons
            self.person_tracker.cleanup(active_person_ids)

            return display_frame  # FIX: Return annotated frame, not raw frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return frame