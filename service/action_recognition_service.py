# service/action_recognition_service
import torch
import cv2
from time import sleep
from loaders.model_loader import ModelLoader
from loaders.classnames_loader import ClassNamesLoader
from utils.action_notification import ActionNotification
from utils.notification_manager import NotificationManager
from inference.inference_engine import InferenceEngine
from preprocessing.frame_buffer import FrameBuffer
from utils.visulaization import overlay_predictions
from stream.stream_handler import StreamHandler
from utils.expected_actions import VALID_KINETICS_ACTIONS
from utils.logger import get_logger
logger = get_logger(__name__)


class ActionRecognitionService:

    def __init__(self, model_name, video_source, stream_id, class_csv, device=None):
        self.model_name = model_name
        self.video_source = video_source
        self.stream_id = stream_id
        self.class_csv = class_csv

        if torch.cuda.is_available():
            self.device = torch.device(device if device else "cuda:0")
        else:
            self.device = torch.device("cpu")

        self.notification_manager = None
        self.expected_actions = VALID_KINETICS_ACTIONS
        self.model = None
        self.frame_window_size = None
        self.img_size = None
        self.preprocessor= None
        self.predictor= None

        logger.info(
            f"Initialized service object | model={model_name} | stream={stream_id} | source={video_source}"
        )

    def initialize(self):
        logger.info("Loading classnames...")
        self.classnames = ClassNamesLoader(self.class_csv).load()
        logger.info("Classnames loaded successfully.")

        logger.info(f"Loading model components for {self.model_name} on {self.device}...")
        loader = ModelLoader(self.model_name, self.device, self.classnames)

        (
            self.model,
            self.frame_window_size,
            self.img_size,
            self.preprocessor,
            self.predictor
        ) = loader.load()

        logger.info(
            f"Model loaded | Requires T frames: {self.frame_window_size} | Preprocessor: {type(self.preprocessor).__name__ if self.preprocessor else 'None'} "
            f"| Predictor: {type(self.predictor).__name__ if self.predictor else 'None'}"
        )

        # Frame buffer
        if self.frame_window_size:
            self.frame_buffer = FrameBuffer(self.frame_window_size)
            logger.debug(f"FrameBuffer initialized with size T={self.frame_window_size}")
        else:
            self.frame_buffer = None
            logger.debug("FrameBuffer not required for this model.")

        # Inference engine
        self.inference_engine = InferenceEngine(self.model, self.predictor, self.device)

        # Notification Manager
        self.notification_manager = NotificationManager(self.stream_id, self.expected_actions)
        logger.info("NotificationManager initialized.")

        # Video stream
        self.stream_handler = StreamHandler(self.video_source)
        logger.info(f"StreamHandler ready for source: {self.video_source}")

        logger.info("ActionRecognitionService fully initialized.")

    def run(self):
        logger.info("Starting ActionRecognitionService...")
        self.initialize()

        self.stream_handler.start_stream()
        logger.info(f"Stream started for {self.video_source}")

        while True:
            frame, motion_flag = self.stream_handler.read_frame()
            source_fps = int(self.stream_handler.capture.get(cv2.CAP_PROP_FPS))

            if frame is None:
                logger.debug("Received empty frame. Skipping...")
                sleep(0.01)
                continue

            if not motion_flag:
                logger.debug("Motion flag false. Skipping frame.")
                sleep(0.01)
                continue

            # Preprocessing
            try:
                processed = (
                    self.preprocessor.preprocess_frame(frame)
                    if self.preprocessor else frame
                )
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                continue

            if self.frame_buffer:
                self.frame_buffer.add_frame(processed)

                if not self.frame_buffer.is_full():
                    logger.debug("FrameBuffer not ready yet. Waiting...")
                    continue

                try:
                    tensor = self.preprocessor.make_tensor(self.frame_buffer.get_frames())
                except Exception as e:
                    logger.error(f"Failed to create tensor from buffer: {e}")
                    continue

            # Prediction
            preds, infer_t = self.inference_engine.run(tensor)
            infer_fps = 1 / infer_t if infer_t else 0
            logger.debug(f"Inference completed | FPS={infer_fps}")

            # Annotate
            annotated = overlay_predictions(frame.copy(), preds, source_fps,infer_fps)

            # Notification logic
            snap = self.notification_manager.update(preds, annotated)
            if snap:
                logger.info(f"Notification captured for stream {self.stream_id}")
                ActionNotification(self.stream_id, snap).register()

            # Display output
            cv2.imshow(f"Action Recognition - {self.video_source}", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit signal received. Stopping service.")
                break

        self.cleanup()

    # Cleanup

    def cleanup(self):
        logger.info("Cleaning up resources...")
        try:
            if self.stream_handler:
                self.stream_handler.release_stream()
                logger.info("Stream handler released.")
        except Exception as e:
            logger.error(f"Error during stream release: {e}")

        cv2.destroyAllWindows()
        logger.info(f"Service stopped for stream  {self.video_source}")
