import torch
import cv2
from time import sleep
from loaders.model_loader import ModelLoader
from loaders.classnames_loader import ClassNamesLoader
from preprocessing.frame_preprocessor import FramePreprocessor
from inference.predictor import Predictor
from stream.stream_handler import StreamHandler
from utils.visulaization import overlay_predictions


class ActionRecognitionService:
    """
     Action Recognition Service.
    - Works with ANY model supported by ModelLoader.
    - Safe for multiprocessing on Windows (no lambdas, no closures).
    - Fully modular and follows clean architecture.
    """

    def __init__(self, model_name, video_source, class_csv, device=None):
        self.device = device or torch.device("cpu")
        self.model_name = model_name
        print(f"Loaded model {model_name} for video source {video_source}")
        self.video_source = video_source
        self.class_csv = class_csv

        self.model = None
        self.T = None
        self.img_size = None
        self.classnames = None
        self.preprocessor = None
        self.predictor = None
        self.stream_handler = None
        self.buffer = []

    # ---------------------------
    # Initialization Phase
    # ---------------------------
    def initialize(self):
        """Initialize all components. Separated for multiprocessing safety."""

        # Load model
        model_loader = ModelLoader(self.model_name, self.device)
        self.model, self.T, self.img_size = model_loader.load()

        # Load classnames
        self.classnames = ClassNamesLoader(self.class_csv).load()

        # Preprocessor & Predictor
        self.preprocessor = FramePreprocessor(self.img_size)
        self.predictor = Predictor(self.model, self.classnames, self.device)

        # Stream Handler
        self.stream_handler = StreamHandler(self.video_source)

        self.buffer = []
        print(f"[INFO] Initialized ActionRecognitionService for → {self.video_source}")

    # ---------------------------
    # Main Run Loop
    # ---------------------------
    def run(self):
        """
        Run real-time inference.
        This is the method that multiprocessing calls → target=service.run
        """
        try:
            self.initialize()
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return

        self.stream_handler.start()
        print(f"[INFO] Running live inference on → {self.video_source}")

        while True:
            frame = self.stream_handler.read_frame()

            if frame is None:
                sleep(0.01)
                continue

            processed = self.preprocessor.preprocess_frame(frame)
            self.buffer.append(processed)

            if len(self.buffer) > self.T:
                self.buffer.pop(0)

            # Perform prediction
            if len(self.buffer) == self.T:
                tensor = self.preprocessor.make_tensor(self.buffer)
                preds, infer_t = self.predictor.predict(tensor)
                fps = 1 / infer_t if infer_t > 0 else 0

                frame = overlay_predictions(frame, preds, fps)

            cv2.imshow(f"Action Recognition - {self.video_source}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    # ---------------------------
    # Cleanup
    # ---------------------------
    def cleanup(self):
        try:
            if self.stream_handler:
                self.stream_handler.release()
        except:
            pass

        cv2.destroyAllWindows()
        print(f"[INFO] Stopped stream → {self.video_source}")
