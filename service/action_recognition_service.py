# service/action_recognition_service
import torch
import cv2
from time import sleep
from loaders.model_loader import ModelLoader
from loaders.classnames_loader import ClassNamesLoader
from utils.action_notification import ActionNotification
from utils.notification_manager import NotificationManager
from utils.visulaization import overlay_predictions
from stream.stream_handler import StreamHandler
from utils.expected_actions import VALID_KINETICS_ACTIONS


class ActionRecognitionService:
    """
    Fully model-agnostic inference service.
    Supports:
        - models with OR without preprocessors
        - models with OR without predictors
        - models requiring temporal buffers (T) OR single-frame models
    """

    def __init__(self, model_name, video_source,stream_id, class_csv, device=None):
        self.notification_manager = None
        self.expected_actions = VALID_KINETICS_ACTIONS
        if torch.cuda.is_available():
            self.device = torch.device(device if device else "cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model_name = model_name
        self.video_source = video_source
        self.stream_id = stream_id
        self.class_csv = class_csv

        # Dynamic components (loader decides what is needed)
        self.model = None
        self.T = None               # Number of frames required (SlowFast=32, others=None)
        self.img_size = None
        self.classnames = None
        self.preprocessor = None    # Can be None
        self.predictor = None       # Can be None
        self.stream_handler = None

        self.buffer = []            # Used ONLY when T is not None

    # ---------------------------------------------------
    # Initialization Phase
    # ---------------------------------------------------
    def initialize(self):
        """ Initialize everything safely for multiprocessing. """

        #Notification Manager
        self.notification_manager = NotificationManager(self.stream_id,self.expected_actions)

        # Load class names
        self.classnames = ClassNamesLoader(self.class_csv).load()

        # Load model + components
        loader = ModelLoader(self.model_name, self.device, self.classnames)
        self.model, self.T, self.img_size, self.preprocessor, self.predictor = loader.load()

        # Stream
        self.stream_handler = StreamHandler(self.video_source)

        # Reset temporal buffer
        self.buffer = []

        print(f"[INFO] Initialized ActionRecognitionService for → {self.video_source}")
        print(f"[INFO] Model: {self.model_name}")
        print(f"[INFO] Requires T frames: {self.T}")
        print(f"[INFO] Preprocessor: {type(self.preprocessor).__name__ if self.preprocessor else 'None'}")
        print(f"[INFO] Predictor: {type(self.predictor).__name__ if self.predictor else 'None'}")

    # ---------------------------------------------------
    # Main Run Loop
    # ---------------------------------------------------
    def run(self):

        try:
            self.initialize()
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return

        self.stream_handler.start()
        print(f"[INFO] Running live inference on → {self.video_source}")

        while True:
            frame, motion_flag = self.stream_handler.read_frame()

            if frame is None or motion_flag is False:
                sleep(0.01)
                continue

            # ---------------------------------------------
            # 1. Preprocess frame if needed
            # ---------------------------------------------
            if self.preprocessor is not None:
                processed = self.preprocessor.preprocess_frame(frame)
            else:
                processed = frame  # raw frame passed directly

            # ---------------------------------------------
            # 2. Handle model types
            # ---------------------------------------------
            # ------------------------------------------------
            # MODELS THAT DO NOT REQUIRE TEMPORAL BUFFERS
            # ------------------------------------------------
            if self.T is None:

                if self.preprocessor:
                    tensor = self.preprocessor.make_tensor([processed])
                else:
                    # processed = (H,W,C), convert manually
                    tensor = (
                        torch.from_numpy(processed)
                        .permute(2, 0, 1)  # -> (C,H,W)
                        .unsqueeze(0)  # -> (1,C,H,W)
                        .unsqueeze(2)  # -> (1,C,1,H,W)
                        .float()
                    )

                preds, infer_t = self._run_prediction(tensor)

                fps = 1 / infer_t if infer_t > 0 else 0
                frame = overlay_predictions(frame, preds, fps)


            else:
                # ------------------------------------------------
                # MODELS THAT REQUIRE TEMPORAL BUFFER (T FRAMES)
                # ------------------------------------------------
                self.buffer.append(processed)

                # Keep fixed-size buffer
                if len(self.buffer) > self.T:
                    self.buffer.pop(0)

                if len(self.buffer) == self.T:
                    tensor = self.preprocessor.make_tensor(self.buffer)
                    preds, infer_t = self._run_prediction(tensor)

                    fps = 1 / infer_t if infer_t > 0 else 0
                    annotated_frame = overlay_predictions(frame, preds, fps)

                    snap = self.notification_manager.update(preds, annotated_frame)

                    if snap is not None:
                        notification = ActionNotification(self.stream_id, snap)
                        notification.register()

            # ---------------------------------------------
            # 3. Show result
            # ---------------------------------------------
            cv2.imshow(f"Action Recognition - {self.video_source}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    # ---------------------------------------------------
    # Unified prediction handler
    # ---------------------------------------------------
    def _run_prediction(self, tensor):
        """Run prediction using predictor or direct model forward."""

        # Handle multi-pathway inputs (e.g., SlowFast)
        if isinstance(tensor, list):
            tensor = [t.to(self.device) for t in tensor]
        else:
            tensor = tensor.to(self.device)

        # If a predictor exists → use it
        if self.predictor:
            return self.predictor.predict(tensor)

        # ELSE → raw model forward
        with torch.no_grad():
            import time
            t0 = time.time()
            out = self.model(tensor)
            infer_t = time.time() - t0
            preds = out
            return preds, infer_t

    # ---------------------------------------------------
    # Cleanup
    # ---------------------------------------------------
    def cleanup(self):
        try:
            if self.stream_handler:
                self.stream_handler.release()
        except:
            pass

        cv2.destroyAllWindows()
        print(f"[INFO] Stopped stream → {self.video_source}")
