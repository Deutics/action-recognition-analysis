# loaders/slowfast_loader/slowfast_loader.py
from pytorchvideo.models.hub import slowfast_r50, slowfast_r101
from loaders.base_loader import BaseModelLoader
from inference.slowfast_predictor import SlowFastPredictor
from preprocessing.slowfast_preprocessor import SlowFastPreprocessor
from utils.logger import get_logger
logger = get_logger(__name__)


class SlowFastLoader(BaseModelLoader):
    """
    Loader for SlowFast models.
    Provides model construction, preprocessing, and predictor setup.
    """

    MODEL_FNS = {
        # model_name: (model_fn, fast_T, img_size, alpha)
        "slowfast_r50": (slowfast_r50, 32, 224, 4),
        "slowfast_r101": (slowfast_r101, 64, 256, 8),
    }

    def __init__(self, model_name, device, classnames):
        self.model_name = model_name
        self.device = device
        self.classnames = classnames
        self.preprocessor = None
        self.predictor = None
        logger.info(f"SlowFastLoader initialized for model '{self.model_name}' on device '{self.device}'")

    def load(self):
        """Load model, preprocessor, and predictor for SlowFast inference."""
        if self.model_name not in self.MODEL_FNS:
            logger.error(f"Model '{self.model_name}' not supported")
            raise ValueError(f"Model '{self.model_name}' not supported")

        fn, fast_T, img_size, alpha = self.MODEL_FNS[self.model_name]
        logger.info(f"Loading SlowFast model '{self.model_name}' with fast_T={fast_T}, img_size={img_size}, alpha={alpha}")

        # Load pretrained SlowFast model
        model = fn(pretrained=True).to(self.device).eval()
        logger.info(f"SlowFast model '{self.model_name}' loaded successfully on {self.device}")

        # Build preprocessor and predictor
        self.preprocessor = SlowFastPreprocessor(fast_T, img_size, alpha)
        self.predictor = SlowFastPredictor(model, self.classnames, self.device)
        logger.info(f"Preprocessor and predictor initialized for '{self.model_name}'")

        return model, fast_T, img_size, self.preprocessor, self.predictor
