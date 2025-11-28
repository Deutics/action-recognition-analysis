# loaders/pytorchvideo_loader/pytorchvideo_loader.py
from pytorchvideo.models.hub import (
    x3d_s, x3d_m, x3d_l,
    r2plus1d_r50, i3d_r50, c2d_r50, csn_r101
)
from inference.classifier_predictor import ClassifierPredictor
from loaders.base_loader import BaseModelLoader
from utils.logger import get_logger
logger = get_logger(__name__)


class PyTorchVideoLoader(BaseModelLoader):
    """
    PyTorchVideoLoader for dynamically loading video models.

    Provides model construction, preprocessing, and predictor setup.
    """

    MODEL_FNS = {
        "x3d_s": (x3d_s, 32, 224),
        "x3d_m": (x3d_m, 32, 224),
        "x3d_l": (x3d_l, 32, 320),
        "r2plus1d_r50": (r2plus1d_r50, 16, 224),
        "i3d_r50": (i3d_r50, 16, 224),
        "c2d_r50": (c2d_r50, 16, 224),
        "csn_r101": (csn_r101, 32, 256),
    }

    def __init__(self, model_name, device, classnames):
        self.model_name = model_name
        self.device = device
        self.classnames = classnames
        self.preprocessor = None
        self.predictor = None
        logger.info(f"PyTorchVideoLoader initialized for model '{self.model_name}' on device '{self.device}'")

    def load(self):
        if self.model_name not in self.MODEL_FNS:
            logger.error(f"Model '{self.model_name}' not supported")
            raise ValueError(f"Model '{self.model_name}' not supported")

        fn, T, img_size = self.MODEL_FNS[self.model_name]
        logger.info(f"Loading model '{self.model_name}' with T={T}, img_size={img_size}")

        # Load pretrained model
        model = fn(pretrained=True).to(self.device).eval()
        logger.info(f"Model '{self.model_name}' loaded successfully on {self.device}")

        # Load preprocessor and predictor
        from preprocessing.frame_preprocessor import FramePreprocessor
        self.preprocessor = FramePreprocessor(img_size)
        self.predictor = ClassifierPredictor(model, self.classnames, self.device)
        logger.info(f"Preprocessor and predictor initialized for '{self.model_name}'")

        return model, T, img_size, self.preprocessor, self.predictor
