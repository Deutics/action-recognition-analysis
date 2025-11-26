# loaders/pytorchvideo_loader/pytorchvideo_loader.py
import torch
from pytorchvideo.models.hub import (
    x3d_s, x3d_m, x3d_l,
    r2plus1d_r50, i3d_r50, c2d_r50, csn_r101
)

from inference.classifier_predictor import ClassifierPredictor
from loaders.base_loader import BaseModelLoader

class PyTorchVideoLoader(BaseModelLoader):
    """Loads any PyTorchVideo model dynamically."""

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

    def load(self):
        fn, T, img_size = self.MODEL_FNS[self.model_name]
        model = fn(pretrained=True).to(self.device).eval()
        from preprocessing.frame_preprocessor import FramePreprocessor
        self.preprocessor = FramePreprocessor(img_size)
        self.predictor = ClassifierPredictor(model, self.classnames, self.device)
        return model, T, img_size, self.preprocessor, self.predictor