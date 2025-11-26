# loaders/slowfast_loader/slowfast_loader.py

import torch
from pytorchvideo.models.hub import slowfast_r50,slowfast_r101

from loaders.base_loader import BaseModelLoader
from inference.slowfast_predictor import SlowFastPredictor
from preprocessing.slowfast_preprocessor import SlowFastPreprocessor


class SlowFastLoader(BaseModelLoader):
    """
    Loader for SlowFast models.
    Fully consistent with PyTorchVideoLoader.
    """

    MODEL_FNS = {
        # model_name: (model_fn, fast_T, img_size, alpha)
        "slowfast_r50": (slowfast_r50, 32, 224, 4),
        # Add more variants here if needed
        "slowfast_r101": (slowfast_r101, 64, 256, 8),
    }

    def __init__(self, model_name, device, classnames):
        self.model_name = model_name
        self.device = device
        self.classnames = classnames
        self.preprocessor = None
        self.predictor = None

    def load(self):
        # Fetch model config dynamically
        fn, fast_T, img_size, alpha = self.MODEL_FNS[self.model_name]

        # Load pretrained SlowFast model
        model = fn(pretrained=True).to(self.device).eval()

        # Build SlowFast preprocessor
        self.preprocessor = SlowFastPreprocessor(fast_T,img_size, alpha)
        self.predictor = SlowFastPredictor(model, self.classnames, self.device)
        return model, fast_T, img_size, self.preprocessor, self.predictor
