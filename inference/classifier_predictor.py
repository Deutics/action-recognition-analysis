# inference/classifier_predictor.py
import time
import torch
from inference.base_predictor import BasePredictor
from utils.logger import get_logger
logger = get_logger(__name__)


class ClassifierPredictor(BasePredictor):
    """
    ClassifierPredictor for running inference with classification models.

    Wraps a model to compute predictions on input tensors, returning
    top class scores and inference time.
    """

    def __init__(self, model, classnames, device):
        self.model = model
        self.classnames = classnames
        self.device = device
        logger.info(f"ClassifierPredictor initialized on device '{self.device}' with {len(self.classnames)} classes")

    def predict(self, buffer):
        """Run inference on the input tensor buffer."""
        tensor = buffer.to(self.device)
        logger.debug(f"Starting inference on tensor of shape {tensor.shape}")

        start = time.time()
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            top5 = torch.topk(probs, 1)
        end = time.time()

        results = [
            {"class": self.classnames[int(idx)], "score": float(score)}
            for idx, score in zip(top5.indices, top5.values)
        ]
        inference_time = end - start
        logger.debug(f"Inference completed in {inference_time:.4f}s, top prediction: {results[0]}")
        return results, inference_time
