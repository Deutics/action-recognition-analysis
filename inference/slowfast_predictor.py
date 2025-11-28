import time
import torch
from inference.base_predictor import BasePredictor
from utils.logger import get_logger
logger = get_logger(__name__)


class SlowFastPredictor(BasePredictor):
    """
    SlowFastPredictor for running inference with SlowFast video models.

    Wraps a model to process slow/fast pathways and returns
    predicted class scores with inference time.
    """

    def __init__(self, model, classnames, device):
        self.model = model
        self.classnames = classnames
        self.device = device
        logger.info(f"SlowFastPredictor initialized on device '{self.device}' with {len(self.classnames)} classes")

    def predict(self, pathways):
        """
        Run inference on SlowFast pathways.

        Args:
            pathways (list[Tensor]): [slow_tensor, fast_tensor] inputs.

        Returns:
            tuple: (results, inference_time)
                - results (list[dict]): predicted class labels and scores
                - inference_time (float): time taken for inference
        """
        slow, fast = pathways
        slow = slow.to(self.device)
        fast = fast.to(self.device)
        logger.debug(f"SlowFastPredictor received tensors: slow={slow.shape}, fast={fast.shape}")

        start = time.time()
        with torch.no_grad():
            out = self.model([slow, fast])   # SlowFast expects a list
            probs = torch.softmax(out, dim=1)[0]
            topk = torch.topk(probs, 1)
        end = time.time()

        results = [
            {"class": self.classnames[int(idx)], "score": float(score)}
            for idx, score in zip(topk.indices, topk.values)
        ]
        inference_time = end - start
        logger.debug(f"Inference completed in {inference_time:.4f}s, top prediction: {results[0]}")
        return results, inference_time
