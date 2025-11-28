import time
import torch
from utils.logger import get_logger
logger = get_logger(__name__)


class InferenceEngine:
    """
    General inference engine that wraps a model and optional predictor.
    Handles device placement and inference timing.
    """

    def __init__(self, model, predictor, device):
        self.model = model
        self.predictor = predictor
        self.device = device
        logger.info(f"InferenceEngine initialized on device '{self.device}' with predictor={bool(self.predictor)}")

    def run(self, tensor):
        """
        Run inference on the provided tensor or list of tensors.

        Args:
            tensor (torch.Tensor or list[torch.Tensor]): Input tensor(s).

        Returns:
            If predictor exists: results, inference_time
            Else: model output, inference_time
        """
        if tensor is None:
            logger.warning("Received None tensor, returning None results")
            return None, None

        # Handle SlowFast-style list of tensors
        if isinstance(tensor, list):
            tensor = [t.to(self.device) for t in tensor]
            logger.debug(f"Moved list of {len(tensor)} tensors to device '{self.device}'")
        else:
            tensor = tensor.to(self.device)
            logger.debug(f"Moved tensor of shape {tensor.shape} to device '{self.device}'")

        # Use predictor if available
        if self.predictor:
            logger.debug("Using predictor for inference")
            return self.predictor.predict(tensor)

        # Direct model forward
        logger.debug("Running direct model forward pass")
        t0 = time.time()
        with torch.no_grad():
            out = self.model(tensor)
        infer_time = time.time() - t0
        logger.debug(f"Direct model inference completed in {infer_time:.4f}s")
        return out, infer_time
