import sys
import numpy as np
# from sympy.printing.pytorch import torch
import torch

from utils.logger import get_logger

sys.path.insert(0, r"E:\PyCharmProjects\DEUTICS-GLOBAL\behaviour\test_action\mmaction2")

from mmaction.apis import init_recognizer, inference_recognizer

logger = get_logger(__name__)


class PoseC3DRecognizer:
    """Wrapper for PoseC3D model initialization and inference."""

    def __init__(self, config_path, checkpoint_path, device="cpu", action_labels=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = init_recognizer(config_path, checkpoint_path, device=self.device)
        self.action_labels = action_labels if action_labels is not None else []

        logger.info(
            f"PoseC3DRecognizer initialized with config={config_path}, "
            f"checkpoint={checkpoint_path}, device={device}, "
            f"labels_loaded={len(self.action_labels)}"
        )

    def infer(self, sample):
        logger.debug("Running PoseC3D inference...")
        result = inference_recognizer(self.model, sample)
        if isinstance(result, list):
            result = result[0]
        logger.debug("PoseC3D inference completed.")
        return result

    def get_top_result(self, result):
        pred_scores = result.pred_score.cpu().numpy()
        top_index = pred_scores.argmax()
        top_score = float(pred_scores[top_index])

        label_name = (
            self.action_labels[top_index]
            if self.action_labels and top_index < len(self.action_labels)
            else str(top_index)
        )

        logger.debug(
            f"Top prediction: label={label_name}, score={top_score:.4f}"
        )
        return label_name, top_score