import os
import platform
from pathlib import Path
from typing import Optional, Tuple

from config.constants import HUMAN_DETECTOR_CANDIDATE_MODELS
from inference.torch_backend import TorchBackend
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from inference.human_detector import HumanDetector
except Exception as e:
    logger.warning(f"HumanDetector import unavailable: {e}")
    HumanDetector = None

try:
    from inference.tensorrt_backend import TensorRTBackend
    TRT_AVAILABLE = True
except Exception as e:
    logger.warning(f"TensorRT backend import unavailable: {e}")
    TensorRTBackend = None
    TRT_AVAILABLE = False


class InferenceRuntimeFactory:
    @staticmethod
    def is_jetson() -> bool:
        return os.path.exists("/etc/nv_tegra_release")

    @staticmethod
    def _engine_path_from_model(model_path: str) -> str:
        model_file = Path(model_path)
        stem = model_file.stem
        return str(model_file.with_name(f"{stem}_fp16.engine"))

    @classmethod
    def create_pose_backend(cls, model_path: str, use_tracking: bool = True) -> Tuple[object, str]:
        if platform.system() == "Darwin":
            logger.info("Using Torch backend (Mac)")
            return TorchBackend(model_path, use_tracking=use_tracking), "torch"

        if cls.is_jetson() and TRT_AVAILABLE:
            engine_path = cls._engine_path_from_model(model_path)
            if os.path.exists(engine_path):
                logger.info(f"Using TensorRT backend (Jetson): {engine_path}")
                return TensorRTBackend(engine_path), "tensorrt"
            logger.warning(f"TensorRT engine not found, falling back to Torch CPU: {engine_path}")

        logger.info("Using Torch backend (fallback)")
        return TorchBackend(model_path, use_tracking=use_tracking), "torch"

    @classmethod
    def create_human_detector(cls, preferred_model: Optional[str] = None) -> object:
        if HumanDetector is None:
            raise RuntimeError("Human detector module unavailable")

        candidate_models = []
        if preferred_model:
            if cls.is_jetson() and preferred_model.lower().endswith(".pt"):
                candidate_models.append(cls._engine_path_from_model(preferred_model))
            candidate_models.append(preferred_model)
        candidate_models.extend(HUMAN_DETECTOR_CANDIDATE_MODELS)

        seen = set()
        ordered_candidates = []
        for model_path in candidate_models:
            if model_path not in seen:
                seen.add(model_path)
                ordered_candidates.append(model_path)

        for model_path in ordered_candidates:
            if os.path.exists(model_path):
                logger.info(f"Using shared human detector model: {model_path}")
                return HumanDetector(model_path=model_path)

        raise FileNotFoundError(
            "No human detector model found. Tried: " + ", ".join(ordered_candidates)
        )
