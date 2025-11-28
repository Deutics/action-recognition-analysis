# loaders/model_loader.py
import importlib
from configs.model_registry import MODEL_REGISTRY
from utils.logger import get_logger
logger = get_logger(__name__)

class ModelLoader:
    """Dynamically loads a model loader class based on model name."""

    def __init__(self, model_name: str, device, classnames):
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Model '{model_name}' not found in registry")
            raise ValueError(f"Model '{model_name}' not found in registry")

        self.model_name = model_name
        self.device = device
        self.classnames = classnames
        logger.info(f"ModelLoader initialized for model '{self.model_name}' on device '{self.device}'")

    def load(self):
        """
        Dynamically imports the module and class for the specified model,
        initializes it, and calls its load() method.
        """
        module_path = MODEL_REGISTRY[self.model_name]["module"]
        class_name = MODEL_REGISTRY[self.model_name]["class"]

        logger.info(f"Loading model '{self.model_name}' using {module_path}.{class_name}")
        module = importlib.import_module(module_path)
        LoaderCls = getattr(module, class_name)

        loader = LoaderCls(self.model_name, self.device, self.classnames)
        model = loader.load()
        logger.info(f"Model '{self.model_name}' loaded successfully")
        return model
