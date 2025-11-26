# loaders/model_loader.py
import importlib
from configs.model_registry import MODEL_REGISTRY

class ModelLoader:
    """Dynamically loads a model loader class based on model name."""

    def __init__(self, model_name: str, device,classnames):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry")

        self.model_name = model_name
        self.device = device
        self.classnames = classnames

    def load(self):
        module_path = MODEL_REGISTRY[self.model_name]["module"]
        class_name = MODEL_REGISTRY[self.model_name]["class"]

        module = importlib.import_module(module_path)
        LoaderCls = getattr(module, class_name)

        loader = LoaderCls(self.model_name, self.device, self.classnames)
        return loader.load()
