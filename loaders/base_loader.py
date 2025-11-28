# action_recognition/loaders/base_loader.py
from abc import ABC, abstractmethod

class BaseModelLoader(ABC):
    """Base class for all action recognition model loaders."""

    @abstractmethod
    def load(self):
        """Load and return (model, frame_window_size, img_size)."""
        pass
