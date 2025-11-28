# loaders/classnames_loader.py
import csv
from utils.logger import get_logger
logger = get_logger(__name__)

class ClassNamesLoader:
    """Load class mapping from CSV file with columns: id,name"""

    def __init__(self, path: str):
        self.path = path
        logger.info(f"ClassNamesLoader initialized with path: {self.path}")

    def load(self) -> dict:
        """Load class mapping from CSV file and return as a dictionary."""
        class_map = {}
        try:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_map[int(row["id"])] = row["name"].strip()
            logger.info(f"Loaded {len(class_map)} class names from '{self.path}'")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load class names from '{self.path}': {e}")
            raise
        return class_map
