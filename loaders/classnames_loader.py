# loader/classnames_loader.py
import csv

class ClassNamesLoader:
    """Load class mapping from CSV file with columns: id,name"""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict:
        class_map = {}
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_map[int(row["id"])] = row["name"].strip()
        return class_map
