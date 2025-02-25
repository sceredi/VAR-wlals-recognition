import os

import joblib
import numpy as np


class FeaturesStorage:
    """FeaturesStorage class to save and load features using joblib."""

    @staticmethod
    def save_feature(feature: np.ndarray, file_path: str) -> None:
        """Save the feature to the specified file path."""
        joblib.dump(feature, file_path)

    @staticmethod
    def load_feature(file_path: str) -> np.ndarray:
        """Load a feature from the specified file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        return joblib.load(file_path)
