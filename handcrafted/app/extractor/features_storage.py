"""Module to save and load features using joblib."""

import os

import joblib
import numpy as np


class FeaturesStorage:
    """FeaturesStorage class to save and load features using joblib."""

    @staticmethod
    def save_feature(feature: np.ndarray, file_path: str) -> None:
        """Save the feature to the specified file path.

        Parameters
        ----------
        feature: np.ndarray
            The feature to save.
        file_path: str
            The file path to save the feature.

        """
        joblib.dump(feature, file_path)

    @staticmethod
    def load_feature(file_path: str) -> np.ndarray:
        """Load a feature from the specified file path.

        Parameters
        ----------
        file_path: str
            The file path to load the feature from.

        Returns
        -------
        np.ndarray
            The loaded feature.

        Raises
        ------
        FileNotFoundError
            If the file_path is not found.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        return joblib.load(file_path)
