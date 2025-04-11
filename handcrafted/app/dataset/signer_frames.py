"""Module to handle signer frames and their features."""

import cv2
import numpy as np

from handcrafted.app.features.extractor.color_histogram_extractor import (
    ColorHistogram,
)
from handcrafted.app.features.extractor.hog_extractor import HOGExtractor
from handcrafted.app.features.extractor.lbp_extractor import LBPExtractor


class SignerFrame:
    """Class to handle signer frames and their features."""

    def __init__(
        self, frame: np.ndarray, signer_id: int, extract_features: bool = True
    ) -> None:
        """Initialize the SignerFrame object.

        Parameters
        ----------
        frame : np.ndarray
            The frame to process.
        signer_id : int
            The ID of the signer.
        extract_features : bool, optional
            If True, extracts features from the frame, by default True.

        """
        self.frame = frame
        self.signer_id = signer_id
        self.extract_features = extract_features
        if extract_features:
            self.features = self._extract_features()

    def _extract_features(self) -> np.ndarray:
        """Extract features from the frame.

        This function uses HOG, LBP, and Color Histogram features.

        Returns
        -------
        np.ndarray
            The extracted features.

        """
        hog_features, _ = HOGExtractor([self.frame]).process_frames()
        lbp_features = LBPExtractor([self.frame]).get_lbp_features()
        color_hist_features = ColorHistogram([self.frame]).process_frames(
            cv2.COLOR_BGR2HSV, separate_colors=False, normalize=True
        )
        hog_features = np.reshape(hog_features, -1)
        lbp_features = np.reshape(lbp_features, -1)
        color_hist_features = np.reshape(color_hist_features, -1)
        return np.concatenate(
            (hog_features, lbp_features, color_hist_features)
        )
