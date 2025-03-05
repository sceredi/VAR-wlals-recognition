from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPExtractor:
    """Local Binary Patterns class to extract LBP features from a list of frames"""

    def __init__(self, frames: List[np.ndarray], radius: int = 3) -> None:
        """
        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to extract LBP features from
        radius : int
            The radius of the LBP circle
        """
        self._frames = frames
        self._radius = radius
        self._n_points = 8 * radius
        self._n_bins = self._n_points + 2
        self._features = None
        self._lbp_frames = None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale

        Parameters
        ----------
        frame : np.ndarray
            The frame to convert to grayscale

        Returns
        -------
        np.ndarray
            The grayscale frame
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def process_frames(self) -> List[np.ndarray]:
        """
        Extracts LBP features from a list of frames

        Returns
        -------
        List[np.ndarray]
            The LBP features
        """
        self._lbp_frames, self._features = zip(
            *(self._extract(self._to_gray(frame)) for frame in self._frames), strict=False
        )
        hists, _ = self._features
        return hists

    def _extract(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts LBP features from a single frame

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract LBP features from

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The LBP features and the LBP frame
        """
        lbp_frame = local_binary_pattern(
            frame, self._n_points, self._radius, method="uniform"
        )

        hist, bins = np.histogram(
            lbp_frame.ravel(),
            bins=np.arange(0, self._n_bins + 1),
            range=(0, self._n_bins),
        )
        hist = hist.astype(np.float32) / hist.sum()

        return lbp_frame, (hist, bins)

    def get_lbp_frames(self) -> List[np.ndarray]:
        """
        Get the LBP frames

        Returns
        -------
        List[np.ndarray]
            The LBP frames
        """
        if self._lbp_frames is None:
            self.process_frames()
        return self._lbp_frames

    def get_lbp_features(self) -> List[np.ndarray]:
        """
        Get the LBP features

        Returns
        -------
        List[np.ndarray]
            The LBP features
        """
        if self._features is None:
            self.process_frames()
        return self._features
