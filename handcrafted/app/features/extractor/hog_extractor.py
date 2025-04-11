"""Module to extract Histogram of Oriented Gradients (HOG) features from a list of frames."""

from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog


class HOGExtractor:
    """HOGExtractor class which is used to extract HOG features in a list of frames."""

    def __init__(
        self,
        frames: List[np.ndarray],
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
    ) -> None:
        """Initialize the HOGExtractor class.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to extract HOG features from.
        orientations : int, optional
            Number of orientation bins, by default 9.
        pixels_per_cell : Tuple[int, int], optional
            Size (in pixels) of a cell, by default (8, 8).
        cells_per_block : Tuple[int, int], optional
            Number of cells in each block, by default (2, 2).

        """
        self.frames = frames
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.hog_params = {
            "orientations": orientations,
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
            "block_norm": "L2-Hys",
            "visualize": True,
            "transform_sqrt": True,
        }

    def process_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract HOG features from the frames.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Tuple containing the HOG features and the processed frames.

        """
        processed_frames = []
        processed_features = []
        for frame in self.frames:
            hog_features, hog_image = self._extract(frame)
            processed_frames.append(hog_image)
            processed_features.append(hog_features)
        return processed_features, processed_frames

    def _extract(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract HOG features from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame to extract HOG features from.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing the HOG features and the processed frame.

        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog_features, hog_image = hog(gray_frame, **self.hog_params)
        hog_image = (hog_image * 255).astype(np.uint8)
        return hog_features, hog_image
