"""Module for extracting edges from a list of frames."""

from typing import List

import cv2
import numpy as np


class EdgeExtractor:
    """EdgeExtractor class which is used to detect edges in a video."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        """Initializes the EdgeExtractor class.

        Parameters
        ----------
        frames: List[np.ndarray]
            List of frames to extract edges from.

        """
        self.frames = frames

    def process_frames(self) -> List[np.ndarray]:
        """Processes the frames to extract edges.

        Returns
        -------
        List[np.ndarray]
            List of frames with edges extracted.

        """
        frames = []
        for frame in self.frames:
            frame = self._extract(frame)
            frames.append(frame)
        return frames

    @staticmethod
    def _extract(frame: np.ndarray) -> np.ndarray:
        """Extracts edges from a frame.

        Parameters
        ----------
        frame: np.ndarray
            Frame to extract edges from.

        Returns
        -------
        np.ndarray
            Frame with edges extracted.

        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        equalize_hist = cv2.equalizeHist(blurred_frame)
        gx = cv2.Sobel(equalize_hist, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(equalize_hist, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = cv2.convertScaleAbs(np.sqrt((gx**2) + (gy**2)))  # type: ignore
        _, binary_frame = cv2.threshold(
            sobel_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = np.ones((3, 3), np.uint8)
        result_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
        return result_frame
