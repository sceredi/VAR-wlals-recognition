"""ContourExtractor class for extracting contours from a list of image frames."""

from typing import List

import cv2
import numpy as np


class ContourExtractor:
    """ContourExtractor class which is used to find contours from frames."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        """Initialize the ContourExtractor.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to extract contours from.

        """
        self.frames = frames

    def process_frames(self) -> List[np.ndarray]:
        """Extract contours from a list of frames.

        Returns
        -------
        List[np.ndarray]
            The processed frames with drawn contours.

        """
        frames = []
        for frame in self.frames:
            frame = self._extract(frame)
            frames.append(frame)
        return frames

    @staticmethod
    def _extract(frame: np.ndarray) -> np.ndarray:
        """Extract contours from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract contours from.

        Returns
        -------
        np.ndarray
            The frame with detected contours drawn.

        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(
            blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result_frame = frame.copy()
        cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)
        return result_frame
