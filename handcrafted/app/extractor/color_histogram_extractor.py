"""ColorHistogram class for extracting color histograms from a list of image frames."""

from typing import List

import cv2
import numpy as np


class ColorHistogram:
    """Color histogram extractor for feature extraction from frames."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        """Initialize the ColorHistogram.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to extract color histograms from.

        """
        self.frames = frames

    def process_frames(
        self, to_color: int = cv2.COLOR_BGR2HSV, separate_colors: bool = False
    ) -> np.ndarray:
        """Extract color histograms from a list of frames.

        Parameters
        ----------
        to_color : int, optional
            The color space conversion code (default is cv2.COLOR_BGR2HSV).
        separate_colors : bool, optional
            Whether to extract histograms for each channel separately (default is False).

        Returns
        -------
        np.ndarray
            The extracted color histograms.

        """
        ret = []
        for frame in self.frames:
            if not separate_colors:
                ret.append(self._extract(cv2.cvtColor(frame, to_color)))
            else:
                ret.append(
                    self._extract_separate(cv2.cvtColor(frame, to_color))
                )
        return np.array(ret)

    @staticmethod
    def _extract(frame: np.ndarray):
        """Extract a color histogram from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract the histogram from.

        Returns
        -------
        np.ndarray
            The color histogram.

        """
        histogram = cv2.calcHist(
            [frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
        )
        return histogram
        # return histogram / histogram.sum()

    @staticmethod
    def _extract_separate(frame: np.ndarray):
        """Extract separate color histograms for each channel.

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract histograms from.

        Returns
        -------
        List[np.ndarray]
            The list of histograms, one for each channel.

        """
        hists = []
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hists.append(hist)
        return hists
