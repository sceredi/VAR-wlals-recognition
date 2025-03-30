"""Module ColorHistogram for extracting color histogram from a list of image frames."""

from typing import List

import cv2
import numpy as np


class ColorHistogram:
    """Color histogram extractor for feature extraction from frames."""

    def __init__(self, frames: List[np.ndarray], n_bins: int = 256) -> None:
        """Initialize the ColorHistogram.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to extract color histograms from.
        n_bins : int, optional
            The number of bins for the histograms (default is 256).

        """
        self.frames = frames
        self.n_bins = n_bins

    def process_frames(
        self,
        to_color=cv2.COLOR_BGR2HSV,
        separate_colors=False,
        normalize=False,
    ) -> np.ndarray:
        """Extract color histograms from a list of frames.

        Parameters
        ----------
        to_color : int, optional
            The color space conversion code (default is cv2.COLOR_BGR2HSV).
        separate_colors : bool, optional
            Whether to extract histograms for each channel separately (default is False).
        normalize: bool, optional
            Whether to normalize the histograms (default is False).

        Returns
        -------
        np.ndarray
            The extracted color histograms.

        """
        ret = []
        for frame in self.frames:
            if not separate_colors:
                ret.append(
                    self._extract(
                        cv2.cvtColor(frame, to_color), normalize, self.n_bins
                    )
                )
            else:
                ret.append(
                    self._extract_separate(
                        cv2.cvtColor(frame, to_color), normalize, self.n_bins
                    )
                )
        return np.array(ret)

    @staticmethod
    def _extract(frame: np.ndarray, normalize: bool, n_bins: int = 256):
        """Extract a color histogram from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract the histogram from.
        normalize: bool
            Whether to normalize the histogram.

        Returns
        -------
        np.ndarray
            The color histogram.

        """
        hist = cv2.calcHist(
            [frame],
            [0, 1, 2],
            None,
            [n_bins, n_bins, n_bins],
            [0, n_bins, 0, n_bins, 0, n_bins],
        )
        if normalize:
            hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _extract_separate(
        frame: np.ndarray, normalize: bool, n_bins: int = 256
    ):
        """Extract separate color histograms for each channel.

        Parameters
        ----------
        frame : np.ndarray
            The frame to extract histograms from.
        normalize: bool
            Whether to normalize the histogram.

        Returns
        -------
        List[np.ndarray]
            The list of histograms, one for each channel.

        """
        hists = []
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [n_bins], [0, n_bins])
            if normalize:
                hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hists.append(hist)
        return hists
