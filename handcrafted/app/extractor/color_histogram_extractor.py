from typing import List

import cv2
import numpy as np


class ColorHistogram:
    def __init__(self, frames: List["np.ndarray"]) -> None:
        self.frames = frames

    def process_frames(
        self, to_color=cv2.COLOR_BGR2HSV, separate_colors=False
    ) -> "np.ndarray":
        ret = []
        for frame in self.frames:
            if not separate_colors:
                ret.append(self._extract(cv2.cvtColor(frame, to_color)))
            else:
                ret.append(
                    self._extract_separate(cv2.cvtColor(frame, to_color))
                )
        return np.array(ret)

    def _extract(self, frame: "np.ndarray"):
        histogram = cv2.calcHist(
            [frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
        )
        return histogram
        # return histogram / histogram.sum()

    def _extract_separate(self, frame: "np.ndarray"):
        hists = []
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hists.append(hist)
        return hists
