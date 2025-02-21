from typing import List

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPExtractor:
    def __init__(self, frames: List["np.ndarray"]) -> None:
        self.frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    def extract_single(
        self, frame: "np.ndarray", radius: int = 1, n_points: int = 25
    ) -> "np.ndarray":
        lbp = local_binary_pattern(frame, n_points, radius, method="uniform")

        # Calculate histogram
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
        )
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        return hist

    def extract(self) -> List["np.ndarray"]:
        ret = []
        for frame in self.frames:
            ret.append(self.extract_single(frame))
        return ret
