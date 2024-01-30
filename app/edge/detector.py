# Description: This file contains the EdgeDetector class which is used to detect edges in a video.

from typing import List
import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, frames: List["np.ndarray"]) -> None:
        self.frames = frames

    def detect(self) -> List["np.ndarray"]:
        frames = []
        for frame in self.frames:
            frame = self._detect_edges(frame)
            frames.append(frame)
        return frames

    def _detect_edges(self, frame: "np.ndarray") -> "np.ndarray":
        frame = self._preprocess(frame)
        edges = cv2.Canny(frame, 40, 80)
        return edges

    def _preprocess(self, frame: "np.ndarray") -> "np.ndarray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur 
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # Equalize Histogram
        frame = cv2.equalizeHist(frame)
        return frame
