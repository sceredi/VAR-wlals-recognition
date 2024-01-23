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
            frame = self.detect_edges(frame)
            frames.append(frame)
        return frames
    
    def detect_edges(self, frame: "np.ndarray") -> "np.ndarray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 150)
        return edged
