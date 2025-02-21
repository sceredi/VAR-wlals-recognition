from typing import List

import cv2
import numpy as np


class ContourDetector:
    def __init__(self, frames: List["np.ndarray"]) -> None:
        self.frames = frames

    def detect(self) -> List["np.ndarray"]:
        frames = []
        for frame in self.frames:
            frame = self._extract_contour(frame)
            frames.append(frame)
        return frames

    def _extract_contour(self, frame: "np.ndarray") -> "np.ndarray":
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = frame.copy()
        cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)
        return result_frame
