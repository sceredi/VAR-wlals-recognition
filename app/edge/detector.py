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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        equalize_hist = cv2.equalizeHist(blurred_frame)

        gx = cv2.Sobel(equalize_hist, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(equalize_hist, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = cv2.convertScaleAbs(np.sqrt((gx ** 2) + (gy ** 2)))

        _, binary_frame = cv2.threshold(sobel_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        result_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

        return result_frame
