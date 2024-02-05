# Description: This file contains the class for calculating the optical flow using Gunnar Farneback's algorithm.

from typing import List
import cv2
import numpy as np


class FlowCalculator:
    def __init__(
        self,
        frames: List["np.ndarray"],
        params: dict = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        },
    ):
        self.frames = frames
        self.params = params

    def calculate(self):
        frames = []
        prev_frame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(self.frames[0])
        hsv[..., 1] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frames.append(bgr)
        for frame in self.frames[1:]:
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, **self.params
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            frames.append(bgr)
            prev_frame = next_frame
        return frames

