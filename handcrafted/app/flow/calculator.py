# Description: This file contains the class for calculating the optical flow using Gunnar Farneback's algorithm.

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FlowCalculator:
    def __init__(
        self,
        frames: List["np.ndarray"],
        last_frame_index: int,
        params: dict = {
            "pyr_scale": 0.3,
            "levels": 5,
            "winsize": 10,
            "iterations": 6,
            "poly_n": 5,
            "poly_sigma": 1.5,
            "flags": 0,
        },
    ):
        self.frames = frames
        self.last_frame_index = last_frame_index
        self.params = params

    def calculate(self, plot_each_frame: bool = False):
        frames = []
        prev_frame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(self.frames[0])
        hsv[..., 1] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frames.append(bgr)
        for i, frame in enumerate(self.frames[1 : self.last_frame_index + 1]):
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, **self.params
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(
                magnitude, None, 0, 255, cv2.NORM_MINMAX
            )
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if plot_each_frame:
                self._plot(frame, bgr, i)
            frames.append(bgr)
            prev_frame = next_frame
        mask = np.zeros_like(self.frames[0])
        for _ in range(self.last_frame_index + 1, len(self.frames)):
            frames.append(mask)
        return frames

    def _plot(self, frame, flow, index):
        fig, axes = plt.subplots(1, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes[0].imshow(frame)
        axes[0].set_title(f"Frame {index}")

        axes[1].imshow(flow)
        axes[1].set_title(f"Flow {index}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
