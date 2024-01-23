import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FramesPlotter:
    def __init__(self, frames: List["np.ndarray"]):
        self.frames = frames

    def plot_grid(self):
        num_frames = len(self.frames)
        side_length = int(math.ceil(math.sqrt(num_frames)))
        _, axes = plt.subplots(side_length, side_length)
        axes = axes.flatten()
        for i, (frame, ax) in enumerate(zip(self.frames, axes)):
            self._update(ax, frame, f"Frame {i+1}")
        plt.show()

    def _to_rgb(self, frame) -> "np.ndarray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _update(self, ax, frame, title):
        frame = self._to_rgb(frame)
        ax.clear()
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
