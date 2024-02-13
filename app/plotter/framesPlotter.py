import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FramesPlotter:
    def __init__(self, frames: List["np.ndarray"], to_rgb=True):
        self.frames = frames
        self.to_rgb = to_rgb

    def plot_grid(self):
        num_frames = len(self.frames)
        print(num_frames)
        side_length = int(math.ceil(math.sqrt(num_frames)))
        print(side_length)
        _, axes = plt.subplots(side_length, side_length)
        axes = axes.flatten()
        for i, (frame, ax) in enumerate(zip(self.frames, axes)):
            self._update(ax, frame, f"Frame {i+1}")
        for ax in axes[num_frames:]:
            self._remove_axis(ax)
        plt.show()

    def _remove_axis(self, ax):
        ax.axis("off")

    def _to_rgb(self, frame) -> "np.ndarray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _update(self, ax, frame, title):
        if self.to_rgb:
            frame = self._to_rgb(frame)
        ax.clear()
        ax.imshow(frame)
        ax.set_title(title)
        self._remove_axis(ax)
