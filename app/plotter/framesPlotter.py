import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FramesPlotter:
    def __init__(self, frames: List["np.ndarray"]):
        self.frames = frames

    def plot(self):
        num_frames = len(self.frames)
        side_length = int(math.ceil(math.sqrt(num_frames)))
        _, axes = plt.subplots(side_length, side_length)
        axes = axes.flatten()
        for i, (frame, ax) in enumerate(zip(self.frames, axes)):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame)
            ax.set_title(f"Frame {i+1}")
            ax.axis("off")
        for i in range(num_frames, len(axes)):
            axes[i].axis("off")
        plt.show()
