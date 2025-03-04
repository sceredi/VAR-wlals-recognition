from typing import List

import cv2
import numpy as np

# Description: This file contains the Frame class which is used to represent
#              a frame in the video with all the extra inforamtions needed for
#              the optical flow calculation.


class Frame:
    def __init__(self, rgb: np.ndarray) -> None:
        self.rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(rgb, dtype=np.uint8)

    def set_keypoints(self, keypoints: np.ndarray) -> None:
        self.keypoints = keypoints

    def set_descriptors(self, descriptors: np.ndarray) -> None:
        self.descriptors = descriptors


class Frames:
    def __init__(self, frames: List[np.ndarray]) -> None:
        self.frames = [Frame(frame) for frame in frames]

    def __getitem__(self, index) -> Frame:
        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)
