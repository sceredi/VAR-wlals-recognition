# Description: This file contains the Frame class which is used to represent
#              a frame in the video with all the extra inforamtions needed for
#              the optical flow calculation.

import cv2
import numpy as np


class Frame:
    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(rgb, dtype=np.uint8)

    def set_keypoints(self, keypoints: np.ndarray):
        self.keypoints = keypoints

    def set_descriptors(self, descriptors: np.ndarray):
        self.descriptors = descriptors


from typing import List


class Frames:
    def __init__(self, frames: List[np.ndarray]):
        self.frames = [Frame(frame) for frame in frames]
