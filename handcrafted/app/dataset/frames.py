"""Module to represent frames in the video."""

from typing import List

import cv2
import numpy as np


class Frame:
    """Class to represent a frame in the video."""

    def __init__(self, rgb: np.ndarray) -> None:
        """Initialize the Frame object.

        Parameters
        ----------
        rgb : np.ndarray
            The RGB frame.

        """
        self.rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(rgb, dtype=np.uint8)

    def set_keypoints(self, keypoints: np.ndarray) -> None:
        """Set the keypoints for the frame.

        Parameters
        ----------
        keypoints : np.ndarray
            The keypoints to set.

        """
        self.keypoints = keypoints

    def set_descriptors(self, descriptors: np.ndarray) -> None:
        """Set the descriptors for the frame.

        Parameters
        ----------
        descriptors : np.ndarray
            The descriptors to set.

        """
        self.descriptors = descriptors


class Frames:
    """Class to represent frames in the video."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        """Initialize the Frames object.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to represent.

        """
        self.frames = [Frame(frame) for frame in frames]

    def __getitem__(self, index) -> Frame:
        """Get the frame at the specified index.

        Parameters
        ----------
        index : int
            The index of the frame to get.

        Returns
        -------
        Frame
            The frame at the specified index.

        """
        return self.frames[index]

    def __len__(self) -> int:
        """Get the number of frames.

        Returns
        -------
        int
            The number of frames.

        """
        return len(self.frames)
