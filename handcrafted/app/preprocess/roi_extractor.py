"""Module to extract the region of interest from the frames."""

from typing import List

import cv2
import numpy as np
from rembg import remove


class RoiExtractor:
    """Class to extract the region of interest from the frames."""

    def __init__(
        self,
        frames: List["np.ndarray"],
        bbox: List[int],
        resize: None | int = None,
    ) -> None:
        """Initialize the RoiExtractor object.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        bbox : List[int]
            The bounding box to extract the region of interest.
        resize : int, optional
            The size to resize the frames to, by default None.

        """
        self.frames = frames
        self.bbox = bbox
        self.resize = resize

    def extract(self, remove_background=False) -> List["np.ndarray"]:
        """Extract the region of interest from the frames.

        Parameters
        ----------
        remove_background : bool, optional
            Whether to remove the background, by default False.

        Returns
        -------
        List[np.ndarray]
            The extracted frames.

        """
        frames = []
        for frame in self.frames:
            frame = frame[
                self.bbox[1] : self.bbox[3], self.bbox[0] : self.bbox[2]
            ]
            if remove_background:
                frame = self._remove_bg(frame)
            frame = self._resize(frame)
            frames.append(frame)
        return frames

    def _remove_bg(self, frame: "np.ndarray"):
        """Remove the background from the frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to process.

        Returns
        -------
        np.ndarray
            The frame with the background removed.

        """
        return remove(frame)

    def _resize(self, frame) -> "np.ndarray":
        """Resize the frame to the given size.

        Parameters
        ----------
        frame : np.ndarray
            The frame to resize.

        Returns
        -------
        np.ndarray
            The resized frame.

        """
        if self.resize is None:
            return frame
        if frame.shape[0] >= frame.shape[1]:
            height = self.resize
            width = int(frame.shape[1] * self.resize / frame.shape[0])
        else:
            width = self.resize
            height = int(frame.shape[0] * self.resize / frame.shape[1])
        frame = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_LINEAR
        )
        frame = self._pad(frame)
        return frame

    def _pad(self, frame: "np.ndarray") -> "np.ndarray":
        """Pad the frame to the given size.

        Parameters
        ----------
        frame : np.ndarray
            The frame to pad.

        Returns
        -------
        np.ndarray
            The padded frame.

        """
        if self.resize is None:
            return frame
        pad_width = self.resize - frame.shape[0]
        pad_height = self.resize - frame.shape[1]
        my_padded_image = np.pad(
            frame, ((0, pad_width), (0, pad_height), (0, 0)), "constant"
        )
        return my_padded_image
