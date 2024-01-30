# Purpose: Extracts a region of interest from a list of frames
import numpy as np
from typing import List, Tuple
from rembg import remove
import cv2


class RoiExtractor:
    def __init__(
        self, frames: List["np.ndarray"], bbox: List[int], resize: None | int = None
    ) -> None:
        self.frames = frames
        self.bbox = bbox
        self.resize = resize

    # Extracts the region of interest from the frames
    # If remove_background is True, then the background will be removed
    # this will slow the computation down as it uses a unet model to do so
    def extract(self, remove_background=False) -> List["np.ndarray"]:
        frames = []
        for frame in self.frames:
            frame = frame[self.bbox[1] : self.bbox[3], self.bbox[0] : self.bbox[2]]
            if remove_background:
                frame = self._remove_bg(frame)
            frame = self._resize(frame)
            frames.append(frame)
        return frames

    def _remove_bg(self, frame: "np.ndarray"):
        return remove(frame)

    def _resize(self, frame: "np.ndarray") -> "np.ndarray":
        if self.resize is None:
            return frame
        if frame.shape[0] >= frame.shape[1]:
            height = self.resize
            width = int(frame.shape[1] * self.resize / frame.shape[0])
        else:
            width = self.resize
            height = int(frame.shape[0] * self.resize / frame.shape[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frame = self._pad(frame)
        return frame

    def _pad(self, frame: "np.ndarray") -> "np.ndarray":
        if self.resize is None:
            return frame
        pad_width = self.resize - frame.shape[0]
        pad_height = self.resize - frame.shape[1]
        my_padded_image = np.pad(
            frame, ((0, pad_width), (0, pad_height), (0, 0)), "constant"
        )
        return my_padded_image
