# Purpose: Extracts a region of interest from a list of frames
import numpy as np
from typing import List
from rembg import remove


class RoiExtractor:
    def __init__(self, frames: List["np.ndarray"], bbox: List[int]) -> None:
        self.frames = frames
        self.bbox = bbox

    # Extracts the region of interest from the frames
    # If remove_background is True, then the background will be removed 
    # this will slow the computation down as it uses a unet model to do so
    def extract(self, remove_background=False) -> List["np.ndarray"]:
        frames = []
        for frame in self.frames:
            frame = frame[self.bbox[1] : self.bbox[3], self.bbox[0] : self.bbox[2]]
            if remove_background:
                frame = self.remove_bg(frame)
            frames.append(frame)
        return frames

    def remove_bg(self, frame: "np.ndarray"):
        return remove(frame)
