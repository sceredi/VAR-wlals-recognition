# Purpose: Extracts a region of interest from a list of frames
import numpy as np
from typing import List

class RoiExtractor:
    def __init__(self, frames: List["np.ndarray"], bbox: List[int]) -> None:
        self.frames = frames
        self.bbox = bbox

    def extract(self) -> List["np.ndarray"]:
        frames = []
        for frame in self.frames:
            frames.append(
                frame[self.bbox[1] : self.bbox[3], self.bbox[0] : self.bbox[2]]
            )
        return frames
