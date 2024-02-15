from typing import List, Tuple
import os
from typing_extensions import deprecated
import cv2
import numpy as np
from scipy.interpolate import interp1d

from app.features.features_container import FeaturesContainer


class Video:
    def __init__(
        self,
        gloss: str,
        video_id: str,
        split: str,
        frame_start: int,
        frame_end: int,
        fps: int,
        bbox: List[int],
    ) -> None:
        self.video_capture = None
        self.gloss = gloss
        self.video_id = video_id
        self.split = split
        self.fps = fps
        self.bbox = bbox
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.features_container = FeaturesContainer(self)

    @classmethod
    def from_instance(cls, gloss: str, instance: dict) -> "Video":
        return cls(
            gloss,
            instance["video_id"],
            instance["split"],
            instance["frame_start"],
            instance["frame_end"],
            instance["fps"],
            instance["bbox"],
        )

    def __str__(self) -> str:
        return f"image(gloss={self.gloss}, video_id={self.video_id}, split={self.split}, frame_start={self.frame_start}, frame_end={self.frame_end}, fps={self.fps}, bbox={self.bbox})\n"

    def __repr__(self) -> str:
        return str(self)

    def is_missing(self) -> bool:
        return os.path.isfile(self.get_path()) == False

    def get_path(self) -> str:
        return f"../data/videos/{self.video_id}.mp4"

    def get_video_capture(self) -> "cv2.VideoCapture":
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.get_path())
        return self.video_capture

    def get_frame(self, frame_number: int) -> Tuple[bool, "np.ndarray"]:
        self.get_video_capture().set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.get_video_capture().read()
        return ret, frame

    def get_frames(self, last_frame = None) -> List["np.ndarray"]:
        if last_frame is None:
            last_frame = self.frame_end
        frames = []
        frame_number = self.frame_start
        while frame_number < last_frame:
            ret, frame = self.get_frame(frame_number)
            if not ret:
                break
            frames.append(frame)
            frame_number += 1
        return frames

    def get_end(self) -> int:
        if self.frame_end != -1:
            return self.frame_end
        ret = self.get_video_capture().get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Video {self.video_id} has {ret} frames")
        self.video_capture.release()
        return int(ret)

    def __len__(self) -> int:
        return len(self.get_frames())

    @deprecated("The dataset is now padded to 232 frames")
    def get_frames_padded(self, frames, target_num_frames=232):
        num_frames_to_add = target_num_frames - len(frames)
        last_frame = frames[-1]
        pad_frames = np.tile(last_frame, (num_frames_to_add, 1, 1, 1))
        new_frames = np.concatenate([frames, pad_frames])
        return new_frames

    def get_frames_interpolated(
        self, frames, target_num_frames=232
    ) -> List["np.ndarray"]:
        x_old = np.linspace(0, 1, len(frames))
        x_new = np.linspace(0, 1, target_num_frames)

        # Calcola l'interpolatore per il video corrente
        interpolator = interp1d(
            x_old, frames, kind="linear", axis=None, fill_value="extrapolate"
        )

        # Applica l'interpolazione al nuovo vettore di tempo
        new_frames = interpolator(x_new)
        new_frames = (new_frames * 255.0).astype(np.uint8)

        return new_frames
