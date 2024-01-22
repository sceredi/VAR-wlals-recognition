from typing import List
import os
import cv2
import numpy as np

class Video:
    def __init__(self, gloss: str, video_id: str, split: str, frame_start: int, frame_end: int, fps:int, bbox: List[int]) -> None:
        self.gloss = gloss
        self.video_id = video_id
        self.split = split
        self.fps = fps
        self.bbox = bbox
        self.frame_start = frame_start
        if frame_end == -1:
            frame_end = int(self.get_video_capture().get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_end = frame_end

    @classmethod
    def from_instance(cls, gloss: str, instance: dict) -> 'Video':
        return cls(
            gloss,
            instance['video_id'],
            instance['split'],
            instance['frame_start'],
            instance['frame_end'],
            instance['fps'],
            instance['bbox']
        )

    def __str__(self) -> str:
        return f"image(gloss={self.gloss}, video_id={self.video_id}, split={self.split}, frame_start={self.frame_start}, frame_end={self.frame_end}, fps={self.fps}, bbox={self.bbox})\n"
    
    def __repr__(self) -> str:
        return str(self)

    def is_missing(self) -> bool:
        return os.path.isfile(self.get_path()) == False

    def get_path(self) -> str:
        return f"data/videos/{self.video_id}.mp4"

    def get_video_capture(self) -> 'cv2.VideoCapture':
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.get_path())
        return self.video_capture
    
    def get_frame(self, frame_number: int) -> 'np.ndarray':
        self.get_video_capture().set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.get_video_capture().read()
        if ret == False:
            raise Exception(f"Error reading frame {frame_number} from video {self.video_id}")
        return frame
    
    def get_frames(self) -> List['np.ndarray']:
        ret = []
        for frame_number in range(self.frame_start, self.frame_end + 1):
            ret.append(self.get_frame(frame_number))
        return ret

