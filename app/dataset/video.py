from typing import List, Tuple
import os
import cv2
import numpy as np

class Video:
    def __init__(self, gloss: str, video_id: str, split: str, frame_start: int, frame_end: int, fps:int, bbox: List[int]) -> None:
        self.video_capture = None
        self.gloss = gloss
        self.video_id = video_id
        self.split = split
        self.fps = fps
        self.bbox = bbox
        self.frame_start = frame_start
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
    
    def get_frame(self, frame_number: int) -> Tuple[bool, 'np.ndarray']:
        self.get_video_capture().set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.get_video_capture().read()
        return ret, frame
    
    def get_frames(self) -> List['np.ndarray']:
        frames = []
        frame_number = self.frame_start
        while True:
            ret, frame = self.get_frame(frame_number)
            if ret == False:
                break
            frames.append(frame)
            frame_number += 1
        return frames
    
    def get_bounded_frames(self) -> List['np.ndarray']:
        frames = []
        for frame in self.get_frames():
            frames.append(frame[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]])
        return frames

