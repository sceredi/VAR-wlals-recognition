from typing import List
import os
import cv2

class Video:
    def __init__(self, gloss: str, video_id: str, split: str, frame_start: int, frame_end: int, fps:int, bbox: List[int]) -> None:
        self.gloss = gloss
        self.video_id = video_id
        self.split = split
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.fps = fps
        self.bbox = bbox

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

    def get_path(self) -> str:
        return f"data/videos/{self.video_id}.mp4"

    def get_video_capture(self) -> 'cv2.VideoCapture':
        return cv2.VideoCapture(self.get_path())
    
    def is_missing(self) -> bool:
        return os.path.isfile(self.get_path()) == False

