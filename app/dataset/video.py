from typing import List
import os

class Video:
    def __init__(self, video_id: str, split: str, frame_start: int, frame_end: int, fps:int, bbox: List[int]) -> None:
        self.video_id = video_id
        self.split = split
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.fps = fps
        self.bbox = bbox

    def __str__(self) -> str:
        return f"Image(video_id={self.video_id}, split={self.split}, frame_start={self.frame_start}, frame_end={self.frame_end}, fps={self.fps}, bbox={self.bbox})"
    
    def __repr__(self) -> str:
        return str(self)

    def get_path(self) -> str:
        return f"../data/videos/{self.video_id}.mp4"
    
    def isMissing(self) -> bool:
        return os.path.isfile(self.get_path()) == False

