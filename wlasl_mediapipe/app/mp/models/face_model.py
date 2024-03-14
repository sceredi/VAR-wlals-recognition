import os
from typing import List

from wlasl_mediapipe.app.utils.mp.file_utils import load_array

class FaceModel(object):
    def __init__(self, pose_list: List[List[float]]):
        self.pose_list = pose_list

    @staticmethod
    def load(video_id: str) -> "FaceModel":
        """
        Load a FaceModel from a file
        """
        path = os.path.join("data", "mp", video_id)
        pose_list = load_array(os.path.join(path, f"pose_{video_id}.pickle"))
        return FaceModel(pose_list)