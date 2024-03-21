import os
from typing import List

from wlasl_mediapipe.app.utils.mp.file_utils import load_array


class FaceModel(object):
    def __init__(self, face_list: List[List[float]]):
        self.face_list = face_list

    @staticmethod
    def load(video_id: str) -> "FaceModel":
        """
        Load a FaceModel from a file
        """
        path = os.path.join("data", "mp", video_id)
        face_list = load_array(os.path.join(path, f"face_{video_id}.pickle"))
        return FaceModel(face_list)
