import os
from typing import List

import numpy as np

from wlasl_mediapipe.app.mp.models.hand_model import HandModel
from wlasl_mediapipe.app.utils.mp.file_utils import load_array


class SignModel(object):
    def __init__(
        self, left_hand_list: List[List[float]], right_hand_list: List[List[float]]
    ):
        """
        Params
            x_hand_list: List of all landmarks for each frame of a video
        Args
            has_x_hand: bool; True if x hand is detected in the video, otherwise False
            xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
        """
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)

    @staticmethod
    def load(video_id: str) -> "SignModel":
        """
        Load a SignModel from a file
        """
        path = os.path.join("data", "mp", video_id)
        left_hand_list = load_array(os.path.join(path, f"lh_{video_id}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_id}.pickle"))
        return SignModel(left_hand_list, right_hand_list)



    @staticmethod
    def _get_embedding_from_landmark_list(
        hand_list: List[List[float]],
    ) -> List[List[float]]:
        """
        Params
            hand_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        # print(f"hand_list: {hand_list}")
        # print({"hand_list shape": np.array(hand_list).shape})
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
        return embedding
