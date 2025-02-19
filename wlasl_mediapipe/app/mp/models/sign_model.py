import os
from typing import List

import numpy as np

from wlasl_mediapipe.app.mp.models.globals import GlobalFilters
from wlasl_mediapipe.app.mp.models.hand_model import HandModel
from wlasl_mediapipe.app.utils.mp.file_utils import load_array


class SignModel(object):
    def __init__(
        self,
        left_hand_list: List[List[float]],
        right_hand_list: List[List[float]],
        pose_list: List[List[float]] = [],
        face_list: List[List[float]] = [],
        expand_keypoints: bool = False,
        all_features: bool = True,
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
        self.has_pose = np.sum(pose_list) != 0
        self.has_face = np.sum(face_list) != 0

        self.left_hand_list = left_hand_list
        self.right_hand_list = right_hand_list
        self.pose_list = pose_list
        self.face_list = face_list

        if expand_keypoints:
            self.expand_keypoints(left_hand_list, right_hand_list)

        self.lh_matrix = np.reshape(left_hand_list, (-1, 21, 3))
        self.rh_matrix = np.reshape(right_hand_list, (-1, 21, 3))
        if all_features:
            self.pose_matrix = np.reshape(
                self._filter_frames_feature_list(
                    pose_list, GlobalFilters().pose_filter
                ),
                (-1, 6, 3),
            )
            self.face_matrix = np.reshape(
                self._filter_frames_feature_list(
                    face_list, GlobalFilters().face_filter_big
                ),
                (-1, 132, 3),
            )

    def expand_keypoints(self, left_hand_list, right_hand_list) -> None:
        self.lh_embedding = self._get_hand_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_hand_embedding_from_landmark_list(right_hand_list)

    @staticmethod
    def load(
        video_id: str, expand_keypoints: bool = False, all_features: bool = True
    ) -> "SignModel":
        """
        Load a SignModel from a file
        """
        path = os.path.join("data", "mp", video_id)
        left_hand_list = load_array(os.path.join(path, f"lh_{video_id}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_id}.pickle"))
        pose_list = load_array(os.path.join(path, f"pose_{video_id}.pickle"))
        face_list = load_array(os.path.join(path, f"face_{video_id}.pickle"))
        return SignModel(
            left_hand_list.tolist(),
            right_hand_list.tolist(),
            pose_list.tolist(),
            face_list.tolist(),
            expand_keypoints=expand_keypoints,
            all_features=all_features,
        )

    @staticmethod
    def _get_hand_embedding_from_landmark_list(
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
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                embedding.append(np.zeros(21 * 21))
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
        return embedding

    @staticmethod
    def _filter_frames_feature_list(
        frames_feature_list: List[List[float]], filter: List[int]
    ) -> List[List[float]]:
        new_frames_feature_list = []
        for frame_feature in frames_feature_list:
            reshaped_frame_feature = np.array(frame_feature).reshape(
                len(frame_feature) // 3, 3
            )
            features_to_keep = reshaped_frame_feature[filter]
            new_frames_feature_list.append(features_to_keep.flatten())
        return new_frames_feature_list
