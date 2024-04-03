from typing import List

import mediapipe as mp
import numpy as np

from wlasl_mediapipe.app.mp.models.globals import GlobalFilters


class FaceModel(object):
    """
    Class that contains the face information for an image
    A bit different from the HandModel, this class does not contain the connections, but calculates the feature vector
    directly from the landmarks and the angle between each one of them
    """

    def __init__(self, landmarks: List[float]):
        self.connections = mp.solutions.holistic.FACEMESH_TESSELATION
        landmarks = np.array(landmarks).reshape((468, 3))
        self.feature_vector = self._get_feature_vector(landmarks)

    def _get_connections_from_landmarks(
        self, landmarks: np.ndarray
    ) -> List[np.ndarray]:
        return list(
            map(
                lambda t: landmarks[t[1]] - landmarks[t[0]],
                filter(
                    lambda c: c[0] in GlobalFilters().face_filter
                    and c[1] in GlobalFilters().face_filter,
                    self.connections,
                ),
            )
        )

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        connections = self._get_connections_from_landmarks(landmarks)
        angles_list = []
        for connection_from in connections:
            for connection_to in connections:
                angle = self._get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not NaN we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
