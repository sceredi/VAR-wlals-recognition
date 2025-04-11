"""Module that contains the HandModel class."""

from typing import List

import mediapipe as mp
import numpy as np


class HandModel(object):
    """Object that contains the hand gesture information for an image."""

    def __init__(self, landmarks: List[float]):
        """Initialize a HandModel object.

        Parameters
        ----------
        landmarks : List[float]
            List of positions for the hands.

        """
        # Define the connections
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS  # type: ignore

        # Create feature vector (list of the angles between all the connections)
        landmarks = np.array(landmarks).reshape((21, 3))  # type: ignore
        self.feature_vector = self._get_feature_vector(landmarks)  # type: ignore

    def _get_feature_vector(self, landmarks: np.ndarray) -> List[float]:
        """Get the feature vector for the hand landmarks.

        Parameters
        ----------
        landmarks : np.ndarray
            The landmarks for the hand.

        Returns
        -------
        List[float]
            List of the angles between the connections.

        """
        connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []
        for connection_from in connections:
            for connection_to in connections:
                angle = self._get_angle_between_vectors(
                    connection_from, connection_to
                )
                # If the angle is not NaN we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        return angles_list

    def _get_connections_from_landmarks(
        self, landmarks: np.ndarray
    ) -> List[np.ndarray]:
        """Get the connections from the landmarks.

        Parameters
        ----------
        landmarks : np.ndarray
            The landmarks for the hand.

        Returns
        -------
        List[np.ndarray]
            List of vectors representing hand connections.

        """
        return [landmarks[t[1]] - landmarks[t[0]] for t in self.connections]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """Calculate the angle between two vectors.

        Parameters
        ----------
        u : np.ndarray
            Vector representing the first element of a connection.
        v : np.ndarray
            Vector representing the second element of a connection.

        Returns
        -------
        float
            Angle between the two vectors in radians.

        """
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)
