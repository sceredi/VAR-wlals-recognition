"""Utility functions for drawing landmarks on images."""

import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_holistic = mp.solutions.holistic  # type: ignore


def draw_landmarks_on_image(image: np.ndarray, results) -> np.ndarray:
    """Draws the landmarks on the image.

    Parameters
    ----------
    image : np.ndarray
        The image to draw the landmarks on.
    results : mediapipe object
        The mediapipe object that contains the landmarks.

    Returns
    -------
    np.ndarray
        The image with the landmarks drawn on it.
    """
    annotated_image = image.copy()

    # Loop through the detected hands to visualize.
    if results:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image
