"""Utility functions to extract the landmarks from the mediapipe object."""

import numpy as np


def landmark_to_array(mp_landmark_list) -> np.ndarray:
    """Convert the mediapipe landmark list to a numpy array.

    Parameters
    ----------
    mp_landmark_list : mediapipe object
        The mediapipe object that contains the landmarks.

    Returns
    -------
    np.ndarray
        The landmarks in a numpy array.

    """
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size if a hand doesn't appear, return an array of zeros.

    Parameters
    ----------
    results : mediapipe object
        The mediapipe object that contains the 3D position of all keypoints.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The first np array has size 1404 and contains the face landmarks.
        The second np array has size 99 and contains the pose landmarks.
        The third np array has size 63 and contains the left hand landmarks.
        The fourth np array has size 63 and contains the right hand landmarks.

    """
    face = np.zeros(1404).tolist()
    if results.face_landmarks:
        face = landmark_to_array(results.face_landmarks).reshape(1404).tolist()
    pose = np.zeros(99).tolist()
    if results.pose_landmarks:
        pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = (
            landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()
        )

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks)
            .reshape(63)
            .tolist()
        )
    return pose, face, left_hand, right_hand
