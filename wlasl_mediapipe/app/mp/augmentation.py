"""Augmentation helper functions for the MediaPipe model."""

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from wlasl_mediapipe.app.mp.models.sign_model import SignModel


def _rotate_hand(data: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate the hand data.

    Parameters
    ----------
    data : np.ndarray
        The hand data.
    rotation_matrix : np.ndarray
        The rotation matrix.

    Returns
    -------
    np.ndarray
        The rotated hand data.

    """
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    data = data.reshape(-1, 3)
    data -= center
    data = np.dot(data, rotation_matrix.T)
    data += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _rotate(data: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate the data.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.
    rotation_matrix : np.ndarray
        The rotation matrix.

    Returns
    -------
    np.ndarray
        The rotated data.

    """
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    data = data.reshape(-1, 3)
    data -= center
    data = np.dot(data, rotation_matrix.T)
    data += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _rotate_z(data: np.ndarray) -> np.ndarray:
    """Rotate the data around the z-axis.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The rotated data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_z_hands(
    lh: np.ndarray, rh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the hands around the z-axis.

    Parameters
    ----------
    lh : np.ndarray
        The left hand data.
    rh : np.ndarray
        The right hand data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotated left and right hand data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _rotate_y(data: np.ndarray) -> np.ndarray:
    """Rotate the data around the y-axis.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The rotated data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_y_hands(
    lh: np.ndarray, rh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the hands around the y-axis.

    Parameters
    ----------
    lh : np.ndarray
        The left hand data.
    rh : np.ndarray
        The right hand data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotated left and right hand data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _rotate_x(data: np.ndarray) -> np.ndarray:
    """Rotate the data around the x-axis.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The rotated data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_x_hands(
    lh: np.ndarray, rh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the hands around the x-axis.

    Parameters
    ----------
    lh : np.ndarray
        The left hand data.
    rh : np.ndarray
        The right hand data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotated left and right hand data.

    """
    angle = np.random.choice(
        [np.random.uniform(-30, -10), np.random.uniform(10, 30)]
    )
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _zoom(data: np.ndarray) -> np.ndarray:
    """Zoom the data.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The zoomed data.

    """
    factor = np.random.uniform(0.8, 1.2)
    center = np.array([0.5, 0.5])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] = (
        data[non_zero[:, 0], non_zero[:, 1], :2] - center
    ) * factor + center
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _shift(data: np.ndarray) -> np.ndarray:
    """Shift the data.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The shifted data.

    """
    x_shift = np.random.uniform(-0.2, 0.2)
    y_shift = np.random.uniform(-0.2, 0.2)
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], 0] += x_shift
    data[non_zero[:, 0], non_zero[:, 1], 1] += y_shift
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _mask(data: np.ndarray) -> np.ndarray:
    """Mask some of the keypoints.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The masked data.

    """
    _, landmarks, _ = data.shape
    num_hands = int(0.3 * 42)
    num_rest = int(0.6 * (landmarks - 42))

    mask = np.zeros(landmarks, dtype=bool)
    indices = np.concatenate(
        [
            np.random.choice(42, num_hands, replace=False),
            np.random.choice(landmarks - 42, num_rest, replace=False) + 42,
        ]
    )
    mask[indices] = True
    data[:, mask] = 0
    return data


def hflip(data: np.ndarray) -> np.ndarray:
    """Flip the data horizontally.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The horizontally flipped data.

    """
    data[:, :, 0] = 1 - data[:, :, 0]
    return data


def speedup(data: np.ndarray) -> np.ndarray:
    """Speedup the data by removing every second frame.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The sped up data.

    """
    return data[::2]


def _apply_augmentations(data: np.ndarray) -> np.ndarray:
    """Apply augmentations to the data.

    Parameters
    ----------
    data : np.ndarray
        The concatenated left and right hand data with pose and possibly face keypoints.

    Returns
    -------
    np.ndarray
        The augmented data.

    """
    aug_functions = [
        _rotate_x,
        _rotate_y,
        _rotate_z,
        _zoom,
        _shift,
        _mask,
        hflip,
        speedup,
    ]
    np.random.shuffle(np.array(aug_functions))
    counter = 0
    for fun in aug_functions:
        if np.random.rand() < 0.5:
            data = fun(data)
            counter += 1
    if counter == 0:
        data = _apply_augmentations(data)
    return data


def _apply_lhrh_augmentations(
    lh: np.ndarray, rh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentations to the left and right hand data.

    Parameters
    ----------
    lh : np.ndarray
        The left hand data.
    rh : np.ndarray
        The right hand data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The augmented left and right hand data.

    """
    aug_functions = [
        _rotate_x_hands,
        _rotate_y_hands,
        _rotate_z_hands,
    ]
    np.random.shuffle(np.array(aug_functions))
    counter = 0
    for fun in aug_functions:
        if np.random.rand() < 0.5:
            lh, rh = fun(lh, rh)
            counter += 1
    if counter == 0:
        lh, rh = _apply_lhrh_augmentations(lh, rh)
    return lh, rh


def augment(
    X: List[np.ndarray], Y: List[str], num: int | None = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Augment the data.

    Parameters
    ----------
    X : List[np.ndarray]
        The data.
    Y : List[str]
        The labels.
    num : int, optional
        The number of augmentations to apply, by default None.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        The augmented data and labels.

    """
    X_aug = X.copy()
    Y_aug = Y.copy()

    if num is None:
        for i in tqdm(range(len(Y)), ncols=100):
            num_aug = np.random.choice([1, 2, 3])
            for _ in range(num_aug):
                X_aug.append(_apply_augmentations(X[i].copy()))
                Y_aug.append(Y[i])
    elif num > 0:
        for i in tqdm(range(len(Y)), ncols=100):
            for _ in range(num):
                X_aug.append(_apply_augmentations(X[i].copy()))
                Y_aug.append(Y[i])

    return X_aug, Y_aug


def augment_video(video, num: int = 1):
    """Augment the video.

    Parameters
    ----------
    video : MediapipeVideo
        The video.
    num : int, optional
        The number of augmentations to apply, by default 1.

    Returns
    -------
    List[MediapipeVideo]
        The augmented videos.

    """
    aug_videos = []
    for _ in range(num):
        lh = video.sign_model.lh_matrix.copy()
        rh = video.sign_model.rh_matrix.copy()
        lh, rh = _apply_lhrh_augmentations(lh, rh)
        newSM = SignModel(
            lh.reshape(-1, 63).tolist(),
            rh.reshape(-1, 63).tolist(),
            expand_keypoints=True,
            all_features=False,
        )
        new_video = video.from_sign_model(newSM)
        aug_videos.append(new_video)

    return aug_videos
