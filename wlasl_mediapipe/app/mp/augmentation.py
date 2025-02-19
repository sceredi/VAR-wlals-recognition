import numpy as np
from tqdm import tqdm

from wlasl_mediapipe.app.mp.models.sign_model import SignModel


def _rotate_hand(data, rotation_matrix):
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data = data.reshape(-1, 3)
    data -= center
    data = np.dot(data, rotation_matrix.T)
    data += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _rotate(data, rotation_matrix):
    # TODO: may be wrong check non_zero
    frames, landmarks, _ = data.shape
    center = np.array([0.5, 0.5, 0])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data = data.reshape(-1, 3)
    # data[non_zero] -= center
    # data[non_zero] = np.dot(data[non_zero], rotation_matrix.T)
    # data[non_zero] += center
    data -= center
    data = np.dot(data, rotation_matrix.T)
    data += center
    data = data.reshape(frames, landmarks, 3)
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _rotate_z(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_z_hands(lh, rh):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _rotate_y(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_y_hands(lh, rh):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _rotate_x(data):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return _rotate(data, rotation_matrix)


def _rotate_x_hands(lh, rh):
    angle = np.random.choice([np.random.uniform(-30, -10), np.random.uniform(10, 30)])
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return _rotate_hand(lh, rotation_matrix), _rotate_hand(rh, rotation_matrix)


def _zoom(data):
    factor = np.random.uniform(0.8, 1.2)
    center = np.array([0.5, 0.5])
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], :2] = (
        data[non_zero[:, 0], non_zero[:, 1], :2] - center
    ) * factor + center
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _shift(data):
    x_shift = np.random.uniform(-0.2, 0.2)
    y_shift = np.random.uniform(-0.2, 0.2)
    non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
    data[non_zero[:, 0], non_zero[:, 1], 0] += x_shift
    data[non_zero[:, 0], non_zero[:, 1], 1] += y_shift
    out_of_range = np.any((data[:, :, :2] < 0) | (data[:, :, :2] > 1), axis=2)
    data[out_of_range] = 0
    return data


def _mask(data):
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


def hflip(data):
    data[:, :, 0] = 1 - data[:, :, 0]
    return data


def speedup(data):
    return data[::2]


def _apply_augmentations(data):
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


def _apply_lhrh_augmentations(lh, rh):
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


def augment(X, Y, num=None):
    X_aug = X.copy()
    Y_aug = Y.copy()

    if num == None:
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


def augment_video(video, num=1):
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
