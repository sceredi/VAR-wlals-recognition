import numpy as np
from sklearn.utils import shuffle

from handcrafted.app.dataset.dataset_loader import Signer


class FramesSplitter:
    def __init__(
        self,
        signers: dict[str, Signer],
        val_split: float = 0.2,
        test_split: float = 0.2,
        frames_split: float = 0.1,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self._signers = signers
        self._val_split = val_split
        self._test_split = test_split
        self._frames_split = frames_split
        self._seed = seed

    def split(self):
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        for signer in self._signers.values():
            train_videos, val_videos, test_videos = signer.split(
                self._val_split, self._test_split, random_state=self._seed
            )
            for video in train_videos:
                for frame in video.extract_frames(
                    self._frames_split, self._seed
                ):
                    X_train.append(frame.path)
                    y_train.append(signer.id)
            for video in val_videos:
                for frame in video.extract_frames(
                    self._frames_split, self._seed
                ):
                    X_val.append(frame.path)
                    y_val.append(signer.id)
            for video in test_videos:
                for frame in video.extract_frames(
                    self._frames_split, self._seed
                ):
                    X_test.append(frame.path)
                    y_test.append(signer.id)
        X_train, y_train = shuffle(X_train, y_train, random_state=self._seed)
        X_val, y_val = shuffle(X_val, y_val, random_state=self._seed)
        X_test, y_test = shuffle(X_test, y_test, random_state=self._seed)
        return (
            np.array(X_train),
            y_train,
            np.array(X_val),
            y_val,
            np.array(X_test),
            y_test,
        )
