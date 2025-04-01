from typing import Callable

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from handcrafted.app.dataset.dataset_loader import Frame, Signer


class FramesSplitter:
    def __init__(
        self,
        signers: dict[str, Signer],
        val_split: float = 0.2,
        test_split: float = 0.2,
        frames_to_extract: int = 500,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self._signers = signers
        self._val_split = val_split
        self._test_split = test_split
        self._frames_to_extract = frames_to_extract
        self._seed = seed

    def split(
        self, X_content: Callable[[Frame], Frame | str] = lambda x: x.path
    ):
        X_train = []
        X_train_aug = []
        y_train = []
        X_val = []
        X_val_aug = []
        y_val = []
        X_test = []
        X_test_aug = []
        y_test = []
        for signer in tqdm(self._signers.values()):
            signer_frames_num = signer.get_frames_num()
            train_videos, val_videos, test_videos = signer.split(
                self._val_split, self._test_split, random_state=self._seed
            )
            for video in train_videos:
                num_aug = 0
                if self._frames_to_extract > signer_frames_num:
                    num_aug = int(self._frames_to_extract / signer_frames_num)
                for frame in video.extract_frames(
                    self._frames_to_extract, self._seed
                ):
                    X_train.append(X_content(frame))
                    X_train_aug.append(num_aug)
                    y_train.append(signer.id)
            for video in val_videos:
                for frame in video.extract_frames(
                    self._frames_to_extract, self._seed
                ):
                    X_val.append(X_content(frame))
                    X_val_aug.append(0)
                    y_val.append(signer.id)
            for video in test_videos:
                for frame in video.extract_frames(
                    self._frames_to_extract, self._seed
                ):
                    X_test.append(X_content(frame))
                    X_test_aug.append(0)
                    y_test.append(signer.id)
        X_train, X_train_aug, y_train = shuffle(
            X_train, X_train_aug, y_train, random_state=self._seed
        )
        X_val, X_val_aug, y_val = shuffle(
            X_val, X_val_aug, y_val, random_state=self._seed
        )
        X_test, X_test_aug, y_test = shuffle(
            X_test, X_test_aug, y_test, random_state=self._seed
        )
        return (
            np.array(X_train),
            np.array(X_train_aug, dtype=np.uint16),
            y_train,
            np.array(X_val),
            np.array(X_val_aug, dtype=np.uint16),
            y_val,
            np.array(X_test),
            np.array(X_test_aug, dtype=np.uint16),
            y_test,
        )
