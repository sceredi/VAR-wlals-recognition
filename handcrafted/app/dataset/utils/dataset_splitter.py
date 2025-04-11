"""Module for splitting a dataset of signer frames into train, validation, and test sets."""

import numpy as np
from sklearn.model_selection import train_test_split

from handcrafted.app.dataset.signer_frames import SignerFrame
from handcrafted.app.dataset.utils.augmentation import DataAugmentation
from handcrafted.app.dataset.video import Video


class SignerDatasetSplitter:
    """Class to split a dataset of signer frames into train, validation, and test sets."""

    def __init__(
        self,
        videos: list[Video],
        frames_split: float = 0.3,
        seed: int = 42,
        extract_features: bool = True,
    ):
        """Initialize the SignerDatasetSplitter object.

        Parameters
        ----------
        videos : list[Video]
            The list of videos to split.
        frames_split : float, optional
            The fraction of frames to split, by default 0.3.
        seed : int, optional
            The random seed for reproducibility, by default 42.
        extract_features : bool, optional
            Whether to extract features from the frames, by default True.

        """
        np.random.seed(seed)
        self.videos = videos
        self._frames_split = frames_split
        self._extract_features = extract_features

    def _signer_dataset(self):
        """Create a dataset of signer frames.

        Returns
        -------
        list[SignerFrame]
            The list of signer frames.

        """
        signer_frames = []
        for video in self.videos:
            signer_id = video.signer_id
            frames_in_video = video.get_frames(
                split_size=self._frames_split, remove_bg=False
            )
            for frame in frames_in_video:
                signer_frames.append(
                    SignerFrame(
                        frame,
                        signer_id,
                        extract_features=self._extract_features,
                    )
                )
        np.random.shuffle(signer_frames)
        return signer_frames

    def train_test_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        """Split the dataset into train, validation, and test sets.

        Parameters
        ----------
        test_size : float, optional
            The fraction of the dataset to use for the test set, by default 0.2.
        val_size : float, optional
            The fraction of the dataset to use for the validation set, by default 0.2.
        random_state : int, optional
            The random seed for reproducibility, by default 42.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test sets.

        """
        signer_frames = self._signer_dataset()

        train_frames, test_frames = train_test_split(
            signer_frames, test_size=test_size, random_state=random_state
        )

        if val_size > 0:
            train_frames, val_frames = train_test_split(
                train_frames, test_size=val_size, random_state=random_state
            )
        else:
            return train_frames, test_frames

        return train_frames, val_frames, test_frames

    @staticmethod
    def apply_data_augmentation(frames, num_augmentations: int = 3):
        """Apply data augmentation to the given frames.

        Parameters
        ----------
        frames : list[SignerFrame]
            The list of frames to augment.
        num_augmentations : int, optional
            The number of augmentations to apply, by default 3.

        Returns
        -------
        list[SignerFrame]
            The list of augmented frames.

        """
        augmentation = DataAugmentation(
            frames, num_augmentations=num_augmentations
        )
        augmented_frames = augmentation.random_augmentations()
        return augmented_frames
