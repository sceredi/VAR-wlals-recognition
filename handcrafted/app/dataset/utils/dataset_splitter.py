import numpy as np
from sklearn.model_selection import train_test_split

from handcrafted.app.dataset.signer_frames import SignerFrame
from handcrafted.app.dataset.utils.augmentation import DataAugmentation
from handcrafted.app.dataset.video import Video


class SignerDatasetSplitter:
    def __init__(
        self,
        videos: list[Video],
        frames_split: float = 0.3,
        seed: int = 42,
        extract_features: bool = True,
    ):
        np.random.seed(seed)
        self.videos = videos
        self._frames_split = frames_split
        self._extract_features = extract_features

    def _signer_dataset(self):
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
        augmentation = DataAugmentation(
            frames, num_augmentations=num_augmentations
        )
        augmented_frames = augmentation.random_augmentations()
        return augmented_frames
