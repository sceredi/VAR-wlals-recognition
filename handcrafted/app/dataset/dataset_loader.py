"""Module for loading the signers dataset."""

import math
import os

import cv2
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm


class DatasetLoader:
    """Class to load the dataset of signers."""

    def __init__(
        self,
        directory: str = "./data/frames_no_bg/",
    ) -> None:
        """Initialize the DatasetLoader object.

        Parameters
        ----------
        directory : str
            The directory where the dataset is stored.

        """
        self._path = directory
        self.signers = self._load_dataset()
        self.num_signers = len(self.signers)

    def _load_dataset(self):
        """Load the dataset from the directory.

        Returns
        -------
        dict
            A dictionary of signers, where the key is the signer id and the value is a Signer object.

        """
        signers = {}
        for dirname, _, filenames in tqdm(os.walk(self._path)):
            for filename in filenames:
                dirname = dirname.removeprefix(self._path)
                dirname = os.path.normpath(dirname)
                signer_id, video_id = dirname.split(os.sep)
                signer_id = signer_id
                video_id = video_id
                if signer_id not in signers:
                    signer = Signer(signer_id)
                    signers[signer_id] = signer
                else:
                    signer = signers[signer_id]
                existing_video = next(
                    (v for v in signer.videos if v.id == video_id), None
                )
                if existing_video is None:
                    video = Video(video_id)
                    signer.videos.append(video)
                else:
                    video = existing_video
                frame_id = filename.removesuffix(".png")
                frame = Frame(
                    frame_id,
                    os.path.join(
                        os.path.normpath(self._path), dirname, filename
                    ),
                )
                video.frames.append(frame)
        return signers


class Signer:
    """Class to handle a signer."""

    def __init__(self, id: str) -> None:
        """Initialize the Signer object.

        Parameters
        ----------
        id : str
            The id of the signer.

        """
        self.id = id
        self.videos: list[Video] = []

    def get_frames_num(self):
        """Get the number of frames in the signer.

        Returns
        -------
        int
            The number of frames in the signer.

        """
        num_frames = 0
        for video in self.videos:
            num_frames += len(video.frames)
        return num_frames

    def split(
        self, val_split: float, test_split: float, random_state: int = 42
    ):
        """Split the dataset into train, validation and test sets.

        Parameters
        ----------
        val_split : float
            The proportion of the dataset to include in the validation split.
        test_split : float
            The proportion of the dataset to include in the test split.
        random_state : int, optional
            The random seed to use for the random number generator, by default 42.

        Returns
        -------
        tuple
            A tuple containing the train, validation and test sets.

        """
        np.random.seed(random_state)
        val_videos = []
        test_videos = []
        if len(self.videos) > 2:
            test_videos_to_extract = math.ceil(len(self.videos) * test_split)
            test_videos_ids = np.random.choice(
                np.arange(0, len(self.videos)),
                test_videos_to_extract,
                replace=False,
            ).tolist()
            for test_video_id in test_videos_ids:
                test_videos.append(self.videos[test_video_id])
            for i, test_video_id in enumerate(test_videos_ids):
                self.videos.remove(self.videos[test_video_id - i])
            val_videos_to_extract = math.ceil(len(self.videos) * val_split)
            val_videos_ids = np.random.choice(
                np.arange(0, len(self.videos)),
                val_videos_to_extract,
                replace=False,
            ).tolist()
            for val_video_id in val_videos_ids:
                val_videos.append(self.videos[val_video_id])
            for i, val_video_id in enumerate(val_videos_ids):
                self.videos.remove(self.videos[val_video_id - i])
            return self.videos, val_videos, test_videos
        else:
            tot_frames = []
            video_id = self.videos[0].id
            for video in self.videos:
                tot_frames.extend(video.frames)
            tot_frames = shuffle(tot_frames, random_state=random_state)
            tot_frames_len = len(tot_frames)
            test_frames_to_extract = math.ceil(tot_frames_len * test_split)
            val_frames_to_extract = math.ceil(tot_frames_len * val_split)
            test_frames = tot_frames[:test_frames_to_extract]
            val_frames = tot_frames[
                test_frames_to_extract : test_frames_to_extract
                + val_frames_to_extract
            ]
            train_frames = tot_frames[
                test_frames_to_extract + val_frames_to_extract :
            ]
            test_videos = [Video(video_id, frames=test_frames)]
            val_videos = [Video(video_id, frames=val_frames)]
            train_videos = [Video(video_id, frames=train_frames)]
            return train_videos, val_videos, test_videos


class Video:
    """Class to handle a video."""

    def __init__(self, id: str, frames=None) -> None:
        """Initialize the Video object.

        Parameters
        ----------
        id : str
            The id of the video.
        frames : list[Frame], optional
            The frames of the video, by default None.

        """
        self.id = id
        if frames is None:
            self.frames: list[Frame] = []
        else:
            self.frames = frames

    def __str__(self):
        """Return a string representation of the Video object.

        Returns
        -------
        str
            A string representation of the Video object.

        """
        return f"Video {self.id}"

    def __eq__(self, other) -> bool:
        """Check if two Video objects are equal.

        Parameters
        ----------
        other : Video
            The other Video object to compare with.

        Returns
        -------
        bool
            True if the Video objects are equal, False otherwise.

        """
        return self.id == other.id

    def extract_frames(self, frames_num: float, seed: int):
        """Extract a number of frames from the video.

        Parameters
        ----------
        frames_num : int
            The number of frames to extract.
        seed : int
            The random seed to use for the random number generator.

        Returns
        -------
        list[Frame]
            A list of Frame objects.

        """
        frames = []
        if frames_num > len(self.frames):
            frames_to_extract = len(self.frames)
        else:
            frames_to_extract = frames_num
        np.random.seed(seed)
        frame_ids = np.random.choice(
            np.arange(0, len(self.frames)), frames_to_extract, replace=False
        ).tolist()
        for frame_id in frame_ids:
            frames.append(self.frames[frame_id])
        return frames


class Frame:
    """Class to handle a frame."""

    def __init__(self, id: str, path: str) -> None:
        """Initialize the Frame object.

        Parameters
        ----------
        id : str
            The id of the frame.
        path : str
            The path to the frame.

        """
        self.id = id
        self.path = path

    def __str__(self):
        """Return a string representation of the Frame object.

        Returns
        -------
        str
            A string representation of the Frame object.

        """
        return f"Frame {self.id}, path: {self.path}"

    def load_frame(self) -> np.ndarray:
        """Load the frame from the path.

        Returns
        -------
        np.ndarray
            The loaded frame.

        """
        return cv2.imread(self.path)
