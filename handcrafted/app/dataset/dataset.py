"""Module for loading and managing datasets."""

import json
from collections.abc import Callable
from typing import List, Tuple

from handcrafted.app.dataset.video import Video


class Dataset:
    """Class to handle a dataset of videos."""

    def __init__(
        self, filename: str, only_keypoints=False, only_samples=False
    ) -> None:
        """Initialize the Dataset object.

        Parameters
        ----------
        filename : str
            The path to the dataset file.
        only_keypoints : bool, optional
            Whether to load only videos with keypoints, by default False.
        only_samples : bool, optional
            Whether to load only samples, by default False.

        """
        self.filename = filename
        self._only_samples = only_samples
        if not only_keypoints:
            self.videos, self.glosses = self.load_videos()
        else:
            self.videos, self.glosses = self.load_videos_with_keypoints()

    def load_videos(self) -> Tuple[List[Video], List[str]]:
        """Load videos from the dataset file.

        Returns
        -------
        Tuple[List[Video], List[str]]
            A tuple containing a list of Video objects and a list of glosses.

        """
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        glosses = []
        for word_data in data:
            gloss = word_data["gloss"]
            glosses.append(gloss)
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(
                    gloss, instance, self._only_samples
                )
                if not video.is_missing():
                    ret.append(video)
        return ret, glosses

    def load_videos_with_keypoints(self) -> Tuple[List[Video], List[str]]:
        """Load videos with keypoints from the dataset file.

        Returns
        -------
        Tuple[List[Video], List[str]]
            A tuple containing a list of Video objects and a list of glosses.

        """
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        glosses = []
        for word_data in data:
            gloss = word_data["gloss"]
            glosses.append(gloss)
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance)
                if video.has_keypoints():
                    ret.append(video)
        return ret, glosses

    def update_videos(self) -> None:
        """Update the videos in the dataset file with their end frames."""
        with open(self.filename) as f:
            data = json.load(f)
        for word_data in data:
            gloss = word_data["gloss"]
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(
                    gloss, instance, self._only_samples
                )
                if not video.is_missing():
                    instance["frame_end"] = video.get_end()
                else:
                    instance = None
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

    def get_videos(self, condition: Callable[[Video], bool]) -> List[Video]:
        """Get videos from the dataset based on a condition.

        Parameters
        ----------
        condition : Callable[[Video], bool]
            A function that takes a Video object and returns a boolean.

        Returns
        -------
        List[Video]
            A list of Video objects that satisfy the condition.

        """
        return [video for video in self.videos if condition(video)]

    def __str__(self) -> str:
        """Return a string representation of the Dataset object.

        Returns
        -------
        str
            A string representation of the Dataset object.

        """
        return f"Dataset(filename={self.filename}, videos={self.videos})\n"
