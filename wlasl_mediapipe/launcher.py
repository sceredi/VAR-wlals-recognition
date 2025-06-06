"""Module containing the Launcher class, which is used to launch the classification process using MediaPipe features."""

import sys
from typing import Dict, List

import pandas as pd

from handcrafted.app.dataset.dataset import Dataset
from handcrafted.app.dataset.video import Video
from wlasl_mediapipe.app.dtw.dtw import classify, pretty_print
from wlasl_mediapipe.app.mp.models.globals import FilteredLabels
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo


class Launcher:
    """Helper class to launch the classification process using MediaPipe features."""

    def start(self) -> None:
        """Launch the classification process using MediaPipe features.

        The parameters are passed as command line arguments as follows:
        1. Number of words to classify
        2. TopN: Number of top predictions to consider
        3. Augment: Number of augmentations to apply to each video.

        """
        # Check if there are at least two arguments
        nwords = 10
        topN = 5
        augment = 3
        if len(sys.argv) >= 3:
            nwords = int(sys.argv[1])
            topN = int(sys.argv[2])
        if len(sys.argv) >= 4:
            augment = int(sys.argv[3])
        print(f"Number of words: {nwords}")
        print(f"TopN: {topN}")
        print(f"Will augment each video by: {augment}")
        data = self.load_data()
        print(len(data.videos))
        glosses = self.load_glosses(filtered=False)[:nwords]
        print("\n\nClassification without augmentation:")
        self._analyze_with_dtw(
            data, glosses, augment=0, output_file="results.log", topN=topN
        )
        print("\n\nClassification with augmentation:")
        self._analyze_with_dtw(
            data,
            glosses,
            augment=augment,
            output_file="results_aug.log",
            topN=topN,
        )

    def load_data(self, filename: str = "data/WLASL_v0.3.json") -> Dataset:
        """Load the WLASL dataset with only the keypoints.

        Parameters
        ----------
        filename : str
            The path to the WLASL dataset file.

        Returns
        -------
        Dataset
            The dataset with only the keypoints.

        """
        return Dataset(filename, only_keypoints=True)

    def load_glosses(
        self,
        filename: str = "data/wlasl_class_list.txt",
        filtered: bool = False,
    ) -> List[str]:
        """Load the wlasl class list.

        Parameters
        ----------
        filename : str
            The path to the file containing the glosses.
        filtered : bool
            If True, returns the filtered labels, filtered by cosine similarity, otherwise by the order they appear in the file.

        Returns
        -------
        List[str]
            List of glosses.

        """
        glosses = []
        if not filtered:
            glosses = pd.read_csv(filename, sep="\t", header=None)[1].tolist()
        else:
            glosses = FilteredLabels.get_labels()
        return glosses

    def get_test_videos(
        self, dataset: Dataset, glosses: List[str]
    ) -> List[MediapipeVideo]:
        """Get the test videos for the given glosses.

        Parameters
        ----------
        dataset : Dataset
            The dataset.
        glosses : List[str]
            The glosses to consider.

        Returns
        -------
        List[MediapipeVideo]
            The test videos.

        """
        test_videos = dataset.get_videos(
            lambda video: (video.split == "test") and video.gloss in glosses
        )
        mp_test_videos = [
            MediapipeVideo(
                video, plot=False, expand_keypoints=True, all_features=False
            )
            for video in test_videos
        ]
        return mp_test_videos

    def get_train_videos(
        self, dataset: Dataset, glosses: List[str]
    ) -> Dict[str, List[Video]]:
        """Get the train videos for the given glosses.

        Parameters
        ----------
        dataset : Dataset
            The dataset.
        glosses : List[str]
            The glosses to consider.

        Returns
        -------
        Dict[str, List[Video]]
            A dictionary with the glosses as keys and the list of train videos as values.

        """
        splitted_train_videos = {}
        for gloss in glosses:
            splitted_train_videos[gloss] = dataset.get_videos(
                lambda video: video.gloss == gloss
                and (video.split == "train" or video.split == "val")
            )
        return splitted_train_videos

    def _analyze_with_dtw(
        self,
        dataset: Dataset,
        glosses: List[str],
        augment: int = 0,
        output_file: str = "results.log",
        topN: int = 1,
    ) -> None:
        """Analyzes the dataset using DTW, and prints the results.

        Parameters
        ----------
        dataset : Dataset
            The dataset.
        glosses : List[str]
            The glosses to consider.
        augment : int, optional
            The number of augmentations to apply to each video, by default 0.
        output_file : str, optional
            The output file where the results will be saved, by default "results.log".
        topN : int, optional
            The number of top predictions to consider, by default 1.

        """
        test_videos = self.get_test_videos(dataset, glosses)
        splitted_train_videos = self.get_train_videos(dataset, glosses)
        classified_glosses = classify(
            test_videos,
            splitted_train_videos,
            augment=augment,
            topN=topN,
        )
        pretty_print(classified_glosses, output_file, 1)
        pretty_print(classified_glosses, output_file, topN)
