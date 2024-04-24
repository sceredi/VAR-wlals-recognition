from typing import List

import pandas as pd

from handcrafted.app.dataset.dataset import Dataset
from wlasl_mediapipe.app.dtw.dtw import classify
from wlasl_mediapipe.app.mp.models.globals import FilteredLabels
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo


class Launcher:
    def start(self) -> None:
        data = self._load_data()
        glosses = self._load_glosses(filtered=True)[:100]
        print("\n\nClassification without augmentation:")
        self._analyze_with_dtw(data, glosses, augment=False, output_file="results.log")
        print("\n\nClassification with augmentation:")
        self._analyze_with_dtw(
            data, glosses, augment=True, output_file="results_aug.log"
        )

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self, filtered: bool = False) -> List[str]:
        glosses = []
        if not filtered:
            glosses = pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
                1
            ].tolist()
        else:
            glosses = FilteredLabels.get_labels()
        return glosses

    def _analyze_with_dtw(
        self,
        dataset: Dataset,
        glosses: List[str],
        augment: bool = False,
        output_file: str = "results.log",
    ) -> None:
        test_videos = dataset.get_videos(
            lambda video: (video.split == "test") and video.gloss in glosses
        )
        test_videos = [
            MediapipeVideo(video, plot=False, expand_keypoints=True, all_features=False)
            for video in test_videos
        ]
        splitted_train_videos = {}
        for gloss in glosses:
            splitted_train_videos[gloss] = dataset.get_videos(
                lambda video: video.gloss == gloss
                and (video.split == "train" or video.split == "val")
            )
        classify(
            test_videos, splitted_train_videos, augment=augment, output_file=output_file
        )
