from typing import List

import pandas as pd

from handcrafted.app.dataset.dataset import Dataset
from wlasl_mediapipe.app.dtw.dtw import classify
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo


class Launcher:
    def start(self) -> None:
        print(len(self._load_data().videos))
        print(len(self._load_glosses()))
        self._analyze_with_dtw(self._load_data(), self._load_glosses()[:500])

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self) -> List[str]:
        return pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
            1
        ].tolist()

    def _analyze_with_dtw(self, dataset: Dataset, glosses: List[str]) -> None:
        test_videos = dataset.get_videos(
            lambda video: (video.split == "test") and video.gloss in glosses
        )
        test_videos = [MediapipeVideo(video, plot=False) for video in test_videos]
        splitted_train_videos = {}
        for gloss in glosses:
            splitted_train_videos[gloss] = dataset.get_videos(
                lambda video: video.gloss == gloss
                and (video.split == "train" or video.split == "val")
            )
        classify(test_videos, splitted_train_videos)
