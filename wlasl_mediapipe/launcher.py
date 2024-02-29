from typing import List

import pandas as pd
from handcrafted.app.dataset.dataset import Dataset
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo
from wlasl_mediapipe.app.dtw.dtw import calc_dtw_distance


class Launcher:
    def start(self) -> None:
        print(len(self._load_data().videos))
        print(len(self._load_glosses()))
        self.analyze(self._load_data(), self._load_glosses()[1:3])

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self) -> List[str]:
        return pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
            1
        ].tolist()

    def analyze(self, dataset: Dataset, glosses: List[str]) -> None:
        training_videos = dataset.get_videos(
            lambda video: video.split == "train" and video.gloss in glosses
        )
        test_videos = dataset.get_videos(
            lambda video: (video.split == "test" or video.split == "val")
            and video.gloss in glosses
        )
        print(f"Training videos: {len(training_videos)}")
        print(f"Test videos: {len(test_videos)}")
        training_videos = [MediapipeVideo(video, plot=False) for video in training_videos]
        test_videos = [MediapipeVideo(video, plot=False) for video in test_videos]
        for video in test_videos:
            best_choice = calc_dtw_distance(video, training_videos)
            print(f"Best choice for {video.video.gloss}: {best_choice}")
                

