from typing import List

import pandas as pd
from handcrafted.app.dataset.dataset import Dataset
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo

class Launcher:
    def start(self) -> None:
        print(len(self._load_data().videos))
        print(len(self._load_glosses()))
        self.analyze(self._load_data().videos)

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self) -> List[str]:
        return pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
            1
        ].tolist()

    def analyze(self, videos) -> None:
        for video in videos:
            MediapipeVideo(video).get_poses(plot=False)
