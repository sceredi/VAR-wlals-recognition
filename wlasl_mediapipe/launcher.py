from typing import List

import pandas as pd
from handcrafted.app.dataset.dataset import Dataset
from handcrafted.app.plotter.framesPlotter import FramesPlotter
from wlasl_mediapipe.app.mp.hands_extractor import MediapipeHandsExtractor

class Launcher:
    def start(self) -> None:
        print(len(self._load_data().videos))
        print(len(self._load_glosses()))
        landmarks, frames = MediapipeHandsExtractor().process_video(self._load_data().videos[0])
        FramesPlotter(frames, to_rgb=False).plot_grid()

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self) -> List[str]:
        return pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
            1
        ].tolist()
