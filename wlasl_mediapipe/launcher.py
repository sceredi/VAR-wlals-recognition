from typing import List

import pandas as pd
from handcrafted.app.dataset.dataset import Dataset


class Launcher:
    def start(self) -> None:
        print(len(self._load_data().videos))
        print(len(self._load_glosses()))

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")

    def _load_glosses(self) -> List[str]:
        return pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)[
            1
        ].tolist()
