from typing import List

import numpy as np


class GlobalFilters(object):
    def __init__(self) -> None:
        self.pose_filter = [11, 12, 13, 14, 15, 16]
        self.face_filter = [
            0,
            4,
            8,
            10,
            13,
            14,
            17,
            33,
            37,
            46,
            52,
            54,
            55,
            61,
            127,
            133,
            145,
            150,
            152,
            159,
            263,
            267,
            276,
            282,
            284,
            285,
            291,
            356,
            362,
            374,
            379,
            386,
        ]

    @staticmethod
    def filter(features: np.ndarray, filter_list: List[int]) -> np.ndarray:
        return features[filter_list]
