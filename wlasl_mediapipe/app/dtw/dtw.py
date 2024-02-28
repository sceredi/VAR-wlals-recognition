from typing import List, Tuple
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo

import numpy as np

from fastdtw import fastdtw

def calc_dtw_distance(
    video: MediapipeVideo, others: List[MediapipeVideo]
) -> List[Tuple[dict[float, float], MediapipeVideo]]:
    ret = []
    distance = {
        "left": np.inf,
        "right": np.inf,
    }
    left_hand = video.sign_model.lh_embedding
    right_hand = video.sign_model.rh_embedding
    for other_video in others:
        if (
            video.sign_model.has_left_hand == other_video.sign_model.has_left_hand
        ) and (
            video.sign_model.has_right_hand == other_video.sign_model.has_right_hand
        ):
            other_left_hand = other_video.sign_model.lh_embedding
            other_right_hand = other_video.sign_model.rh_embedding
            if video.sign_model.has_left_hand:
                distance["left"] = fastdtw(left_hand, other_left_hand)[0]
            if video.sign_model.has_right_hand:
                distance["right"] = fastdtw(right_hand, other_right_hand)[0]

        ret.append((distance, other_video))
    return ret
