from typing import List, Tuple
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo

import numpy as np

from fastdtw import fastdtw

def calc_dtw_distance(
    video: MediapipeVideo, others: List[MediapipeVideo]
) -> List[Tuple[dict[float, float], MediapipeVideo]]:
    ret = []
    left_hand = video.sign_model.lh_embedding
    right_hand = video.sign_model.rh_embedding
    curr = 0
    for other_video in others:
        curr += 1
        print(f"Calculating distance for {curr}/{len(others)}")
        print(f"Video: {video.video.get_path()}")
        print(f"Other Video: {other_video.video.get_path()}")
        if (
            video.sign_model.has_left_hand == other_video.sign_model.has_left_hand
        ) and (
            video.sign_model.has_right_hand == other_video.sign_model.has_right_hand
        ):

            distance = {
                "left": np.inf,
                "right": np.inf,
            }
            other_left_hand = other_video.sign_model.lh_embedding
            other_right_hand = other_video.sign_model.rh_embedding
            if video.sign_model.has_left_hand:
                distance["left"] = fastdtw(left_hand, other_left_hand)[0]
            if video.sign_model.has_right_hand:
                distance["right"] = fastdtw(right_hand, other_right_hand)[0]
        else:
            distance = {
                "left": np.inf,
                "right": np.inf,
            }

        ret.append((distance, other_video))
    ret.sort(key=lambda x: x[0]["left"] + x[0]["right"])
    return ret
