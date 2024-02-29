from typing import List, Tuple
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo

import numpy as np
import pandas as pd

from fastdtw import fastdtw


def calc_dtw_distance(video: MediapipeVideo, others: List[MediapipeVideo]):
    ret = {}
    left_hand = video.sign_model.lh_embedding
    right_hand = video.sign_model.rh_embedding
    curr = 0
    for other_video in others:
        curr += 1
        # print(f"Calculating distance for {curr}/{len(others)}")
        # print(f"Video: {video.video.get_path()}")
        # print(f"Other Video: {other_video.video.get_path()}")
        if (
            video.sign_model.has_left_hand == other_video.sign_model.has_left_hand
        ) and (
            video.sign_model.has_right_hand == other_video.sign_model.has_right_hand
        ):
            distance = {
                "left": 0.0,
                "right": 0.0,
            }
            other_left_hand = other_video.sign_model.lh_embedding
            other_right_hand = other_video.sign_model.rh_embedding
            if video.sign_model.has_left_hand:
                distance["left"] = fastdtw(left_hand, other_left_hand)[0]
            if video.sign_model.has_right_hand:
                distance["right"] = fastdtw(right_hand, other_right_hand)[0]
        else:
            distance = {
                "left": 100000.0,
                "right": 100000.0,
            }

        if other_video.video.gloss not in ret:
            ret[other_video.video.gloss] = []
        ret[other_video.video.gloss].append(distance["left"] + distance["right"])
        # ret.append((distance["left"] + distance["right"], other_video))
    return _best_choice(ret)


def _best_choice(distances):
    """Given a list of distances, calculates the average the distances for each gloss"""
    for gloss in distances:
        distances[gloss] = np.mean(distances[gloss])
    print(f"Distances again: {distances}")
    return min(distances.items(), key=lambda x: x[1])
