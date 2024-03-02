from typing import List
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo
from handcrafted.app.dataset.video import Video

import numpy as np
import gc

from fastdtw import fastdtw


def calc_dtw_distance(video: MediapipeVideo, others: List[MediapipeVideo]):
    ret = {}
    left_hand = video.sign_model.lh_embedding
    right_hand = video.sign_model.rh_embedding
    curr = 0
    for other_video in others:
        curr += 1
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
                "left": np.inf,
                "right": np.inf,
            }

        if other_video.video.gloss not in ret:
            ret[other_video.video.gloss] = []
        ret[other_video.video.gloss].append(distance["left"] + distance["right"])
    return _best_choice(ret)


def _best_choice(distances):
    """Given a list of distances, calculates the average the distances for each gloss"""
    for gloss in distances:
        distances[gloss] = np.min(distances[gloss])
    # print(f"Distances again: {distances}")
    return min(distances.items(), key=lambda x: x[1])


def calc_accuracy(real_glosses, classified_glosses) -> float:
    correct = 0
    total = len(real_glosses)

    for key, value in real_glosses.items():
        if key in classified_glosses:
            if classified_glosses[key][0] == value:
                correct += 1

    accuracy = correct / total * 100
    return accuracy


def classify(
    test_videos: List[List[Video]], train_videos: dict
) -> None:
    real_glosses = {}
    classified_glosses = {}
    acc = 0
    for test_batch in test_videos:
        gc.collect()
        print(f"Now testing batch #{acc}/{len(test_videos)}")
        test_batch = [MediapipeVideo(video, plot=False) for video in test_batch]
        for gloss in train_videos:
            # print(f"Getting training set for gloss {gloss}")
            current_train = [
                MediapipeVideo(train_video, plot=False)
                for train_video in train_videos[gloss]
            ]
            for video in test_batch:
                if real_glosses.get(video.video.video_id) is None:
                    real_glosses[video.video.video_id] = video.video.gloss
                best_choice = calc_dtw_distance(video, current_train)
                if classified_glosses.get(video.video.video_id) is None:
                    classified_glosses[video.video.video_id] = best_choice
                if best_choice[1] < classified_glosses[video.video.video_id][1]:
                    classified_glosses[video.video.video_id] = best_choice
    print(f"Real glosses: {real_glosses}")
    print(f"Classified glosses: {classified_glosses}")
    accuracy = calc_accuracy(real_glosses, classified_glosses)
    print(f"Accuracy: {accuracy}")
    with open("results.log", "w") as file:
        file.write(f"Real glosses: {real_glosses}\n")
        file.write(f"Classified glosses: {classified_glosses}\n")
        file.write(
            f"Accuracy: {np.mean([real == classified[0] for real, classified in zip(real_glosses, classified_glosses)])}"
        )
