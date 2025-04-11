"""Module to perform the DTW classification of the videos using MediaPipe features."""

from typing import Dict, List, Tuple

import numpy as np
from fastdtw import fastdtw  # type: ignore
from tabulate import tabulate
from tqdm import tqdm

from handcrafted.app.dataset.video import Video
from wlasl_mediapipe.app.mp.mp_video import MediapipeVideo


def calc_dtw_distance(
    video: MediapipeVideo, others: List[MediapipeVideo]
) -> Tuple[str, float]:
    """Calculates the DTW distance between the video and the others.

    Parameters
    ----------
    video : MediapipeVideo
        The video to compare.
    others : List[MediapipeVideo]
        The other videos.

    Returns
    -------
    Tuple[str, float]
        The gloss and the distance.

    """
    ret = {"IncompatibleHands": [np.inf]}
    left_hand = video.sign_model.lh_embedding
    right_hand = video.sign_model.rh_embedding
    curr = 0
    for other_video in others:
        curr += 1
        if (
            video.sign_model.has_left_hand
            == other_video.sign_model.has_left_hand
        ) and (
            video.sign_model.has_right_hand
            == other_video.sign_model.has_right_hand
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
        ret[other_video.video.gloss].append(
            distance["left"] + distance["right"]
        )
    return _best_choice(ret)


def _best_choice(distances) -> Tuple[str, float]:
    """Returns the best choice from the distances.

    Parameters
    ----------
    distances : Dict[str, List[float]]
        The distances.

    """
    for gloss in distances:
        distances[gloss] = np.min(distances[gloss])
    return min(distances.items(), key=lambda x: x[1])


def classify(
    test_videos: List[MediapipeVideo],
    train_videos: Dict[str, List[Video]],
    augment: int,
    topN: int = 1,
) -> Dict[MediapipeVideo, List[Tuple[str, float]]]:
    """Classifies the test videos using the train videos.

    Parameters
    ----------
    test_videos : List[MediapipeVideo]
        The test videos.
    train_videos : Dict[str, List[Video]]
        The train videos, in the format {gloss: [videos]}.
    augment : int
        The number of augmentations to apply.
    topN : int, optional
        The number of top predictions to consider, by default 1.

    Returns
    -------
    Dict[MediapipeVideo, List[Tuple[str, float]]]
        The classified glosses, in the format {video: [(gloss, distance)]}.

    """
    classified_glosses: Dict[MediapipeVideo, List[Tuple[str, float]]] = {}
    for gloss in tqdm(train_videos):
        current_train = [
            MediapipeVideo(
                train_video,
                plot=False,
                expand_keypoints=True,
                all_features=True,
            )
            for train_video in train_videos[gloss]
        ]
        if augment != 0:
            augmented_train = [
                video.augment(augment) for video in current_train
            ]
            for videos in augmented_train:
                current_train.extend(videos)
        for video in test_videos:
            classified_glosses = _do_classification(
                video, current_train, classified_glosses, topN
            )
    return classified_glosses


def _do_classification(
    video: MediapipeVideo,
    current_train,
    classified: Dict[MediapipeVideo, List[Tuple[str, float]]],
    topN: int,
) -> Dict[MediapipeVideo, List[Tuple[str, float]]]:
    """Internal function to classify a video.

    Parameters
    ----------
    video : MediapipeVideo
        The video to classify.
    current_train : List[MediapipeVideo]
        The current train videos.
    classified : Dict[MediapipeVideo, List[Tuple[str, float]]]
        The classified glosses until now.
    topN : int
        The number of top predictions to consider.

    Returns
    -------
    Dict[MediapipeVideo, List[Tuple[str, float]]]
        The classified glosses, in the format {video: [(gloss, distance)].

    """
    closest_word: Tuple[str, float] = calc_dtw_distance(video, current_train)
    classifications = classified.get(video)
    if classifications is None:
        classified[video] = [closest_word]
    else:
        for i, cl in enumerate(classifications):
            if closest_word[1] < cl[1]:
                classifications.insert(i, closest_word)
                break
            elif i == len(classifications) - 1:
                classifications.append(closest_word)
                break
    return classified


def _calc_acc(
    classified_glosses: Dict[MediapipeVideo, List[Tuple[str, float]]],
    topN: int,
) -> float:
    """Calculates the accuracy of the classification.

    Parameters
    ----------
    classified_glosses : Dict[MediapipeVideo, List[Tuple[str, float]]]
        The classified glosses, in the format {video: [(gloss, distance)]}.
    topN : int
        The number of top predictions to consider.

    Returns
    -------
    float
        The accuracy of the classification.

    """
    right = 0
    tot = len(classified_glosses)
    for el in classified_glosses.items():
        cl = [val[0] for val in el[1][:topN]]
        if cl.count(el[0].get_base_video().gloss) == 1:
            right += 1
    return right / tot


def pretty_print(
    classified_glosses: Dict[MediapipeVideo, List[Tuple[str, float]]],
    output_file: str,
    topN: int,
):
    """Prints the results of the classification in a pretty format.

    Parameters
    ----------
    classified_glosses : Dict[MediapipeVideo, List[Tuple[str, float]]]
        The classified glosses, in the format {video: [(gloss, distance)]}.
    output_file : str
        The output file where the results will be saved.
    topN : int
        The number of top predictions to consider.

    """
    output_file = f"top{topN}-{output_file}"
    rows = []
    for el in classified_glosses.items():
        rows.append([el[0].get_base_video().gloss, el[1][:topN]])

    print(f"Top{topN} score:")
    print(tabulate(rows, headers=["Real Word", "Classified Words"]))
    acc = _calc_acc(classified_glosses, topN)
    print(f"\nAccuracy: {acc}")
    with open(output_file, "w") as file:
        file.write(f"Top{topN} score:")
        file.write(
            tabulate(
                rows, headers=["Real Word", "Classified Word", "Distance"]
            )
        )
        file.write(f"\nAccuracy: {acc}")
