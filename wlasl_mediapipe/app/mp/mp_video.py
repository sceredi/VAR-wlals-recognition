"""Module to handle a video with MediaPipe features."""

import os
from typing import List

from handcrafted.app.dataset.video import Video
from handcrafted.app.features.plotter.frames_plotter import FramesPlotter
from wlasl_mediapipe.app.mp.augmentation import augment_video
from wlasl_mediapipe.app.mp.keypoints_extractor import (
    MediapipeLandmarksExtractor,
)
from wlasl_mediapipe.app.mp.models.sign_model import SignModel
from wlasl_mediapipe.app.utils.mp.file_utils import save_array
from wlasl_mediapipe.app.utils.mp.helper.landmark_analyzer import (
    extract_landmarks,
)


class MediapipeVideo:
    """Class to handle a video with MediaPipe features."""

    def __init__(
        self,
        video: Video,
        plot: bool = True,
        expand_keypoints: bool = False,
        all_features: bool = True,
        sign_model: SignModel | None = None,
    ):
        """Initialize the MediapipeVideo object.

        Parameters
        ----------
        video : Video
            The video.
        plot : bool, optional
            If True, plots the annotated frames, by default True.
        expand_keypoints : bool, optional
            If True, expands the hands keypoints, by calculating the angle,
            in radians, between the connected keypoints, by default False.
        all_features : bool, optional
            If True, extracts all the features, by default True.
        sign_model : SignModel, optional
            The sign model, by default None.

        """
        self.video = video
        if sign_model is None:
            if not os.path.exists(f"data/mp/{self.video.video_id}"):
                self.model = MediapipeLandmarksExtractor()
            self._load_models(plot, expand_keypoints, all_features)
        else:
            self.sign_model = sign_model

    def get_base_video(self) -> Video:
        """Return the base video.

        Returns
        -------
        Video
            The base video.

        """
        return self.video

    def _load_models(
        self,
        plot: bool = True,
        expand_keypoints: bool = False,
        all_features: bool = True,
    ) -> None:
        """Load the models.

        Parameters
        ----------
        plot : bool, optional
            If True, plots the annotated frames, by default True.
        expand_keypoints : bool, optional
            If True, expands the hands keypoints, by calculating the angle,
            in radians, between the connected keypoints, by default False.
        all_features : bool, optional
            If True, extracts all the features, by default True.

        """
        if os.path.exists(f"data/mp/{self.video.video_id}"):
            self.sign_model = SignModel.load(
                self.video.video_id, expand_keypoints, all_features
            )
        else:
            print(f"Path does not exist: data/mp/{self.video.video_id}")
            results, annotated_frames = self.model.process_video(self.video)
            if plot:
                FramesPlotter(annotated_frames, to_rgb=False).plot_grid()
            pose_list = []
            face_list = []
            left_hand_list = []
            right_hand_list = []
            for result in results:
                pose, face, left_hand, right_hand = extract_landmarks(result)
                pose_list.append(pose)
                face_list.append(face)
                left_hand_list.append(left_hand)
                right_hand_list.append(right_hand)
            if not os.path.exists(f"data/mp/{self.video.video_id}"):
                os.mkdir(f"data/mp/{self.video.video_id}")
            save_array(
                pose_list,
                f"data/mp/{self.video.video_id}/pose_{self.video.video_id}.pickle",
            )
            save_array(
                face_list,
                f"data/mp/{self.video.video_id}/face_{self.video.video_id}.pickle",
            )
            save_array(
                left_hand_list,
                f"data/mp/{self.video.video_id}/lh_{self.video.video_id}.pickle",
            )
            save_array(
                right_hand_list,
                f"data/mp/{self.video.video_id}/rh_{self.video.video_id}.pickle",
            )
            self.sign_model = SignModel(
                left_hand_list,
                right_hand_list,
                pose_list,
                face_list,
                expand_keypoints,
            )

    def from_sign_model(self, sign_model: SignModel) -> "MediapipeVideo":
        """Return a new MediapipeVideo object from the given sign model.

        Parameters
        ----------
        sign_model : SignModel
            The sign model.

        Returns
        -------
        MediapipeVideo
            The new MediapipeVideo object.

        """
        return MediapipeVideo(
            self.video,
            plot=False,
            expand_keypoints=True,
            all_features=False,
            sign_model=sign_model,
        )

    def augment(self, n: int = 1) -> List["MediapipeVideo"]:
        """Augment the video.

        Parameters
        ----------
        n : int, optional
            The number of augmentations, by default 1.

        Returns
        -------
        List[MediapipeVideo]
            The augmented videos.

        """
        others = augment_video(self, n)
        return others
