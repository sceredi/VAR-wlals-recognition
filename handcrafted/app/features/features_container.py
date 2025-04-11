"""Module to extract features from video frames."""

import os
from typing import List, Tuple

import cv2
import numpy as np

from handcrafted.app.features.extractor.color_histogram_extractor import (
    ColorHistogram,
)
from handcrafted.app.features.extractor.contour_extractor import (
    ContourExtractor,
)
from handcrafted.app.features.extractor.edge_extractor import EdgeExtractor
from handcrafted.app.features.extractor.features_storage import FeaturesStorage
from handcrafted.app.features.extractor.flow_calculator import FlowCalculator
from handcrafted.app.features.extractor.haar.haar_detector import HaarDetector
from handcrafted.app.features.extractor.hog_extractor import HOGExtractor
from handcrafted.app.features.extractor.lbp_extractor import LBPExtractor
from handcrafted.app.features.extractor.skin import SkinExtractor


class FeaturesContainer:
    """Class to extract features from video frames."""

    def __init__(self, video, save: bool = False) -> None:
        """Initialize the FeaturesContainer object.

        Parameters
        ----------
        video : Video
            The video to process.
        save : bool, optional
            Whether to save the features, by default False.

        """
        self.video = video

        self.save = save
        self.fs = FeaturesStorage()
        self.path = os.path.join("data", "hf", self.video.video_id)

        self._hog_features = None
        self._hog_frames = None
        self._flow_features = None
        self._flow_features_flattened = None
        self._contour_features = None
        self._contour_features_flattened = None
        self._edge_features = None
        self._edge_features_flattened = None
        self._haar_features = None
        self._skin_features = None
        self._skin_features_flattened = None
        self._lbp_features = None
        self._color_hist_features = None
        self._color_hist_features_flattened = None

    def get_all_features(
        self, until_frame_number: None | int = None
    ) -> "np.ndarray":
        """Get all the interesting features from the video.

        Parameters
        ----------
        until_frame_number : int, optional
            The last frame number to process, by default None.

        Returns
        -------
        np.ndarray
            The features.

        """
        frames = self.video.get_frames(last_frame=until_frame_number)

        self._hog_features = self.load_or_compute_feature(
            "hog_features", self.get_hog_features, frames
        )
        print(f"hog_features shape: {np.array(self._hog_features).shape}")
        self._lbp_features = self.load_or_compute_feature(
            "lbp_features", self.get_lbp_features, frames
        )
        print(f"lbp_features shape: {np.array(self._lbp_features).shape}")
        return np.concatenate((self._hog_features, self._lbp_features), axis=1)

    def load_or_compute_feature(
        self, feature_name: str, extraction_function, *args, **kwargs
    ):
        """Load the feature from the file system or compute it if it doesn't exist.

        Parameters
        ----------
        feature_name : str
            The name of the feature to load or compute.
        extraction_function : callable
            The function to compute the feature.
        *args : list
            The arguments to pass to the extraction function.
        **kwargs : dict
            The keyword arguments to pass to the extraction function.

        Returns
        -------
        np.ndarray
            The feature.

        """
        feature_path = os.path.join(self.path, feature_name)

        if self.save and os.path.exists(feature_path):
            print(f"Caricamento da file: {feature_path}")
            return self.fs.load_feature(feature_path)

        feature = extraction_function(*args, **kwargs)

        if self.save:
            os.makedirs(self.path, exist_ok=True)
            print(f"Salvataggio della feature: {feature_path}")
            self.fs.save_feature(feature, feature_path)

        return feature

    def get_hog_features(self, frames) -> "np.ndarray":
        """Get the HOG features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.

        Returns
        -------
        np.ndarray
            The HOG features.

        """
        if self._hog_features is None:
            self._hog_features, self._hog_frames = HOGExtractor(
                frames
            ).process_frames()
        return self._hog_features  # type: ignore

    def get_flow_features(
        self,
        frames: List["np.ndarray"],
        last_frame_index: int = -1,
        flatten: bool = True,
    ) -> List["np.ndarray"]:
        """Get the flow features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        last_frame_index : int, optional
            The last frame index to process, by default -1.
        flatten : bool, optional
            Whether to flatten the features, by default True.

        Returns
        -------
        List[np.ndarray]
            The flow features.

        """
        if last_frame_index == -1:
            last_frame_index = len(frames) - 1
        if self._flow_features is None:
            self._flow_features = FlowCalculator(
                frames, last_frame_index
            ).calculate(plot_each_frame=False)[0]
        if flatten:
            if self._flow_features_flattened is None:
                self._flow_features_flattened = [
                    feature.flatten() for feature in self._flow_features
                ]
            return self._flow_features_flattened
        return self._flow_features

    def get_contour_features(
        self, frames, flatten: bool = True
    ) -> "np.ndarray":
        """Get the contour features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        flatten : bool, optional
            Whether to flatten the features, by default True.

        Returns
        -------
        np.ndarray
            The contour features.

        """
        if self._contour_features is None:
            self._contour_features = np.array(
                ContourExtractor(frames).process_frames()
            )
        if flatten:
            if self._contour_features_flattened is None:
                self._contour_features_flattened = (
                    self._contour_features.copy().reshape(
                        self._contour_features.shape[0], -1
                    )
                )
            return self._contour_features_flattened
        return self._contour_features

    def get_edge_features(
        self, frames: List["np.ndarray"], flatten: bool = True
    ) -> "np.ndarray":
        """Get the edge features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        flatten : bool, optional
            Whether to flatten the features, by default True.

        Returns
        -------
        np.ndarray
            The edge features.

        """
        if self._edge_features is None:
            self._edge_features = np.array(
                EdgeExtractor(frames).process_frames()
            )
        if flatten:
            if self._edge_features_flattened is None:
                self._edge_features_flattened = (
                    self._edge_features.copy().reshape(
                        self._edge_features.shape[0], -1
                    )
                )
            return self._edge_features_flattened
        return self._edge_features

    def _get_haar_features(
        self, frames: List["np.ndarray"]
    ) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
        """Get the Haar features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            The Haar features.

        """
        classifier = cv2.CascadeClassifier()
        if not classifier.load(
            cv2.samples.findFile(
                "handcrafted/app/features/extractor/haar/haarcascades/face.xml"
            )
        ):
            print("Error loading face cascade")
            exit(1)
        if self._haar_features is None:
            self._haar_features = HaarDetector(
                frames, classifier, detections_to_keep=3
            ).detect()
        return self._haar_features

    def get_skin_features(
        self, frames: List["np.ndarray"], face_rects, flatten: bool = True
    ) -> List["np.ndarray"]:
        """Get the skin features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        face_rects : List[Tuple[int, int, int, int]]
            The face rectangles.
        flatten : bool, optional
            Whether to flatten the features, by default True.

        Returns
        -------
        List[np.ndarray]
            The skin features.

        """
        if self._skin_features is None:
            self._skin_features = SkinExtractor(frames, face_rects).extract()
        if flatten:
            if self._skin_features_flattened is None:
                self._skin_features_flattened = (
                    self._skin_features.copy().reshape(  # type: ignore
                        self._skin_features.shape[0],  # type: ignore
                        -1,  # type: ignore
                    )
                )
            return self._skin_features_flattened
        return self._skin_features

    def get_lbp_features(self, frames: List["np.ndarray"]) -> "np.ndarray":
        """Get the LBP features from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.

        Returns
        -------
        np.ndarray
            The LBP features.

        """
        if self._lbp_features is None:
            self._lbp_features = LBPExtractor(frames).process_frames()
        return self._lbp_features  # type: ignore

    def get_color_histogram(
        self, frames, to_color=cv2.COLOR_BGR2HSV, flatten: bool = True
    ):
        """Get the color histogram from the frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        to_color : int, optional
            The color conversion code, by default cv2.COLOR_BGR2HSV.
        flatten : bool, optional
            Whether to flatten the features, by default True.

        Returns
        -------
        np.ndarray
            The color histogram features.

        """
        if self._color_hist_features is None:
            self._color_hist_features = ColorHistogram(frames).process_frames(
                to_color
            )
        if flatten:
            if self._color_hist_features_flattened is None:
                self._color_hist_features_flattened = (
                    self._color_hist_features.copy().reshape(
                        self._color_hist_features.shape[0], -1
                    )
                )
        return self._color_hist_features
