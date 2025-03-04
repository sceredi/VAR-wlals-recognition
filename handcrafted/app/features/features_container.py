import os
from typing import List, Tuple

import cv2
import numpy as np

from handcrafted.app.extractor.color_histogram_extractor import ColorHistogram
from handcrafted.app.extractor.contour_extractor import ContourExtractor
from handcrafted.app.extractor.edge_extractor import EdgeExtractor
from handcrafted.app.extractor.features_storage import FeaturesStorage
from handcrafted.app.extractor.hog_extractor import HOGExtractor
from handcrafted.app.extractor.lbp_extractor import LBPExtractor
from handcrafted.app.extractor.skin import SkinExtractor
from handcrafted.app.flow.calculator import FlowCalculator
from handcrafted.app.haar.detector import HaarDetector


class FeaturesContainer:
    def __init__(self, video, save: bool = False) -> None:
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

    # TODO
    def get_all_features(
        self, until_frame_number: None | int = None
    ) -> "np.ndarray":
        frames = self.video.get_frames(last_frame=until_frame_number)

        self._hog_features = self._load_or_compute_feature(
            "hog_features", self.get_hog_features, frames
        )
        print(f"hog_features shape: {np.array(self._hog_features).shape}")
        self._lbp_features = self._load_or_compute_feature(
            "lbp_features", self.get_lbp_features, frames
        )
        print(f"lbp_features shape: {np.array(self._lbp_features).shape}")
        # self.flow_features = self._load_or_compute_feature("flow_features", self.get_flow_features, frames)
        # self.contour_features = self._load_or_compute_feature("contour_features", self.get_contour_features, frames)
        # self.edge_features = self._load_or_compute_feature("edge_features", self.get_edge_features, frames)
        # self.skin_features = self._load_or_compute_feature("skin_features", self.get_skin_features, frames)
        # self.color_histogram = self._load_or_compute_feature("color_histogram", self.get_color_histogram, frames, flatten=True)

        # hog_features, _ = self.get_hog_features(frames)
        # print(f"hog_features shape: {np.array(hog_features).shape}")
        # # flow_frames = self.get_flow_features(frames)
        # # print(f"flow_frames shape: {np.array(flow_frames).shape}")
        # # contour_features = self.get_contour_features(frames)
        # # print(f"contour_features shape: {np.array(contour_features).shape}")
        # # edge_features = self.get_edge_features(frames)
        # # print(f"edge_features shape: {np.array(edge_features).shape}")
        # # _, face_rects = self._get_haar_features(frames)
        # # skin_features = self.get_skin_features(frames, face_rects, flatten = False)
        # # print(f"skin_features shape: {np.array(skin_features).shape}")
        # # lbp_features = self.get_lbp_features(skin_features)
        # lbp_features = self.get_lbp_features(frames) # TODO frames or skin_frames ?
        # print(f"lbp_features shape: {np.array(lbp_features).shape}")
        # # color_histogram = self.get_color_histogram(frames, flatten = True)
        # # print(f"color_histogram shape: {np.array(color_histogram).shape}")
        # return np.concatenate(
        #     #(hog_features, flow_frames, contour_features, edge_features, skin_features, lbp_features, color_histogram),
        #     (hog_features, lbp_features),
        #     axis=1,
        # )
        return np.concatenate((self._hog_features, self._lbp_features), axis=1)

    def _load_or_compute_feature(
        self, feature_name: str, extraction_function, *args, **kwargs
    ):
        """Carica una feature se già esiste, altrimenti la calcola, la salva e crea la cartella video_id se necessario."""
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

    def get_hog_features(
        self, frames
    ) -> "np.ndarray":  # Tuple[List["np.ndarray"], List["np.ndarray"]]:
        if self._hog_features is None:
            self._hog_features, self._hog_frames = HOGExtractor(
                frames
            ).process_frames()
        return self._hog_features

    def get_flow_features(
        self,
        frames: List["np.ndarray"],
        last_frame_index: int = -1,
        flatten: bool = True,
    ) -> List["np.ndarray"]:
        if last_frame_index == -1:
            last_frame_index = len(frames) - 1
        if self._flow_features is None:
            self._flow_features = FlowCalculator(
                frames, last_frame_index
            ).calculate(plot_each_frame=False)
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
        classifier = cv2.CascadeClassifier()
        if not classifier.load(
            cv2.samples.findFile("app/haar/haarcascades/face.xml")
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
        if self._skin_features is None:
            self._skin_features = SkinExtractor(frames, face_rects).extract()
        if flatten:
            if self._skin_features_flattened is None:
                self._skin_features_flattened = (
                    self._skin_features.copy().reshape(
                        self._skin_features.shape[0], -1
                    )
                )
            return self._skin_features_flattened
        return self._skin_features

    def get_lbp_features(self, frames: List["np.ndarray"]) -> "np.ndarray":
        if self._lbp_features is None:
            self._lbp_features = LBPExtractor(frames).process_frames()
        return self._lbp_features

    def get_color_histogram(
        self, frames, to_color=cv2.COLOR_BGR2HSV, flatten: bool = True
    ):
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
