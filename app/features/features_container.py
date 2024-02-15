from typing import List, Tuple
import cv2
from matplotlib.cbook import flatten

import numpy as np
from app.color_histogram.extractor import ColorHistogram

from app.extractor.hog_extractor import HOGExtractor
from app.flow.calculator import FlowCalculator
from app.extractor.contour_extractor import ContourDetector
from app.edge.detector import EdgeDetector
from app.extractor.skin import SkinExtractor
from app.haar.detector import HaarDetector
from app.lbp.extractor import LBPExtractor

class FeaturesContainer:
    def __init__(self, video) -> None:
        self.video = video

    def get_all_features(self, until_frame_number: None | int = None) -> "np.ndarray":
        frames = self.video.get_frames(last_frame=until_frame_number)
        # hog_features, _ = self.get_hog_features(frames)
        # print(f"hog_features shape: {np.array(hog_features).shape}")
        # flow_frames = self.get_flow_features(frames)
        # print(f"flow_frames shape: {np.array(flow_frames).shape}")
        # contour_features = self.get_contour_features(frames)
        # print(f"contour_features shape: {np.array(contour_features).shape}")
        # edge_features = self.get_edge_features(frames)
        # print(f"edge_features shape: {np.array(edge_features).shape}")
        _, face_rects = self._get_haar_features(frames)
        skin_features = self.get_skin_features(frames, face_rects, flatten = False)
        print(f"skin_features shape: {np.array(skin_features).shape}")
        lbp_frames = self.get_lbp_features(skin_features)
        print(f"lbp_frames shape: {np.array(lbp_frames).shape}")
        # color_histogram = self.get_color_histogram(frames, flatten=True)
        # print(f"color_histogram shape: {np.array(color_histogram).shape}")
        return np.concatenate(
            [lbp_frames],
            axis=1,
        )

    def get_hog_features(self, frames) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
        return HOGExtractor(frames).process_frames()

    def get_flow_features(
        self, frames: List["np.ndarray"], last_frame_index: int = -1, flatten: bool = True
    ) -> List["np.ndarray"]:
        if last_frame_index == -1:
            last_frame_index = len(frames) - 1
        features = FlowCalculator(frames, last_frame_index).calculate(plot_each_frame=False)
        if flatten:
            features = [feature.flatten() for feature in features]
        return features

    def get_contour_features(self, frames, flatten: bool = True) -> "np.ndarray":
        features = np.array(ContourDetector(frames).detect())
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features

    def get_edge_features(self, frames: List["np.ndarray"], flatten: bool = True) -> "np.ndarray":
        features = np.array(EdgeDetector(frames).detect())
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features

    def _get_haar_features(
        self, frames: List["np.ndarray"]
    ) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
        classifier = cv2.CascadeClassifier()
        if not classifier.load(cv2.samples.findFile("app/haar/haarcascades/face.xml")):
            print("Error loading face cascade")
            exit(1)
        return HaarDetector(frames, classifier, detections_to_keep=3).detect()

    def get_skin_features(
        self, frames: List["np.ndarray"], face_rects, flatten: bool = True
    ) -> "np.ndarray":
        features = np.array(SkinExtractor(frames, face_rects).extract())
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features

    def get_lbp_features(self, frames: List["np.ndarray"]) -> "np.ndarray":
        return LBPExtractor(frames).extract()

    def get_color_histogram(self, frames, to_color = cv2.COLOR_BGR2HSV, flatten: bool = True):
        features = ColorHistogram(frames).extract(to_color)
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features
