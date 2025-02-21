import math
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from handcrafted.app.dataset.video import Video
from handcrafted.app.edge.detector import EdgeDetector
from handcrafted.app.extractor.contour_extractor import ContourDetector
from handcrafted.app.extractor.hog_extractor import HOGExtractor
from handcrafted.app.extractor.skin import SkinExtractor
from handcrafted.app.flow.calculator import FlowCalculator
from handcrafted.app.haar.detector import HaarDetector
from handcrafted.app.roi.extractor import RoiExtractor


class FramesPlotter:
    def __init__(self, frames: List["np.ndarray"], to_rgb=True):
        self.frames = frames
        self.to_rgb = to_rgb

    def plot_grid(self):
        num_frames = len(self.frames)
        cols = int(math.ceil(math.sqrt(num_frames)))
        rows = int(math.ceil(num_frames / cols))
        _, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for i, (frame, ax) in enumerate(zip(self.frames, axes)):
            self._update(ax, frame, f"Frame {i+1}")
        for ax in axes[num_frames:]:
            self._remove_axis(ax)
        plt.show()

    def _remove_axis(self, ax):
        ax.axis("off")

    def _to_rgb(self, frame) -> "np.ndarray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _update(self, ax, frame, title):
        if self.to_rgb:
            frame = self._to_rgb(frame)
        ax.clear()
        ax.imshow(frame)
        ax.set_title(title)
        self._remove_axis(ax)


def plot_frames(frames: List["np.ndarray"], is_gray_scale=False) -> None:
    plotter = FramesPlotter(frames, to_rgb=not is_gray_scale)
    plotter.plot_grid()


def plot_roi_frames(video: Video, remove_background=False) -> None:
    roi_extractor = RoiExtractor(video.get_frames(), video.bbox, resize=224)
    roi_frames = roi_extractor.extract(remove_background=remove_background)
    plot_frames(roi_frames)


def plot_edge_frames(frames: List["np.ndarray"]) -> None:
    edge_detector = EdgeDetector(frames)
    edge_frames = edge_detector.detect()
    plot_frames(edge_frames)


def plot_flow_frames(frames: List["np.ndarray"], last_frame_index: int = -1) -> None:
    if last_frame_index == -1:
        last_frame_index = len(frames) - 1
    flow_calculator = FlowCalculator(frames, last_frame_index)
    flow_frames = flow_calculator.calculate(plot_each_frame=False)
    plot_frames(flow_frames)


def plot_hog_frames(frames: List["np.ndarray"]) -> None:
    hog_extractor = HOGExtractor(frames)
    hog_features, hog_frames = hog_extractor.process_frames()
    plot_frames(hog_frames, is_gray_scale=True)


def plot_contour(frames: List[np.ndarray]) -> None:
    contour_detector = ContourDetector(frames)
    contour_frames = contour_detector.detect()
    plot_frames(contour_frames)


def plot_haar_frames(frames: List["np.ndarray"], plot: bool = True):
    classifier = cv2.CascadeClassifier()
    if not classifier.load(
        cv2.samples.findFile("handcrafted/app/haar/haarcascades/face.xml")
    ):
        print("Error loading face cascade")
        exit(1)
    haar_detector = HaarDetector(frames, classifier, detections_to_keep=3)
    face_frames, rects = haar_detector.detect()
    if plot:
        plot_frames(face_frames)
    return rects


def plot_skin_frames(frames: List["np.ndarray"]) -> None:
    face_rects = plot_haar_frames(frames, False)
    skin_extractor = SkinExtractor(frames, face_rects)
    skin_frames = skin_extractor.extract()
    plot_frames(skin_frames)
