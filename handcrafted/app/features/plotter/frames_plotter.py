import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from handcrafted.app.dataset.video import Video
from handcrafted.app.features.extractor.color_histogram_extractor import (
    ColorHistogram,
)
from handcrafted.app.features.extractor.contour_extractor import (
    ContourExtractor,
)
from handcrafted.app.features.extractor.edge_extractor import EdgeExtractor
from handcrafted.app.features.extractor.flow_calculator import FlowCalculator
from handcrafted.app.features.extractor.haar.haar_detector import HaarDetector
from handcrafted.app.features.extractor.hog_extractor import HOGExtractor
from handcrafted.app.features.extractor.lbp_extractor import LBPExtractor
from handcrafted.app.features.extractor.skin import SkinExtractor
from handcrafted.app.preprocess.roi_extractor import RoiExtractor


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
        for i, (frame, ax) in enumerate(zip(self.frames, axes, strict=False)):
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
    edge_detector = EdgeExtractor(frames)
    edge_frames = edge_detector.process_frames()
    plot_frames(edge_frames)


def plot_flow_frames(
    frames: List["np.ndarray"], last_frame_index: int = -1
) -> None:
    if last_frame_index == -1:
        last_frame_index = len(frames) - 1
    flow_calculator = FlowCalculator(frames, last_frame_index)
    flow_frames = flow_calculator.calculate(plot_each_frame=False)
    plot_frames(flow_frames)


def plot_hog_frames(frames: List["np.ndarray"]) -> None:
    hog_extractor = HOGExtractor(frames)
    hog_features, hog_frames = hog_extractor.process_frames()
    plot_frames(hog_frames, is_gray_scale=True)


def plot_lbp_frames(frames: List[np.ndarray]) -> None:
    lbp_extractor = LBPExtractor(frames)
    lbp_frames = lbp_extractor.get_lbp_frames()
    lbp_features = lbp_extractor.get_lbp_features()

    num_frames = len(frames) * 2
    cols = int(math.ceil(math.sqrt(num_frames)))
    rows = int(math.ceil(num_frames / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, (lbp_frame, hist) in enumerate(
        zip(lbp_frames, lbp_features, strict=False)
    ):
        ax_img = axes[2 * i]
        ax_img.imshow(lbp_frame, cmap="gray")
        ax_img.set_title(f"LBP Frame {i + 1}")
        ax_img.axis("off")
        bins = np.arange(0, len(hist) + 1)

        ax_hist = axes[2 * i + 1]
        ax_hist.bar(
            bins[:-1],
            hist,
            width=1,
            color="gray",
            edgecolor="black",
            alpha=0.7,
        )
        ax_hist.set_title(f"Histogram {i + 1}")
        ax_hist.set_xlabel("LBP Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_xlim([0, len(bins) - 1])

    for ax in axes[num_frames:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_color_hist(
    frames: List[np.ndarray],
    to_color=cv2.COLOR_BGR2HSV,
    colors=("h", "s", "v"),
    normalize=False,
) -> None:
    color_hist_extractor = ColorHistogram(frames)
    frames_hists = color_hist_extractor.process_frames(
        to_color, separate_colors=True, normalize=normalize
    )
    cols = 1 + len(colors)
    rows = len(frames)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    subplot_index = 0  # Keep track of subplot index
    for i, (frame, hists) in enumerate(
        zip(frames, frames_hists, strict=False)
    ):
        ax_img = axes[subplot_index]  # Use subplot_index
        ax_img.imshow(cv2.cvtColor(frame, to_color))
        ax_img.set_title(f"Frame {i + 1}")
        ax_img.axis("off")
        subplot_index += 1  # Increment for the next subplot

        # More descriptive colors for histograms, and ensure enough colors
        hist_cols = [
            "red",
            "green",
            "blue",
            "purple",
            "orange",
            "cyan",
        ]  # Extended color list
        hist_labels = (
            list(colors)
            if colors
            else [f"Channel {j+1}" for j in range(len(hists))]
        )  # Use provided colors or generic labels

        for j, (color, hist) in enumerate(
            zip(hist_labels, hists, strict=False)
        ):
            if subplot_index < len(
                axes
            ):  # Check if subplot index is within bounds
                ax_hist = axes[subplot_index]  # Use subplot_index
                bins = np.arange(257)
                hist = hist.flatten()
                ax_hist.bar(
                    bins[:-1],
                    hist,
                    width=1,
                    color=hist_cols[
                        j % len(hist_cols)
                    ],  # Cycle through colors if more histograms than colors
                    alpha=0.7,
                )
                ax_hist.set_title(
                    f"{color.upper()} Histogram"
                )  # Use color label in title
                ax_hist.set_xlabel("Pixel Value")  # Corrected x-axis label
                ax_hist.set_ylabel("Frequency")
                ax_hist.set_xlim([0, len(bins) - 1])
                subplot_index += 1  # Increment for the next subplot

    # Turn off any unused axes
    for ax in axes[subplot_index:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_contour(frames: List[np.ndarray]) -> None:
    contour_detector = ContourExtractor(frames)
    contour_frames = contour_detector.process_frames()
    plot_frames(contour_frames)


def plot_haar_frames(frames: List["np.ndarray"], plot: bool = True):
    classifier = cv2.CascadeClassifier()
    if not classifier.load(
        cv2.samples.findFile(
            "handcrafted/app/features/extractor/haar/haarcascades/face.xml"
        )
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
