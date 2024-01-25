from typing import List
import numpy as np
from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.extractor.hog_extractor import HOGExtractor
from app.hand.detector import HandsDetector
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator


def plot(frames: List["np.ndarray"]) -> None:
    plotter = FramesPlotter(frames)
    plotter.plot_grid()


def get_roi_frames(video: Video, print=False) -> List["np.ndarray"]:
    roi_extractor = RoiExtractor(video.get_frames(), video.bbox)
    roi_frames = roi_extractor.extract(remove_background=False)
    if print:
        plot(roi_frames)
    return roi_frames


def get_edge_frames(frames: List["np.ndarray"], print=False) -> List["np.ndarray"]:
    edge_detector = EdgeDetector(frames)
    edge_frames = edge_detector.detect()
    if print:
        plot(edge_frames)
    return edge_frames


def get_flow_frames(frames: List["np.ndarray"], print=False) -> List["np.ndarray"]:
    flow_calculator = FlowCalculator(frames)
    flow_frames = flow_calculator.calculate()
    if print:
        plot(flow_frames)
    return flow_frames


def get_hog_frames(frames: List["np.ndarray"], print=False) -> List["np.ndarray"]:
    hog_extractor = HOGExtractor(frames)
    hog_frames = hog_extractor.process_frames()
    if print:
        plot(hog_frames)
    return hog_frames


def get_hands_frames(frames: List["np.ndarray"], print=False) -> List["np.ndarray"]:
    hands_detector = HandsDetector(frames)
    hands_frames, _ = hands_detector.detect()
    if print:
        plot(hands_frames)
    return hands_frames


def plot_video(video: Video) -> None:
    roi_frames = get_roi_frames(video)
    edge_frames = get_edge_frames(roi_frames)
    flow_frames = get_flow_frames(roi_frames)
    hog_frames = get_hog_frames(roi_frames)
    hands_frames = get_hands_frames(roi_frames, print=True)


if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    for video in dataset.videos:
        print("Plotting video: ", video.get_path())
        plot_video(video)
        # plot_video_with_hog(video)
    # plot_video(dataset.videos[0])
