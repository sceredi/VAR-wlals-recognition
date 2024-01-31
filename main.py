from typing import List
from matplotlib.pyplot import cla
import numpy as np
import cv2
from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.extractor.hog_extractor import HOGExtractor
from app.hand.detector import HandsDetector
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator


def plot_frames(frames: List["np.ndarray"]) -> None:
    plotter = FramesPlotter(frames)
    plotter.plot_grid()


def get_roi_frames(current_video: Video, plot=False) -> List["np.ndarray"]:
    roi_extractor = RoiExtractor(current_video.get_frames(), current_video.bbox)
    roi_frames = roi_extractor.extract(remove_background=True)
    if plot:
        plot_frames(roi_frames)
    return roi_frames


def get_edge_frames(frames: List["np.ndarray"], plot=False) -> List["np.ndarray"]:
    edge_detector = EdgeDetector(frames)
    edge_frames = edge_detector.detect()
    if plot:
        plot_frames(edge_frames)
    return edge_frames


def get_flow_frames(frames: List["np.ndarray"], plot=False) -> List["np.ndarray"]:
    flow_calculator = FlowCalculator(frames)
    flow_frames = flow_calculator.calculate()
    if plot:
        plot_frames(flow_frames)
    return flow_frames


def get_hog_frames(frames: List["np.ndarray"], plot=False) -> List["np.ndarray"]:
    hog_extractor = HOGExtractor(frames)
    hog_frames = hog_extractor.process_frames()
    if plot:
        plot_frames(hog_frames)
    return hog_frames


def detect_contour(frames: List[np.ndarray], plot=False) -> List[np.ndarray]:

    result_frames = []
    for frame in frames:
        # Converte il frame in scala di grigi
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Applica un filtro di smoothing (ad esempio, filtro Gaussiano)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Esegui la binarizzazione per enfatizzare i contorni delle mani
        _, binary_frame = cv2.threshold(blurred_frame, 100, 255, cv2.THRESH_BINARY)

        # Trova i contorni nell'immagine binarizzata
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Trova i contorni che potrebbero rappresentare le mani
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

        # Disegna i contorni sul frame originale
        result_frame = frame.copy()
        cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)

        result_frames.append(result_frame)

    if plot:
        plot_frames(result_frames)
    return result_frames


def get_hands_frames(frames: List["np.ndarray"], plot=False) -> List["np.ndarray"]:
    classifier = cv2.CascadeClassifier()
    # if not classifier.load(cv2.samples.findFile('app/hand/haarcascades/hand.xml')):
    #     print('Error loading hand cascade')
    #     exit(1)
    if not classifier.load(cv2.samples.findFile("app/hand/haarcascades/hand.xml")):
        print("Error loading face cascade")
        exit(1)
    hands_detector = HandsDetector(frames, classifier)
    hands_frames, _ = hands_detector.detect()
    if plot:
        plot_frames(hands_frames)
    return hands_frames


def plot_video(current_video: Video) -> None:
    roi_frames = get_roi_frames(current_video)
    edge_frames = get_edge_frames(roi_frames)
    flow_frames = get_flow_frames(roi_frames)
    hog_frames = get_hog_frames(roi_frames)
    hands_frames = detect_contour(roi_frames, plot=True)


if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    for video in dataset.videos:
        print("Plotting video: ", video.get_path())
        plot_video(video)
        # plot_video_with_hog(video)
    # plot_video(dataset.videos[0])
