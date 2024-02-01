import os
from typing import List
import numpy as np
import cv2
from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.extractor.hog_extractor import HOGExtractor
from app.extractor.skin import SkinExtractor
from app.haar.detector import HaarDetector
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator
import time
from scipy.spatial.distance import euclidean
import dtw
from sklearn.svm import SVC


def plot_frames(frames: List["np.ndarray"]) -> None:
    plotter = FramesPlotter(frames)
    plotter.plot_grid()


def get_roi_frames(
        video: Video, remove_background=False, plot=False
) -> List["np.ndarray"]:
    roi_extractor = RoiExtractor(video.get_frames(), video.bbox, resize=224)
    roi_frames = roi_extractor.extract(remove_background=remove_background)
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


def get_haar_frames(frames: List["np.ndarray"], plot=False):
    classifier = cv2.CascadeClassifier()
    # if not classifier.load(cv2.samples.findFile('app/haar/haarcascades/hand.xml')):
    #     print('Error loading hand cascade')
    #     exit(1)
    if not classifier.load(cv2.samples.findFile("app/haar/haarcascades/face.xml")):
        print("Error loading face cascade")
        exit(1)
    haar_detector = HaarDetector(frames, classifier, detections_to_keep=3)
    face_frames, rects = haar_detector.detect()
    if plot:
        plot_frames(face_frames)
    return face_frames, rects


def get_skin_frames(frames: List["np.ndarray"], face_rects, plot=False):
    skin_extractor = SkinExtractor(frames, face_rects)
    skin_frames = skin_extractor.extract()
    if plot:
        plot_frames(skin_frames)
    return skin_frames


def plot_video(current_video: Video) -> None:
    roi_frames = get_roi_frames(current_video, remove_background=True, plot=True)
    # hog_frames = get_hog_frames(roi_frames)
    # haar_frames, face_rects = get_haar_frames(roi_frames)
    # skin_frames = get_skin_frames(roi_frames, face_rects)
    # edge_frames = get_edge_frames(skin_frames)
    # edge_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in edge_frames]
    # flow_frames = get_flow_frames(edge_frames, plot=True)

def svm_test(dataset: Dataset):
    videos = [video for video in dataset.videos if video.gloss == "book" or video.gloss == "drink"]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for video in videos:
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=False)
        hog_frames = get_hog_frames(roi_frames)
        # roi_frames = np.array(roi_frames).flatten()
        hog_frames = [frame.flatten() for frame in hog_frames]
        hog_frames = np.array(hog_frames).flatten()
        features = hog_frames
        if video.split == "test":
            X_test.append(features)
            if video.gloss == "book":
                Y_train.append(0)
            else:
                Y_train.append(1)
        else:
            X_train.append(features)
            if video.gloss == "book":
                Y_test.append(0)
            else:
                Y_test.append(1)
    svc = SVC(kernel='precomputed')
    print("Training SVM...")
    X_train = np.array(X_train)
    print(X_train.shape)
    svc.fit(X_train, Y_train)
    print("Testing SVM...")
    y_pred = svc.predict(X_test)
    correct_predictions = np.sum(y_pred == Y_test)
    total_predictions = len(Y_test)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Accuracy: {accuracy:.2f}%")


def fix_and_save(dataset: Dataset):
    for video in dataset.videos:
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=True)
        folder = "data/preprocessed/" + video.video_id
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, frame in enumerate(roi_frames):
            cv2.imwrite(folder + "/" + str(i) + ".jpg", frame)
        break


if __name__ == "__main__":
    start_time = time.perf_counter()
    # -------------------------------------
    dataset = Dataset("data/WLASL_v0.3.json")
    fix_and_save(dataset)
    # svm_test(dataset)
    # for video in dataset.videos:
    #     print("Plotting video: ", video.get_path())
    #     plot_video(video)
    #     plot_video_with_hog(video)
    # plot_video(dataset.videos[0])
    # -------------------------------------
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")
