import os
from typing import List
import numpy as np
import cv2
from fastdtw import fastdtw
# from dtaidistance import dtw
from dtw import *

from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.extractor.contour_extractor import ContourDetector
from app.extractor.feature_extractor import FeatureExtractor
from app.extractor.hog_extractor import HOGExtractor
from app.extractor.skin import SkinExtractor
from app.haar.detector import HaarDetector
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator
import time
from scipy.spatial.distance import euclidean
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


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


def detect_contour(frames: List[np.ndarray], plot=False) -> List[np.ndarray]:
    contour_detector = ContourDetector(frames)
    contour_frames = contour_detector.detect()
    if plot:
        plot_frames(contour_frames)
    return contour_frames


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
    roi_frames = get_roi_frames(current_video, remove_background=False)
    # hog_frames = get_hog_frames(roi_frames)
    haar_frames, face_rects = get_haar_frames(roi_frames, plot=True)
    # skin_frames = get_skin_frames(roi_frames, face_rects)
    # edge_frames = get_edge_frames(skin_frames)
    # edge_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in edge_frames]
    # flow_frames = get_flow_frames(edge_frames, plot=True)
    contour_frames = detect_contour(roi_frames, plot=True)


def compute_dtw_distance(seq1, seq2):
    # alignment = dtw(seq1, seq2, dist=euclidean)
    # return alignment.distance
    return fastdtw(seq1, seq2, dist=euclidean)


def flatten_frames(frames):
    new_frames = [frame.flatten() for frame in frames]
    # new_frames = np.array(new_frames).flatten()
    return new_frames


def process_video(videos, glosses):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    i = 1
    for video in videos[:5]:
        print(f"Processing video {i}/{len(videos)}")
        i += 1
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=False)
        hog_frames = get_hog_frames(roi_frames)
        haar_frames, face_rects = get_haar_frames(roi_frames)
        skin_frames = get_skin_frames(roi_frames, face_rects)
        edge_frames = get_edge_frames(skin_frames)

        hog_frames = flatten_frames(hog_frames)
        skin_frames = flatten_frames(skin_frames)
        # edge_frames = flatten_frames(edge_frames)

        features = np.concatenate((hog_frames, skin_frames))  # , edge_frames))
        print(features.shape)
        if video.split == "train":
            X_train.append(features)
            Y_train.append(glosses.index(video.gloss))
        else:
            X_test.append(features)
            Y_test.append(glosses.index(video.gloss))
    return np.array(X_train), np.array(X_test), Y_train, Y_test


def svm_test(dataset: Dataset, glosses: List[str]):
    videos = [video for video in dataset.videos if video.gloss in glosses]
    X_train, X_test, Y_train, Y_test = process_video(videos, glosses)

    X_train_dtw = [compute_dtw_distance(seq, X_train[0]) for seq in X_train]
    X_test_dtw = [compute_dtw_distance(seq, X_test[0]) for seq in X_test]

    X_train = np.array(X_train_dtw).reshape(-1, 1)
    X_test = np.array(X_test_dtw).reshape(-1, 1)

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {len(Y_train)}")
    svc = SVC(verbose=True)
    print("Training SVM...")
    svc.fit(X_train, Y_train)
    print("Testing SVM...")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {len(Y_test)}")
    Y_pred = svc.predict(X_test)
    correct_predictions = np.sum(Y_pred == Y_test)
    total_predictions = len(Y_test)
    print("X_test", X_test)
    print("Y_test", Y_test)
    print("Y_pred", Y_pred)

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy:.2f}%")

    cfm = confusion_matrix(Y_test, Y_pred)
    df_cfm = pd.DataFrame(cfm, index=glosses, columns=glosses)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("cfm/cfm.png")


def fix_and_save(dataset: Dataset):
    for video in dataset.videos:
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=True)
        folder = "data/preprocessed/" + video.video_id
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, frame in enumerate(roi_frames):
            cv2.imwrite(folder + "/" + str(i) + ".jpg", frame)


def prova_dtw(dataset: Dataset, glosses: List[str]):
    distances = []
    videos = [video for video in dataset.videos if video.gloss in glosses]

    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            print(f"Processing video pair: {videos[i].get_path()} and {videos[j].get_path()}")

            # Estrai le caratteristiche HOG per entrambi i video
            hog_sequence1 = flatten_frames(get_hog_frames(get_roi_frames(videos[i])))
            hog_sequence2 = flatten_frames(get_hog_frames(get_roi_frames(videos[j])))

            # Calcola la distanza DTW
            distance, path = compute_dtw_distance(hog_sequence1, hog_sequence2)
            distances.append((videos[i].get_path(), videos[j].get_path(), distance, path))

            # print(f"DTW Distance: {distance}")

            print(f"DTW distance between {videos[i]} and {videos[j]}: {distance}")

            plt.figure(figsize=(8, 6))
            plt.plot(path, marker='o', linestyle='-', color='b')
            plt.title('Distanza DTW tra sequenze temporali')
            plt.xlabel('Coppie di sequenze')
            plt.ylabel('Distanza DTW')
            plt.show()


    # Stampa le distanze
    # for video1, video2, distance, path in distances:
    #     print(f"DTW distance between {video1} and {video2}: {distance}")
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(path, marker='o', linestyle='-', color='b')
    #     plt.title('Distanza DTW tra sequenze temporali')
    #     plt.xlabel('Coppie di sequenze')
    #     plt.ylabel('Distanza DTW')
    #     plt.show()


if __name__ == "__main__":
    start_time = time.perf_counter()
    # -------------------------------------
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = pd.read_csv("data/wlasl_class_list.txt", sep='\t', header=None)
    glosses = glosses[1].tolist()
    # fix_and_save(dataset)
    # svm_test(dataset, glosses[:2])
    # for video in dataset.videos:
    #     print("Plotting video: ", video.get_path())
    #     plot_video(video)
    #     plot_video_with_hog(video)
    # plot_video(dataset.videos[0])

    # prova_dtw(dataset, glosses[:2])

    feature_extractor = FeatureExtractor(dataset, glosses)
    feature_extractor.extract_and_save_all_features("all_video_features.joblib")
    features = feature_extractor.load_features("all_video_features.joblib")
    print(features.keys())
    print(features.values())

    # -------------------------------------
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")
