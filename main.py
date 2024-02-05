import os
from typing import List
import numpy as np
import cv2
from fastdtw import fastdtw
# from dtaidistance import dtw
from dtw import *
from tslearn.metrics import cdist_dtw

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
from scipy.spatial.distance import euclidean, cdist
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.preprocessing import MinMaxScaler

from app.utilities.file_zipper import FileZipper


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
    frames = current_video.get_frames()
    # hog_frames = get_hog_frames(frames)
    haar_frames, face_rects = get_haar_frames(frames)
    skin_frames = get_skin_frames(frames, face_rects, plot=True)
    # edge_frames = get_edge_frames(skin_frames, plot=True)
    # edge_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in edge_frames]
    flow_frames = get_flow_frames(skin_frames, plot=True)
    # contour_frames = detect_contour(frames, plot=True)


def compute_dtw_distance(seq1, seq2):
    # alignment = dtw(seq1, seq2, dist=euclidean)
    # return alignment.distance
    return fastdtw(seq1, seq2, dist=euclidean)


def flatten_frames(frames):
    new_frames = [frame.flatten() for frame in frames]
    new_frames = np.array(new_frames).flatten() # non de-commentare con una feature
    return new_frames


def process_video(videos, glosses):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    i = 1
    for video in videos:
        print(f"Processing video {i}/{len(videos)}")
        i += 1
        print("Processing video: ", video.get_path())
        roi_frames = video.get_frames()  # get_roi_frames(video, remove_background=False)
        hog_frames = get_hog_frames(roi_frames)
        haar_frames, face_rects = get_haar_frames(roi_frames)
        skin_frames = get_skin_frames(roi_frames, face_rects)
        # edge_frames = get_edge_frames(skin_frames)
        flow_frames = get_flow_frames(skin_frames)


        hog_frames = flatten_frames(hog_frames)
        skin_frames = flatten_frames(skin_frames)
        # edge_frames = flatten_frames(edge_frames)
        flow_frames = flatten_frames(flow_frames)
        print(hog_frames[0].shape)
        print(skin_frames[0].shape)
        print(flow_frames[0].shape)

        features = np.concatenate((flow_frames, hog_frames, skin_frames))  # , edge_frames))
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

    # X_train_dtw = [compute_dtw_distance(seq, X_train[0]) for seq in X_train]
    # X_test_dtw = [compute_dtw_distance(seq, X_test[0]) for seq in X_test]
    #
    # X_train = np.array(X_train_dtw).reshape(-1, 1)
    # X_test = np.array(X_test_dtw).reshape(-1, 1)

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
    # adds a , between each element of the list
    Y_pred = ", ".join(map(str, Y_pred))
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


def dtw_kernel(sequence1, sequence2):
    distance, path = fastdtw(sequence1, sequence2)
    print(f"Distance: {distance}")
    normalized_distance = distance / (1 + distance)
    similarity = np.exp(-normalized_distance)
    return similarity


def process_video_pair(i, j, videos):
    print(f"Processing video pair: {videos[i].get_path()} and {videos[j].get_path()}")

    frames_i = videos[i].get_frames()
    frames_j = videos[j].get_frames()

    hog_sequence1 = flatten_frames(get_hog_frames(frames_i))
    # haar_frames1, face_rects1 = get_haar_frames(frames_i)
    # skin_sequence1 = flatten_frames(get_skin_frames(frames_i, face_rects1))
    # contour_sequence1 = flatten_frames(detect_contour(frames_i))

    hog_sequence2 = flatten_frames(get_hog_frames(frames_j))
    # haar_frames2, face_rects2 = get_haar_frames(frames_j)
    # skin_sequence2 = flatten_frames(get_skin_frames(frames_j, face_rects2))
    # contour_sequence2 = flatten_frames(detect_contour(frames_j))
    #
    # sequence1 = np.concatenate((hog_sequence1, skin_sequence1, contour_sequence1))
    # sequence2 = np.concatenate((hog_sequence2, skin_sequence2, contour_sequence2))

    return dtw_kernel(hog_sequence1, hog_sequence2)


def similarity_matrix(dataset: Dataset, gloss: str):
    videos = [video for video in dataset.videos if video.gloss == gloss and video.split == "train"]

    n = len(videos)
    sim_matrix = np.zeros((n, n))
    print(f"Processing gloss: {gloss}")
    print(f"Number of videos: {n}")
    print(f"Dimension of similarity matrix: {sim_matrix.shape}")
    print(f"--------------------------------------------")
    z = 0
    zz = (((n * n) - n) // 2) + n  # numero elementi matrice triangolare superiore + diagonale
    for i in range(n):
        for j in range(i, n):
            z += 1
            print(f"Processing video: {z}/{zz}")
            sim_matrix[i, j] = 1.0 if i == j else process_video_pair(i, j, videos)
            sim_matrix[j, i] = sim_matrix[i, j]
            print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {sim_matrix[i, j]}")
            print(f"--------------------------------------------")
    return sim_matrix


def aggregate_similarity_matrix(similarity_matrix, method='mean'):
    if method == 'mean':
        return np.mean(similarity_matrix)
    elif method == 'max':
        return np.max(similarity_matrix)
    elif method == 'min':
        return np.min(similarity_matrix)
    elif method == 'sum':
        return np.sum(similarity_matrix)
    elif method == 'median':
        return np.median(similarity_matrix)
    else:
        raise ValueError("Metodo di aggregazione non valido.")


if __name__ == "__main__":
    start_time = time.perf_counter()
    # -------------------------------------
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = pd.read_csv("data/wlasl_class_list.txt", sep='\t', header=None)
    glosses = glosses[1].tolist()
    # -------------------------------------

    # fix_and_save(dataset)

    # -------------------------------------

    # for video in dataset.videos:
    #     print("Plotting video: ", video.get_path())
    #     plot_video(video)
    #
    # -------------------------------------

    svm_test(dataset, glosses[:5])

    # similarity_matrix = similarity_matrix(dataset, glosses[0])
    # print(similarity_matrix)

    # similarity_matrix_book_hog_train = [[1.0, 1.43478064e-06, 1.78389937e-06, 1.00215394e-06],
    #                                 [1.43478064e-06, 1.0, 1.69621180e-06, 1.05909392e-06],
    #                                 [1.78389937e-06, 1.69621180e-06, 1.0, 1.18465465e-06],
    #                                 [1.00215394e-06, 1.05909392e-06, 1.18465465e-06, 1.0]]
    #
    # similarity_matrix_book_hog_train_norm = [[1.0, 0.36787997, 0.3678801, 0.36787981],
    #                                     [0.36787997, 1.0, 0.36788007, 0.36787983],
    #                                     [0.3678801, 0.36788007, 1.0, 0.36787988],
    #                                     [0.36787981, 0.36787983, 0.36787988, 1.0]]
    # aggregated_value = aggregate_similarity_matrix(similarity_matrix_book_hog_train_norm, method='mean')
    # print("Aggregated Value:", aggregated_value)

    # -------------------------------------

    # feature_extractor = FeatureExtractor(dataset, glosses)
    # feature_extractor.extract_and_save_all_features("all_video_features.joblib")
    #
    # file_zipper = FileZipper()
    # file_to_zip = "all_video_features.joblib"
    # file_zipper.zip_file(file_to_zip)
    #
    # zip_filename = "all_video_features.joblib.zip"
    # file_zipper.unzip_file(zip_filename)
    #
    # features = feature_extractor.load_features("all_video_features.joblib")
    # print(features.keys())
    # print(features.values())
    # feature_extractor.delete_features("all_video_features.joblib")

    # -------------------------------------
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")
