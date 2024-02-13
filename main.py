import gc
import json
import os
from typing import List, Tuple
import numpy as np
import cv2
from fastdtw import fastdtw

# from dtaidistance import dtw
from dtw import *
from tslearn.datasets import CachedDatasets
from tslearn.metrics import cdist_dtw
from tslearn.svm import TimeSeriesSVC

from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.extractor.contour_extractor import ContourDetector
from app.extractor.feature_extractor import FeatureExtractor
from app.extractor.hog_extractor import HOGExtractor
from app.extractor.skin import SkinExtractor
from app.haar.detector import HaarDetector
from app.pca.compute import compute_pca
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator
import time
from scipy.spatial.distance import euclidean, cdist
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler

from app.utilities.file_zipper import FileZipper


def plot_frames(frames: List["np.ndarray"], is_gray_scale=False) -> None:
    plotter = FramesPlotter(frames, to_rgb=not is_gray_scale)
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


def get_flow_frames(
    frames: List["np.ndarray"], last_frame_index: int = -1, plot=False
) -> List["np.ndarray"]:
    if last_frame_index == -1:
        last_frame_index = len(frames) - 1
    flow_calculator = FlowCalculator(frames, last_frame_index)
    flow_frames = flow_calculator.calculate(plot_each_frame=False)
    if plot:
        plot_frames(flow_frames)
    return flow_frames


def get_hog_frames(
    frames: List["np.ndarray"], plot=False
) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
    hog_extractor = HOGExtractor(frames)
    hog_features, hog_frames = hog_extractor.process_frames()
    if plot:
        plot_frames(hog_frames, is_gray_scale=True)
    return hog_features, hog_frames


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
    # hog_frames = get_hog_frames(frames, plot=True)
    # haar_frames, face_rects = get_haar_frames(frames)
    # skin_frames = get_skin_frames(frames, face_rects)
    # edge_frames = get_edge_frames(frames, plot=True)
    # edge_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in edge_frames]
    # flow_frames = get_flow_frames(frames, last_frame_index=current_video.frame_end)
    # contour_frames = detect_contour(frames, plot=True)


def compute_dtw_distance(seq1, seq2):
    # alignment = dtw(seq1, seq2, dist=euclidean)
    # return alignment.distance
    return fastdtw(seq1, seq2, dist=euclidean)


def flatten_frames(frames):
    new_frames = [frame.flatten() for frame in frames]
    # new_frames = np.array(new_frames).flatten()  # non de-commentare con una feature
    return new_frames


def process_video(videos, glosses):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    i = 1
    for video in videos:
        print(f"Processing gloss: {video.gloss}")
        print(f"Processing video {i}/{len(videos)}")
        i += 1
        print("Processing video: ", video.get_path())
        roi_frames = video.get_frames()  # get_roi_frames(video, remove_background=False)
        hog_frames = get_hog_frames(roi_frames)
        # haar_frames, face_rects = get_haar_frames(roi_frames)
        # skin_frames = get_skin_frames(roi_frames, face_rects)
        # edge_frames = get_edge_frames(roi_frames)
        contour_frames = detect_contour(roi_frames)

        hog_frames = flatten_frames(hog_frames)
        # skin_frames = flatten_frames(skin_frames)
        # edge_frames = flatten_frames(edge_frames)
        contour_frames = flatten_frames(contour_frames)

        features = np.concatenate((hog_frames, contour_frames), axis=1)
        print(features.shape)
        if video.split == "train":
            X_train.append(features)
            Y_train.append(glosses.index(video.gloss))
        else:
            X_test.append(features)
            Y_test.append(glosses.index(video.gloss))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    return X_train, X_test, Y_train, Y_test


def calculate_dtw_distance(sequences):
    num_sequences = len(sequences)
    dtw_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            dtw_matrix[i, j] = fastdtw(sequences[i], sequences[j], dist=euclidean)[0]
            dtw_matrix[j, i] = dtw_matrix[i, j]
    return [row for row in dtw_matrix]


def dtw_per_class(dataset: Dataset, glosses: List[str]):
    videos = [video for video in dataset.videos if video.gloss in glosses]


def svm_test(dataset: Dataset, glosses: List[str]):
    # videos = [video for video in dataset.videos if video.gloss in glosses]
    # X_train, X_test, Y_train, Y_test = process_video(videos, glosses)

    max_train_videos_per_gloss = 10
    max_test_videos_per_gloss = 10
    selected_videos_train = []
    selected_videos_test = []

    train_videos_per_gloss_count = {}
    test_videos_per_gloss_count = {}

    for video in dataset.videos:
        if video.gloss in glosses:
            if video.split == "train":
                count_train = train_videos_per_gloss_count.get(video.gloss, 0)
                if count_train < max_train_videos_per_gloss:
                    train_videos_per_gloss_count[video.gloss] = count_train + 1
                    selected_videos_train.append(video)
            else:
                count_test = test_videos_per_gloss_count.get(video.gloss, 0)
                if count_test < max_test_videos_per_gloss:
                    test_videos_per_gloss_count[video.gloss] = count_test + 1
                    selected_videos_test.append(video)

    gc.collect()
    print(selected_videos_train)
    print(selected_videos_test)

    X_train, X_test, Y_train, Y_test = process_video(selected_videos_train + selected_videos_test, glosses)

    # X_train_dtw = [compute_dtw_distance(seq, X_train[0]) for seq in X_train]
    # X_test_dtw = [compute_dtw_distance(seq, X_test[0]) for seq in X_test]
    #
    # X_train = np.array(X_train_dtw).reshape(-1, 1)
    # X_test = np.array(X_test_dtw).reshape(-1, 1)

    # {'book': [0.5259099561524061, 0.5259095810380596]}
    # {'drink': [0.4357935868563727, 0.4357932203568296]}

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {len(Y_train)}")
    svc = SVC(random_state=42, verbose=True)
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
    # Y_pred = ", ".join(map(str, Y_pred))
    print("Y_pred", Y_pred)

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy:.2f}%")

    cfm = confusion_matrix(Y_test, Y_pred)
    df_cfm = pd.DataFrame(cfm, index=glosses, columns=glosses)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("cfm/cfm4.png")


def svm_test_similarity(dataset: Dataset, glosses: List[str]):
    max_train_videos_per_gloss = 3
    max_test_videos_per_gloss = 2
    selected_videos_train = []
    selected_videos_test = []

    train_videos_per_gloss_count = {}
    test_videos_per_gloss_count = {}

    for video in dataset.videos:
        if video.gloss in glosses:
            if video.split == "train":
                count_train = train_videos_per_gloss_count.get(video.gloss, 0)
                if count_train < max_train_videos_per_gloss:
                    train_videos_per_gloss_count[video.gloss] = count_train + 1
                    selected_videos_train.append(video)
            else:
                count_test = test_videos_per_gloss_count.get(video.gloss, 0)
                if count_test < max_test_videos_per_gloss:
                    test_videos_per_gloss_count[video.gloss] = count_test + 1
                    selected_videos_test.append(video)

    gc.collect()
    print("Lunghezza train:", len(selected_videos_train))
    print(selected_videos_train)
    print("Lunghezza test:", len(selected_videos_test))
    print(selected_videos_test)

    # SIMILARITY MATRIX WITH DTW
    # X_train = similarity_matrix_training(selected_videos_train)
    with open('sim/similarity_matrix_10_3_2.json', 'r') as file:
        lines = file.read()
    matrice = np.array(eval(lines), dtype=float)
    print(matrice)
    X_train = matrice

    X_test = np.zeros((len(selected_videos_test), len(X_train)))

    z = 0
    zz = len(selected_videos_test) * len(selected_videos_train)
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            z += 1
            print(f"Processing video: {z}/{zz}")
            similarity = process_video_pair(selected_videos_test[i], selected_videos_train[j])
            X_test[i, j] = similarity

    Y_train = []
    Y_test = []

    for video in selected_videos_train + selected_videos_test:
        if video.split == "train":
            Y_train.append(glosses.index(video.gloss))
        else:
            Y_test.append(glosses.index(video.gloss))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    svc = SVC(random_state=42, verbose=True)
    print("Training SVM...")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {len(Y_train)}")
    svc.fit(X_train, Y_train)
    print("X_train", X_train)
    print("Y_train", Y_train)
    print("Testing SVM...")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {len(Y_test)}")
    Y_pred = svc.predict(X_test)
    print("X_test", X_test)
    print("Y_test", Y_test)
    print("Y_pred", Y_pred)

    score = svc.score(X_test, Y_test)
    print("Correct classification rate:", score)
    print(f"Accuracy: {score * 100:.2f}%")

    n_classes = len(set(Y_train))

    plt.figure()
    support_vectors = svc.support_vectors_
    for i, cl in enumerate(set(Y_train)):
        plt.subplot(n_classes, 1, i + 1)
        plt.title("Support vectors for class %d" % cl)
        for ts in support_vectors[i]:
            plt.plot(ts.ravel())

    plt.tight_layout()
    plt.show()

    cfm = confusion_matrix(Y_test, Y_pred)
    df_cfm = pd.DataFrame(cfm, index=glosses, columns=glosses)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("cfm/cfm.png")

    plt.figure(figsize=(10, 7))
    plt.scatter(X_test[:, 0], Y_test, c=Y_pred, cmap='viridis', edgecolors='k')
    plt.title('Classificazioni risultanti da SVC con kernel RBF e DTW')
    plt.xlabel('DTW Distance')
    plt.ylabel('Classe')
    plt.show()


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
    print(f"Similarity: {similarity}")
    return similarity


def process_video_pair(video_i, video_j):
    print(f"Processing video pair: {video_i.get_path()} and {video_j.get_path()}")

    frames_i = video_i.get_frames()
    frames_j = video_j.get_frames()

    hog_sequence1 = flatten_frames(get_hog_frames(frames_i))
    print(f"hog_sequence1 len: {len(hog_sequence1)}")
    haar_frames1, face_rects1 = get_haar_frames(frames_i)
    haar_sequence1 = flatten_frames(haar_frames1)
    print(f"haar_sequence1 len: {len(haar_sequence1)}")
    # skin_sequence1 = flatten_frames(get_skin_frames(frames_i, face_rects1))
    # print(f"skin_sequence1 len: {len(skin_sequence1)}")
    edge_sequence1 = flatten_frames(get_edge_frames(frames_i))
    print(f"edge_sequence1 len: {len(edge_sequence1)}")
    # contour_sequence1 = flatten_frames(detect_contour(frames_i))
    # print(f"contour_sequence1 len: {len(contour_sequence1)}")
    # flow_sequence1 = flatten_frames(get_flow_frames(frames_i))
    # print(f"flow_sequence1 len: {len(flow_sequence1)}")

    _, hog_frames_j = get_hog_frames(frames_j)
    hog_sequence2 = flatten_frames(hog_frames_j)
    # haar_frames2, face_rects2 = get_haar_frames(frames_j)
    # skin_sequence2 = flatten_frames(get_skin_frames(frames_j, face_rects2))
    # print(f"skin_sequence2 len: {len(skin_sequence2)}")
    edge_sequence2 = flatten_frames(get_edge_frames(frames_j))
    print(f"edge_sequence2 len: {len(edge_sequence2)}")
    # contour_sequence2 = flatten_frames(detect_contour(frames_j))
    # print(f"contour_sequence2 len: {len(contour_sequence2)}")
    # flow_sequence2 = flatten_frames(get_flow_frames(frames_j))
    # print(f"flow_sequence2 len: {len(flow_sequence2)}")

    hog_sequence1 = standardize_features(hog_sequence1)
    hog_sequence2 = standardize_features(hog_sequence2)
    haar_sequence1 = standardize_features(haar_sequence1)
    haar_sequence2 = standardize_features(haar_sequence2)
    edge_sequence1 = standardize_features(edge_sequence1)
    edge_sequence2 = standardize_features(edge_sequence2)
    # skin_sequence1 = standardize_features(skin_sequence1)
    # skin_sequence2 = standardize_features(skin_sequence2)
    # contour_sequence1 = standardize_features(contour_sequence1)
    # contour_sequence2 = standardize_features(contour_sequence2)
    # flow_sequence1 = standardize_features(flow_sequence1)
    # flow_sequence2 = standardize_features(flow_sequence2)

    sequence1 = np.concatenate((hog_sequence1, haar_sequence1, edge_sequence1), axis=1)
    sequence2 = np.concatenate((hog_sequence2, haar_sequence2, edge_sequence2), axis=1)

    similarity = dtw_kernel(sequence1, sequence2)
    # similarity_skin = dtw_kernel(skin_sequence1, skin_sequence2)
    # print(f"Similarity Skin: {similarity_skin}")
    # similarity_contour = dtw_kernel(contour_sequence1, contour_sequence2)
    # print(f"Similarity Contour: {similarity_contour}")

    # n_features = 3.0
    # similarity_combined = (similarity_hog + similarity_skin + similarity_contour) / n_features

    # return similarity_hog, similarity_contour
    return similarity


def standardize_features(features):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features


def similarity_matrix(dataset: Dataset, gloss: str):
    videos = [
        video
        for video in dataset.videos
        if video.gloss == gloss and video.split == "train"
    ]

    n = len(videos)
    sim_matrix = np.zeros((n, n))
    sim_matrix_hog = np.zeros((n, n))
    # sim_matrix_skin = np.zeros((n, n))
    sim_matrix_contour = np.zeros((n, n))
    print(f"Processing gloss: {gloss}")
    print(f"Number of videos: {n}")
    print(f"--------------------------------------------")
    z = 0
    zz = (
        ((n * n) - n) // 2
    ) + n  # numero elementi matrice triangolare superiore + diagonale
    for i in range(n):
        for j in range(i, n):
            z += 1
            print(f"Processing video: {z}/{zz}")
            # sim_matrix[i, j] = 1.0 if i == j else process_video_pair(i, j, videos)
            # sim_matrix[j, i] = sim_matrix[i, j]

            if i == j:
                similarity = 1.0
                similarity_hog = 1.0
                # similarity_skin = 1.0
                similarity_contour = 1.0
            else:
                # similarity_hog, similarity_contour = process_video_pair(i, j, videos)
                similarity = process_video_pair_std(i, j, videos)

            sim_matrix[i, j] = similarity
            sim_matrix[j, i] = similarity

            # sim_matrix_hog[i, j] = similarity_hog
            # sim_matrix_hog[j, i] = similarity_hog

            # sim_matrix_skin[i, j] = similarity_skin
            # sim_matrix_skin[j, i] = similarity_skin

            # sim_matrix_contour[i, j] = similarity_contour
            # sim_matrix_contour[j, i] = similarity_contour

            print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {sim_matrix[i, j]}")
            # print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {sim_matrix_hog[i, j]}")
            # print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {sim_matrix_skin[i, j]}")
            # print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {sim_matrix_contour[i, j]}")
            print(f"--------------------------------------------")

    # mean_sim_matrix_hog = np.mean(sim_matrix_hog)
    # mean_sim_matrix_skin = np.mean(sim_matrix_skin)
    # mean_sim_matrix_contour = np.mean(sim_matrix_contour)

    # sim_dict = {
    #     gloss: [mean_sim_matrix_hog, mean_sim_matrix_contour]
    # }

    return sim_matrix  # sim_dict

def similarity_matrix_training(videos):
    n = len(videos)
    M = np.zeros((n, n))
    print(f"Number of videos: {n}")
    print(f"--------------------------------------------")
    z = 0
    zz = (((n * n) - n) // 2) + n
    for i in range(n):
        for j in range(i, n):
            z += 1
            print(f"Processing video: {z}/{zz}")
            similarity = 1.0 if i == j else process_video_pair(videos[i], videos[j])
            M[i, j] = similarity
            # M[j, i] = similarity

            print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {M[i, j]}")
            print(f"--------------------------------------------")

    # mean_sim_matrix_hog = np.mean(sim_matrix_hog)
    # mean_sim_matrix_skin = np.mean(sim_matrix_skin)
    # mean_sim_matrix_contour = np.mean(sim_matrix_contour)

    # sim_dict = {
    #     gloss: [mean_sim_matrix_hog, mean_sim_matrix_contour]
    # }

    return sim_matrix  # sim_dict


def similarity_matrix_training(videos):
    n = len(videos)
    M = np.zeros((n, n))
    print(f"Number of videos: {n}")
    print(f"--------------------------------------------")
    z = 0
    zz = (((n * n) - n) // 2) + n
    for i in range(n):
        for j in range(i, n):
            z += 1
            print(f"Processing video: {z}/{zz}")
            similarity = 1.0 if i == j else process_video_pair(videos[i], videos[j])
            M[i, j] = similarity
            # M[j, i] = similarity

            print(f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {M[i, j]}")
            print(f"--------------------------------------------")
    M = M + M.T - np.diag(M.diagonal())
    np.set_printoptions(precision=17, suppress=True)
    print(M)
    return M


def aggregate_similarity_matrix(similarity_matrix, method="mean"):
    if method == "mean":
        return np.mean(similarity_matrix)
    elif method == "max":
        return np.max(similarity_matrix)
    elif method == "min":
        return np.min(similarity_matrix)
    elif method == "sum":
        return np.sum(similarity_matrix)
    elif method == "median":
        return np.median(similarity_matrix)
    else:
        raise ValueError("Metodo di aggregazione non valido.")


if __name__ == "__main__":
    start_time = time.perf_counter()
    # -------------------------------------
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = pd.read_csv("data/wlasl_class_list.txt", sep="\t", header=None)
    glosses = glosses[1].tolist()
    # -------------------------------------

    # fix_and_save(dataset)

    # -------------------------------------

    # for video in dataset.videos:
    #     print("Plotting video: ", video.get_path())
    #     plot_video(video)
    #
    # -------------------------------------

    # svm_test(dataset, glosses[:3])  # con 10 10: 55.56%
    # knn_classifier(dataset, glosses[:3])
    svm_test_similarity(dataset, glosses[1:3])

    # for gloss in glosses:
    #     videos = [video for video in dataset.videos if video.gloss == gloss and video.split == "train"]
    #     print(f"Number of videos for gloss {gloss}: {len(videos)}")

    # similarity = similarity_matrix(dataset, glosses[0])
    # print(similarity)
    #
    # with open('sim/similarity_book_std.json', 'w') as json_file:
    #     json.dump(similarity, json_file)

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
