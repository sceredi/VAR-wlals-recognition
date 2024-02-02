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


def svm_test(dataset: Dataset, glosses: List[str]):

    videos = [video for video in dataset.videos if video.gloss in glosses]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    i = 1
    for video in videos:
        print(f"Processing video {i}/{len(videos)}")
        i += 1
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=False)
        hog_frames = get_hog_frames(roi_frames)
        haar_frames, face_rects = get_haar_frames(roi_frames)
        skin_frames = get_skin_frames(roi_frames, face_rects)
        edge_frames = get_edge_frames(skin_frames)
        # roi_frames = np.array(roi_frames).flatten()
        hog_frames = [frame.flatten() for frame in hog_frames]
        skin_frames = [frame.flatten() for frame in skin_frames]
        edge_frames = [frame.flatten() for frame in edge_frames]
        hog_frames = np.array(hog_frames).flatten()
        skin_frames = np.array(skin_frames).flatten()
        edge_frames = np.array(edge_frames).flatten()

        features = np.concatenate((hog_frames, skin_frames, edge_frames))
        print(features.shape)
        if video.split == "train":
            X_train.append(features)
            Y_train.append(glosses.index(video.gloss))
        else:
            X_test.append(features)
            Y_test.append(glosses.index(video.gloss))
    X_train = np.array(X_train)
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {len(Y_train)}")
    svc = SVC(verbose=True)
    print("Training SVM...")
    svc.fit(X_train, Y_train)
    print("Testing SVM...")
    X_test = np.array(X_test)
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
    cfm_plot.figure.savefig("cfm.png")


def fix_and_save(dataset: Dataset):
    for video in dataset.videos:
        print("Processing video: ", video.get_path())
        roi_frames = get_roi_frames(video, remove_background=True)
        folder = "data/preprocessed/" + video.video_id
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, frame in enumerate(roi_frames):
            cv2.imwrite(folder + "/" + str(i) + ".jpg", frame)


if __name__ == "__main__":
    start_time = time.perf_counter()
    # -------------------------------------
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = pd.read_csv("data/wlasl_class_list.txt", sep='\t', header=None)
    glosses = glosses[1].tolist()
    # fix_and_save(dataset)
    svm_test(dataset, glosses[:3])
    # for video in dataset.videos:
    #     print("Plotting video: ", video.get_path())
    #     plot_video(video)
    #     plot_video_with_hog(video)
    # plot_video(dataset.videos[0])
    # -------------------------------------
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")
