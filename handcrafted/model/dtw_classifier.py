from typing import List

import numpy as np
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler

from handcrafted.app.dataset.dataset import Dataset
from handcrafted.app.dataset.video import Video


def standardize_features(features: np.ndarray) -> np.ndarray:
    """Standardize the feature set."""
    scaler = StandardScaler()
    return scaler.fit_transform(features)


class DTWClassifier:
    def __init__(self, dataset: Dataset, glosses: List[str]) -> None:
        self.dataset = dataset
        self.glosses = glosses
        self.train_videos = []
        self.test_videos = []

    def train_test_videos(self, num_glosses=10):
        """Create training and testing video sets."""
        selected_videos_train = []
        selected_videos_test = []

        for video in self.dataset.videos:
            if video.gloss in self.glosses[1:num_glosses]:
                if video.split == "train":
                    selected_videos_train.append(video)
                else:
                    selected_videos_test.append(video)

        print(
            f"Train videos: {len(selected_videos_train)}, Test videos: {len(selected_videos_test)}"
        )
        self.train_videos = selected_videos_train
        self.test_videos = selected_videos_test

    @staticmethod
    def dtw_kernel(sequence1: np.ndarray, sequence2: np.ndarray) -> float:
        """Calculate the DTW similarity kernel."""
        distance, _ = fastdtw(sequence1, sequence2)
        normalized_distance = distance / (1 + distance)
        similarity = np.exp(-normalized_distance)
        print(f"Similarity: {similarity}")
        return similarity

    def process_video_pair(self, video_i: Video, video_j: Video) -> float:
        """Process and compute similarity between two video pairs."""
        print(
            f"Processing video pair: {video_i.get_path()} and {video_j.get_path()}"
        )
        frames_i, frames_j = video_i.get_frames(), video_j.get_frames()
        print(f"len 1st: {len(frames_i)}, 2nd: {len(frames_j)}")

        sequence1 = video_i.features_container.get_all_features()
        sequence2 = video_j.features_container.get_all_features()

        return self.dtw_kernel(sequence1, sequence2)

    def similarity_matrix_training(self, videos: List[Video]) -> np.ndarray:
        """Compute the training similarity matrix."""
        n = len(videos)
        M = np.zeros((n, n))

        z = 0
        zz = (((n * n) - n) // 2) + n

        for i in range(n):
            for j in range(i, n):
                z += 1
                print(f"Processing video: {z}/{zz}")
                similarity = (
                    1.0
                    if i == j
                    else self.process_video_pair(videos[i], videos[j])
                )
                M[i, j] = similarity
                print(
                    f"Similarity between {videos[i].video_id} and {videos[j].video_id}: {similarity}"
                )

        M = M + M.T - np.diag(M.diagonal())
        return M

    def similarity_matrix_test(self, X_train_len: int) -> np.ndarray:
        """Compute the test similarity matrix."""
        X_test = np.zeros((len(self.test_videos), X_train_len))

        z = 0
        zz = len(self.test_videos) * len(self.train_videos)

        for i in range(len(self.test_videos)):
            for j in range(X_train_len):
                z += 1
                print(f"Processing video: {z}/{zz}")
                similarity = self.process_video_pair(
                    self.test_videos[i], self.train_videos[j]
                )
                X_test[i, j] = similarity

        print(X_test)
        return X_test

    def compute_dtw_similarity_matrix(self) -> tuple:
        """Compute the final DTW similarity matrices for training and testing."""
        # X_train = self.similarity_matrix_training(self.train_videos)
        X_test = self.similarity_matrix_test(
            len(self.train_videos)
        )  # (len(X_train))

        y_train = [
            self.glosses.index(video.gloss) for video in self.train_videos
        ]
        y_test = [
            self.glosses.index(video.gloss) for video in self.test_videos
        ]

        # return X_train.reshape((X_train.shape[0], -1)), X_test.reshape((X_test.shape[0], -1)), y_train, y_test
        return X_test.reshape((X_test.shape[0], -1)), y_train, y_test

    @staticmethod
    def dtw_predict(X_test: np.ndarray, y_train: List[int]) -> List[int]:
        """Predict the classes for test data."""
        nearest_neighbor_indices = np.argmax(X_test, axis=1)
        return [y_train[idx] for idx in nearest_neighbor_indices]
