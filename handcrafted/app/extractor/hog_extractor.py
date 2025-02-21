from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog


class HOGExtractor:
    def __init__(
        self,
        frames: List["np.ndarray"],
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    ) -> None:
        self.frames = frames
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        self.hog_params = {
            "orientations": orientations,
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
            "block_norm": "L2-Hys",
            "visualize": True,
            "transform_sqrt": True,
        }

    def extract_features(self, frame: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog_features, hog_image = hog(gray_frame, **self.hog_params)
        hog_image = (hog_image * 255).astype(np.uint8)
        return hog_features, hog_image

    def visualize_hog(self, frame: "np.ndarray", hog_image: "np.ndarray") -> None:
        # Visualizza l'immagine HOG
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(frame, cmap="gray")
        plt.title("Frame Originale")
        plt.subplot(122)
        plt.imshow(hog_image, cmap="gray")
        plt.title("HOG Features")
        plt.show()

    def process_frames(self) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
        processed_frames = []
        processed_features = []
        for frame in self.frames:
            hog_features, hog_image = self.extract_features(frame)
            # self.visualize_hog(frame, hog_image)
            processed_frames.append(hog_image)
            processed_features.append(hog_features)

        return processed_features, processed_frames
