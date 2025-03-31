import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
from tqdm import tqdm

from handcrafted.app.dataset.dataset_loader import Frame
from handcrafted.app.dataset.utils.augmentation import DataAugmentation
from handcrafted.app.features.extractor.color_histogram_extractor import (
    ColorHistogram,
)
from handcrafted.app.features.extractor.hog_extractor import HOGExtractor
from handcrafted.app.features.extractor.lbp_extractor import LBPExtractor


class DatasetCreator:
    def __init__(self, seed: int = 42):
        tf.random.set_seed(seed)
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal"
                ),  # Flip images horizontally
                tf.keras.layers.RandomRotation(
                    0.2
                ),  # Rotate images by up to 20%
                tf.keras.layers.RandomZoom(0.2),  # Zoom images by up to 20%
                tf.keras.layers.RandomContrast(0.2),  # Change contrast
                tf.keras.layers.RandomBrightness(0.2),  # Change brightness
            ]
        )

    # TODO: delete
    @staticmethod
    def _extract_features(frame: np.ndarray) -> np.ndarray:
        hog_features, _ = HOGExtractor([frame]).process_frames()
        lbp_features = LBPExtractor([frame]).get_lbp_features()
        color_hist_features = ColorHistogram([frame], n_bins=8).process_frames(
            cv2.COLOR_BGR2HSV, separate_colors=False, normalize=True
        )
        hog_features = np.reshape(hog_features, -1)
        lbp_features = np.reshape(lbp_features, -1)
        color_hist_features = np.reshape(color_hist_features, -1)
        return np.concatenate(
            (hog_features, lbp_features, color_hist_features)
        )

    @staticmethod
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        return image

    def load_and_preprocess_image_with_label(self, image_path, label, num_aug):
        image = self.load_image(image_path)
        image = (image * 2) - 1

        if num_aug > 0:
            augmented_images = [
                self.data_augmentation(image) for _ in range(num_aug)
            ]
            return tf.stack(augmented_images), tf.stack([label] * num_aug)
        return image, label

    def create_dataset(
        self,
        file_paths,
        labels,
        num_aug: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if num_aug > 0:
            dataset = dataset.flat_map(
                lambda image_path, label: tf.data.Dataset.from_tensors(
                    self.load_and_preprocess_image_with_label(
                        image_path, label, num_aug
                    )
                ).unbatch()
            )
        else:
            dataset = dataset.map(
                lambda image_path, label: self.load_and_preprocess_image_with_label(
                    image_path, label, 0
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    # TODO: delete
    def load_and_preprocess_features_with_label(
        self, image_path, label, num_aug
    ):
        def _process(image_path):
            image = self.load_image(image_path)
            image = tf.cast(image * 255, tf.uint8)
            image_np = image.numpy()
            features = self._extract_features(image_np)
            return features

        features = tf.py_function(
            func=_process, inp=[image_path], Tout=tf.float32
        )

        if num_aug > 0:
            augmented_features = [
                tf.py_function(
                    func=_process, inp=[image_path], Tout=tf.float32
                )
                for _ in range(num_aug)
            ]
            return tf.stack(augmented_features), tf.stack([label] * num_aug)

        return features, label

    @staticmethod
    def create_custom_dataset(
            frames: list[Frame],
            labels: np.ndarray,
            batch_size: int = 32,
            seed: int = 42,
    ):
        batches = []
        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i: i + batch_size]
            batch_labels = labels[i: i + batch_size]
            batches.append(MiniBatch(batch_frames, batch_labels, seed=seed))
        return batches


class MiniBatch:
    def __init__(
        self, frames: list[Frame], labels: np.ndarray, seed: int = 42
    ):
        self.frames = frames
        self.labels = labels
        self.seed = seed

    @staticmethod
    def _extract_features(frame: np.ndarray) -> np.ndarray:
        hog_features, _ = HOGExtractor([frame]).process_frames()
        lbp_features = LBPExtractor([frame]).get_lbp_features()
        color_hist_features = ColorHistogram([frame], n_bins=8).process_frames(
            cv2.COLOR_BGR2HSV, separate_colors=False, normalize=True
        )
        hog_features = np.reshape(hog_features, -1)
        lbp_features = np.reshape(lbp_features, -1)
        color_hist_features = np.reshape(color_hist_features, -1)
        return np.concatenate(
            (hog_features, lbp_features, color_hist_features)
        )

    def load(self, num_aug: int = 0, shuffle: bool = True):
        features = []
        lbl = []
        augmenter = DataAugmentation(num_augmentations=num_aug)
        for full_frame, label in zip(self.frames, self.labels):
            frame = full_frame.load_frame()
            augs = augmenter.augment_image(frame, label)
            augs = augs + [(frame, label)]
            for f, _ in augs:
                features.append(self._extract_features(f))
                lbl.append(label)
        features = np.array(features)
        lbl = np.array(lbl)
        if shuffle:
            features, lbl = sk_shuffle(features, lbl, random_state=self.seed)
        return features, lbl
