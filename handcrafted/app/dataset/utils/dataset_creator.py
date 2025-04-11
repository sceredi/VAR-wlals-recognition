"""Module for creating a dataset from images and labels."""

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
    """Class to create a dataset from images and labels."""

    def __init__(self, seed: int = 42):
        """Initialize the DatasetCreator object.

        Parameters
        ----------
        seed : int, optional
            The random seed for reproducibility, by default 42.

        """
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

    @staticmethod
    def load_image(image_path):
        """Load and preprocess an image.

        Parameters
        ----------
        image_path : str
            The path to the image file.

        Returns
        -------
        tf.Tensor
            The preprocessed image tensor.

        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        return image

    def load_and_preprocess_image_with_label(self, image_path, label, num_aug):
        """Load and preprocess an image with its label.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        label : int
            The label of the image.
        num_aug : int
            The number of augmentations to apply.

        Returns
        -------
        tuple
            A tuple containing the augmented images and their corresponding labels.

        """
        image = self.load_image(image_path)
        image = (image * 2) - 1

        def augment_images():
            # Use tf.range for dynamic loops compatible with tf.function
            augmented_images = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True
            )
            i = tf.constant(0, dtype=tf.int32)

            def condition(i, augmented_images):
                return tf.less(i, num_aug)

            def body(i, augmented_images):
                augmented_image = self.data_augmentation(image)
                augmented_images = augmented_images.write(i, augmented_image)
                return tf.add(i, 1), augmented_images

            i, augmented_images = tf.while_loop(
                condition, body, [i, augmented_images]
            )
            augmented_images = augmented_images.write(i, image)
            return augmented_images.stack()

        # Use tf.cond to choose the appropriate branch based on num_aug
        images = tf.cond(
            tf.greater(num_aug, 0), augment_images, lambda: tf.stack([image])
        )

        # Ensure label is a tensor and has the correct shape
        label = tf.convert_to_tensor(label)
        if label.shape.rank == 0:
            label = tf.expand_dims(label, 0)

        # Repeat the label to match the number of images
        labels = tf.tile(tf.expand_dims(label, 0), [tf.shape(images)[0], 1])

        return images, labels

    def create_dataset(
        self,
        file_paths,
        augs,
        labels,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """Create a TensorFlow dataset from file paths and labels.

        Parameters
        ----------
        file_paths : list[str]
            List of file paths to the images.
        augs : list[int]
            List of integers representing the number of augmentations for each image.
        labels : list[int]
            List of labels corresponding to the images.
        batch_size : int, optional
            The batch size for the dataset, by default 32.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset object.

        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (file_paths, augs, labels)
        )
        dataset = dataset.flat_map(
            lambda img, aug, label: tf.data.Dataset.from_tensors(
                self.load_and_preprocess_image_with_label(img, label, aug)
            ).unbatch()
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def create_custom_dataset(
        frames: list[Frame],
        labels: np.ndarray,
        augmentation: list[int],
        batch_size: int = 32,
        seed: int = 42,
    ):
        """Create a custom dataset from frames and labels.

        Parameters
        ----------
        frames : list[Frame]
            List of Frame objects.
        labels : np.ndarray
            Numpy array of labels.
        augmentation : list[int]
            List of integers representing the number of augmentations for each frame.
        batch_size : int, optional
            The batch size for the dataset, by default 32.
        seed : int, optional
            The random seed for reproducibility, by default 42.

        Returns
        -------
        list[MiniBatch]
            A list of MiniBatch objects.

        """
        batches = []
        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            batch_aug = augmentation[i : i + batch_size]
            batches.append(
                MiniBatch(batch_frames, batch_labels, aug=batch_aug, seed=seed)
            )
        return batches


class MiniBatch:
    """Class to create a mini-batch of images and labels."""

    def __init__(
        self,
        frames: list[Frame],
        labels: np.ndarray,
        aug: list[int],
        seed: int = 42,
    ):
        """Initialize the MiniBatch object.

        Parameters
        ----------
        frames : list[Frame]
            List of Frame objects.
        labels : np.ndarray
            Numpy array of labels.
        aug : list[int]
            List of integers representing the number of augmentations for each frame.
        seed : int, optional
            The random seed for reproducibility, by default 42.

        """
        self.frames = frames
        self.labels = labels
        self.aug = aug
        self.seed = seed

    @staticmethod
    def _extract_features(frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to process.

        Returns
        -------
        np.ndarray
            The extracted features.

        """
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

    def load(self, shuffle: bool = True):
        """Load the mini-batch of images and labels.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.

        Returns
        -------
        tuple
            A tuple containing the features and labels.

        """
        features = []
        lbl = []
        for full_frame, label, aug in zip(
            self.frames, self.labels, self.aug, strict=False
        ):
            frame_image = full_frame.load_frame()
            ims = [(frame_image, label)]
            augmenter = DataAugmentation(num_augmentations=aug)
            augs = augmenter.augment_image(frame_image, label)
            ims = ims + augs
            for f, _ in ims:
                features.append(self._extract_features(f))
                lbl.append(label)
        features = np.array(features)
        lbl = np.array(lbl)
        if shuffle:
            features, lbl = sk_shuffle(features, lbl, random_state=self.seed)
        return features, lbl
