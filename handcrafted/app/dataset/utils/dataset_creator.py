import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from handcrafted.app.features.extractor.color_histogram_extractor import ColorHistogram
from handcrafted.app.features.extractor.hog_extractor import HOGExtractor
from handcrafted.app.features.extractor.lbp_extractor import LBPExtractor


class DatasetCreator:
    def __init__(self, seed: int = 42):
        tf.random.set_seed(seed)
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),  # Flip images horizontally
            tf.keras.layers.RandomRotation(0.2),  # Rotate images by up to 20%
            tf.keras.layers.RandomZoom(0.2),  # Zoom images by up to 20%
            tf.keras.layers.RandomContrast(0.2),  # Change contrast
            tf.keras.layers.RandomBrightness(0.2)  # Change brightness
        ])

    # def _extract_features(self, frame: np.ndarray) -> np.ndarray:
    #     hog_features, _ = HOGExtractor([frame]).process_frames()
    #     lbp_features = LBPExtractor([frame]).get_lbp_features()
    #     color_hist_features = ColorHistogram([frame]).process_frames(
    #         cv2.COLOR_BGR2HSV, separate_colors=False, normalize=True
    #     )
    #     hog_features = np.reshape(hog_features, -1)
    #     lbp_features = np.reshape(lbp_features, -1)
    #     color_hist_features = np.reshape(color_hist_features, -1)
    #     return np.concatenate(
    #         (hog_features, lbp_features, color_hist_features)
    #     )

    def load_and_preprocess_image_with_label(self, image_path, label, num_aug):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = (image * 2) - 1

        if num_aug > 0:
            augmented_images = [self.data_augmentation(image) for _ in range(num_aug)]
            return tf.stack(augmented_images), tf.stack([label] * num_aug)
        return image, label

    # def load_and_preprocess_features_with_label(self, image_path, label, num_aug):
    #     image = tf.io.read_file(image_path)
    #     image = tf.image.decode_png(image, channels=3)
    #     image = tf.image.resize(image, (224, 224))
    #     image = tf.cast(image * 255, tf.uint8)
    #
    #     image_np = image.numpy()
    #     features = self._extract_features(image_np)
    #     features = tf.convert_to_tensor(features, dtype=tf.float32)
    #
    #     if num_aug > 0:
    #         augmented_features = [self._extract_features() for _ in range(num_aug)]
    #         return tf.stack(augmented_features), tf.stack([label] * num_aug)
    #
    #     return features, label

    def create_dataset(self, file_paths, labels, num_aug: int = 0, batch_size: int = 32, shuffle: bool = True):
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if num_aug > 0:
            dataset = dataset.flat_map(
                lambda image_path, label: tf.data.Dataset.from_tensors(
                    self.load_and_preprocess_image_with_label(image_path, label, num_aug)).unbatch())
        else:
            dataset = dataset.map(
                lambda image_path, label: self.load_and_preprocess_image_with_label(image_path, label, num_aug),
                num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        hog_features, _ = HOGExtractor([frame]).process_frames()
        lbp_features = LBPExtractor([frame]).get_lbp_features()
        color_hist_features = ColorHistogram([frame]).process_frames(
            cv2.COLOR_BGR2HSV, separate_colors=False, normalize=True
        )
        return np.concatenate([
            np.ravel(hog_features), np.ravel(lbp_features), np.ravel(color_hist_features)
        ]).astype(np.float32)

    def load_and_preprocess_features_with_label(self, image_path, label, num_aug):
        def _process(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, (224, 224))
            image = tf.cast(image * 255, tf.uint8)

            image_np = image.numpy()
            features = self._extract_features(image_np)
            return features

        features = tf.py_function(func=_process, inp=[image_path], Tout=tf.float32)

        if num_aug > 0:
            augmented_features = [tf.py_function(func=_process, inp=[image_path], Tout=tf.float32) for _ in
                                  range(num_aug)]
            return tf.stack(augmented_features), tf.stack([label] * num_aug)

        return features, label

    # OPZIONE 1
    def create_dataset_with_features(self, file_paths, labels, num_aug: int = 0, batch_size: int = 32, shuffle: bool = True):
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if num_aug > 0:
            dataset = dataset.flat_map(
                lambda image_path, label: tf.data.Dataset.from_tensors(
                    self.load_and_preprocess_features_with_label(image_path, label, num_aug)).unbatch())
        else:
            dataset = dataset.map(
                lambda image_path, label: self.load_and_preprocess_features_with_label(image_path, label, num_aug),
                num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    # OPZIONE 2
    # def create_dataset_with_features(self, file_paths, labels, num_aug: int = 0):
    #     features = []
    #
    #     for image_path, label in tqdm(zip(file_paths, labels)):
    #         if num_aug > 0:
    #             augmented_features = self.load_and_preprocess_features_with_label(image_path, label, num_aug)
    #             features.extend(augmented_features)
    #             labels.extend([label] * num_aug)
    #         else:
    #             feature = self.load_and_preprocess_features_with_label(image_path, label, num_aug)
    #             features.append(feature)
    #             labels.append(label)
    #
    #     features = np.array(features)
    #     labels = np.array(labels)
    #
    #     return features, labels

    # OPZIONE 3
    # def create_dataset_with_features(self, file_paths, labels, num_aug: int = 0, batch_size: int = 32):
    #     def data_generator():
    #         batch_features = []
    #         batch_labels = []
    #
    #         for image_path, label in zip(file_paths, labels):
    #             if num_aug > 0:
    #                 # Apply data augmentation here if num_aug > 0
    #                 augmented_features = self.load_and_preprocess_features_with_label(image_path, label, num_aug)
    #                 # Assuming load_and_preprocess returns a list of augmented features
    #                 for aug_feature in augmented_features:
    #                     batch_features.append(aug_feature)
    #                     batch_labels.append(label)
    #             else:
    #                 # No augmentation, just process the original feature
    #                 feature = self.load_and_preprocess_features_with_label(image_path, label, num_aug)
    #                 batch_features.append(feature)
    #                 batch_labels.append(label)
    #
    #             # Yield a batch of features and labels when the batch is full
    #             if len(batch_features) >= batch_size:
    #                 yield np.array(batch_features), np.array(batch_labels)
    #                 batch_features = []  # Reset the batch
    #                 batch_labels = []  # Reset the labels
    #
    #         # Yield any remaining data after the loop ends
    #         if batch_features:
    #             yield np.array(batch_features), np.array(batch_labels)
    #
    #     return data_generator

