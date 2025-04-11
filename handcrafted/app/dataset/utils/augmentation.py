"""Module for data augmentation."""

import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np


class DataAugmentation:
    """Class for data augmentation using imgaug."""

    def __init__(
        self,
        num_augmentations: int = 3,
        seed: int = 42,
    ):
        """Initialize the DataAugmentation object.

        Parameters
        ----------
        num_augmentations : int, optional
            The number of augmentations to apply, by default 3.
        seed : int, optional
            The random seed for reproducibility, by default 42.

        """
        ia.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.num_augmentations = num_augmentations
        self.augmenters = [
            iaa.Rotate((-45, 45)),  # Random rotation
            iaa.Affine(
                translate_percent={"x": (-0.3, 0.3), "y": (0, 0.3)}
            ),  # Translation
            iaa.AddToBrightness((-40, 40)),  # Brightness adjustment
            iaa.Fliplr(0.5),  # Horizontal flip
            iaa.GaussianBlur(sigma=(0.0, 3.0)),  # Gaussian blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Gaussian noise
            iaa.Multiply((0.5, 1.5)),  # Brightness adjustment
            iaa.ElasticTransformation(
                alpha=(0, 5.0)
            ),  # Elastic transformation
            iaa.Affine(shear=(-20, 20)),  # Shearing
            iaa.Affine(scale=(0.8, 1.2)),  # Zoom in/out
        ]

    @staticmethod
    def _apply_augmentation(augmenter, frame: np.ndarray) -> np.ndarray:
        """Apply the augmentation to the frame.

        Parameters
        ----------
        augmenter : imgaug.augmenters
            The augmenter to apply.
        frame : np.ndarray
            The frame to augment.

        Returns
        -------
        np.ndarray
            The augmented frame.

        """
        return augmenter(images=[frame])[0]

    def augment_image(
        self, frame: np.ndarray, label: int
    ) -> list[tuple[np.ndarray, int]]:
        """Apply augmentations to the image.

        Parameters
        ----------
        frame : np.ndarray
            The frame to augment.
        label : int
            The label of the frame.

        Returns
        -------
        list[tuple[np.ndarray, int]]
            A list of tuples containing the augmented frame and its label.

        """
        augmentations = []
        for _ in range(self.num_augmentations):
            augmenter = random.choice(self.augmenters)
            augmented_frame = self._apply_augmentation(augmenter, frame)
            augmentations.append((augmented_frame, label))
        return augmentations
