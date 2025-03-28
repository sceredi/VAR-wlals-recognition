import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np


class DataAugmentation:
    def __init__(
        self,
        num_augmentations: int = 3,
        seed: int = 42,
    ):
        ia.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.num_augmentations = num_augmentations
        self.augmenters = [
            iaa.Rotate((-45, 45)),  # Rotazione casuale
            iaa.Affine(
                translate_percent={"x": (-0.3, 0.3), "y": (0, 0.3)}
            ),  # Traslazione
            iaa.AddToBrightness((-40, 40)),  # Modifica luminosità
            iaa.Fliplr(0.5),  # Flipping orizzontale con probabilità 50%
            iaa.GaussianBlur(sigma=(0.0, 3.0)),  # Sfocatura gaussiana
            iaa.AdditiveGaussianNoise(
                scale=(0, 0.05 * 255)
            ),  # Rumore gaussiano
            iaa.Multiply((0.5, 1.5)),  # Aumenta/diminuisce contrasto
            iaa.ElasticTransformation(alpha=(0, 5.0)),  # Deformazione elastica
            iaa.Affine(shear=(-20, 20)),  # Shearing
            iaa.Affine(scale=(0.8, 1.2)),  # Zoom in/out
        ]

    @staticmethod
    def _apply_augmentation(augmenter, frame: np.ndarray) -> np.ndarray:
        return augmenter(images=[frame])[0]

    def augment_image(
        self, frame: np.ndarray, label: int
    ) -> list[tuple[np.ndarray, int]]:
        augmentations = []
        for _ in range(self.num_augmentations):
            augmenter = random.choice(self.augmenters)
            augmented_frame = self._apply_augmentation(augmenter, frame)
            augmentations.append((augmented_frame, label))
        return augmentations
