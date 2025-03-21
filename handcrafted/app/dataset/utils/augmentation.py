import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

from handcrafted.app.dataset.signer_frames import SignerFrame


class DataAugmentation:
    def __init__(
        self,
        signer_frames: list[SignerFrame],
        num_augmentations: int = 3,
        seed: int = 42,
    ):
        ia.seed(seed)
        random.seed(seed)
        self.signer_frames = signer_frames
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

    def random_augmentations(self):
        augmented_signer_frames = self.signer_frames.copy()
        for signer_frame in self.signer_frames:
            for _ in range(self.num_augmentations):
                augmenter = random.choice(self.augmenters)
                augmented_frame = self._apply_augmentation(
                    augmenter, signer_frame.frame
                )
                augmented_signer_frames.append(
                    SignerFrame(
                        frame=augmented_frame, signer_id=signer_frame.signer_id
                    )
                )
        random.shuffle(augmented_signer_frames)
        return augmented_signer_frames
