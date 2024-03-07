import imgaug.augmenters as iaa
import imgaug as ia


class DataAugmentation:
    def __init__(self, frames, labels):
        ia.seed(42)
        self.frames = frames
        self.labels = labels

    def augment(self, augmenter):
        augmented_frames = augmenter(images=self.frames)
        return augmented_frames, self.labels

    def rotation(self, angle=-30):
        augmentation = iaa.Rotate(angle)
        return self.augment(augmentation)

    def translation(self, x=0.25, y=0):
        augmentation = iaa.Affine(translate_percent={"x": x, "y": y})
        return self.augment(augmentation)
