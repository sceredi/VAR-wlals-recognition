import tensorflow as tf


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
