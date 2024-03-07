import cv2
import matplotlib.pyplot as plt

from augmentation.Augmentation import DataAugmentation
from handcrafted.app.dataset.dataset import Dataset
from pre_trained_model.dataset_utils.dataset_splitter import DatasetSplitter

if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = dataset.glosses[1:4]
    splitter = DatasetSplitter(dataset, glosses, shuffle=False, info=True)

    # for idx in range(len(splitter.X_train)):
    #     plt.imshow(splitter.X_train[idx])
    #     plt.show()

    augmentation = DataAugmentation(splitter.X_train, splitter.y_train)
    X_train_rotation, y_train_rotation = augmentation.rotation()
    X_train_translation, y_train_translation = augmentation.translation()
    for idx in range(len(X_train_rotation)):
        plt.imshow(X_train_rotation[idx])
        plt.show()

