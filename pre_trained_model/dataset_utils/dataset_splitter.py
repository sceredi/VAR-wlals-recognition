import random
from typing import List
import tensorflow as tf

import numpy as np
from keras.src.utils import to_categorical

from handcrafted.app.dataset.dataset import Dataset


class DatasetSplitter:
    def __init__(self, dataset: Dataset, glosses: List[str], shuffle=True, info=False):
        """
        Split the dataset into train, validation and test sets
        :param dataset: Dataset object
        :param glosses: List of glosses
        :param shuffle: if True, shuffle the videos
        :param info: if True, print the number of videos in each set
        """
        self.dataset = dataset
        self.glosses = glosses
        self.shuffle = shuffle
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_train_val_test()
        if info:
            self.print_info()

    def split_train_val_test(self):
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []

        all_videos = list(self.dataset.videos)
        if self.shuffle:
            random.shuffle(all_videos)

        for video in all_videos:
            if video.gloss in self.glosses:
                # frames = np.array(video.get_frames())
                frames = video.get_frames()
                label = self.glosses.index(video.gloss)
                for frame in frames:
                    if video.split == "train":
                        X_train.append(frame)
                        y_train.append(label)
                    elif video.split == "val":
                        X_val.append(frame)
                        y_val.append(label)
                    else:
                        X_test.append(frame)
                        y_test.append(label)

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        # y_train = np.asarray(y_train).astype(np.float32)
        # y_val = np.asarray(y_val).astype(np.float32)
        # y_test = np.asarray(y_test).astype(np.float32)

        num_classes = len(self.glosses)
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # train_indices = np.arange(X_train.shape[0])
        # np.random.shuffle(train_indices)
        # X_train = X_train[train_indices]
        # y_train = y_train[train_indices]
        #
        # val_indices = np.arange(X_val.shape[0])
        # np.random.shuffle(val_indices)
        # X_val = X_val[val_indices]
        # y_val = y_val[val_indices]
        #
        # test_indices = np.arange(X_test.shape[0])
        # np.random.shuffle(test_indices)
        # X_test = X_test[test_indices]
        # y_test = y_test[test_indices]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def print_info(self):
        print("Train")
        print(self.X_train, self.y_train)
        print("Val")
        print(self.X_val, self.y_val)
        print("Test")
        print(self.X_test, self.y_test)
        print("Train X shape: ", self.X_train.shape)
        print("Train y shape: ", self.y_train.shape)
        print("Val X shape: ", self.X_val.shape)
        print("Val y shape: ", self.y_val.shape)
        print("Test X shape: ", self.X_test.shape)
        print("Test y shape: ", self.y_test.shape)
        print("Total videos: ", len(self.X_train) + len(self.X_val) + len(self.X_test))

