import os

import keras
import tensorflow as tf
from keras import layers, losses, Input
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten
from keras.callbacks import TensorBoard
from keras.src.applications import VGG16, InceptionV3
from keras.src.layers import Conv2D, concatenate, Reshape, MaxPooling2D, TimeDistributed, Rescaling, \
    GlobalAveragePooling2D


class PreTrainedModel:
    def __init__(self, X_train, y_train, X_val, y_val, input_shape, num_classes, info=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model, self.history = self.load_model()
        if info:
            self.summary = self.model.summary()

    def load_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        log_dir = os.path.join("logs")
        tensorboard_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(self.X_train, self.y_train,
                            batch_size=64,
                            epochs=15,
                            verbose=True,
                            validation_data=(self.X_val, self.y_val))
        # callbacks=[tensorboard_callbacks])
        return model, history

    def draw(self):
        pass
