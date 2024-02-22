import os

import keras
import tensorflow as tf
from keras import layers, losses
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten
from keras.callbacks import TensorBoard
from keras.src.layers import Conv2D


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
        base_model = keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=self.input_shape)
        # for layer in base_model.layers:
        #     layer.trainable = False
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(self.num_classes, activation='relu')(x)
        predictions = layers.Dense(10, activation='softmax')(x)
        head_model = Model(inputs=base_model.input, outputs=predictions)
        head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        # efficientnet_model = Sequential()
        #
        # model = keras.applications.EfficientNetB7(include_top=False,
        #                                           input_shape=self.input_shape,
        #                                           pooling="avg",
        #                                           classes=self.num_classes,
        #                                           weights="imagenet")
        #
        # # for layer in model.layers:
        # #     layer.trainable = False
        #
        # efficientnet_model.add(model)
        # efficientnet_model.add(Flatten())
        # efficientnet_model.add(Dense(512, activation="relu"))
        # efficientnet_model.add(Dense(self.num_classes, activation="softmax"))
        # efficientnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # return efficientnet_model

        # cnn_model = Sequential()
        # cnn_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape))
        # cnn_model.add(Flatten())
        #
        # # Estrai le features
        # X_train_features = cnn_model.predict(self.X_train)
        # X_val_features = cnn_model.predict(self.X_val)
        #
        # # Reshape per il layer LSTM
        # X_train_lstm = X_train_features.reshape(self.X_train.shape[0], -1, X_train_features.shape[-1])
        # X_val_lstm = X_val_features.reshape(self.X_val.shape[0], -1, X_val_features.shape[-1])
        #
        # model = Sequential()
        # model.add(LSTM(64, activation='relu', return_sequences=True,
        #                input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        # model.add(LSTM(128, activation='relu', return_sequences=True))
        # model.add(LSTM(64, activation='relu', return_sequences=False))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(self.num_classes, activation='softmax'))
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        #
        history = head_model.fit(self.X_train, self.y_train,
                                 batch_size=128,
                                 epochs=500,
                                 validation_data=(self.X_val, self.y_val))
        return head_model, history

    def draw(self):
        pass
