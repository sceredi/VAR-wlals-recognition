import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.src.layers import GlobalAveragePooling3D


class C3DModel:
    def __init__(self, input_shape, num_classes, info=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.load_model()
        if info:
            self.summary = self.model.summary()

    def load_model(self):
        model = Sequential()

        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))

        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    # def fit(self, x, y, batch_size, epochs, validation_data):
    #     for layer in self.model.layers:
    #         layer.trainable = False
    #     x = GlobalAveragePooling3D()(self.model.output)
    #     x = Dense(self.num_classes, activation='softmax')(x)
    #     custom_model = Model(inputs=self.model.input, outputs=x)
    #     custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
