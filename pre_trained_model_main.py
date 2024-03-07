import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard

from handcrafted.app.dataset.dataset import Dataset
from pre_trained_model.dataset_utils.dataset_splitter import DatasetSplitter
from pre_trained_model.model.PreTrainedModel import PreTrainedModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def plot_accuracy_loss(history):
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['categorical_accuracy'])
    axs[1].plot(history.history['val_categorical_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    plt.show()


def plot_confusion_matrix(y_test, y_pred, glosses):
    cfm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=glosses)
    df_cfm = pd.DataFrame(cfm, index=glosses, columns=glosses)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("cfm.png")


def plot_labels_distribution(y_train):
    labels = pd.DataFrame(y_train.copy())
    num_labels = labels.value_counts()
    print(num_labels)

    plt.figure(figsize=(8, 8))
    plt.pie(num_labels, labels=num_labels.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribuzione delle label in X_train')
    plt.show()


if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")

    num_glosses = 10
    glosses = dataset.glosses[5:num_glosses]  # 1:4

    splitter = DatasetSplitter(dataset, glosses, shuffle=True, info=True)

    # plot_labels_distribution(splitter.y_train)

    test_videos_per_gloss = {gloss: 0 for gloss in glosses}
    for video in splitter.dataset.videos:
        if video.gloss in glosses and video.split == "test":
            test_videos_per_gloss[video.gloss] += 1
    print(glosses)
    # Stampa del risultato
    print("Numero di video di test per ogni parola del glossario:")
    for gloss, num_videos in test_videos_per_gloss.items():
        print(f"{gloss}: {num_videos} video di test")

    model, history = PreTrainedModel(splitter.X_train,
                                     splitter.y_train,
                                     splitter.X_val,
                                     splitter.y_val, input_shape=(224, 224, 3), num_classes=5, info=True).load_model()
    # plot_model(model, show_shapes=True, show_layer_names=True)

    # history = model.fit(splitter.X_train, splitter.y_train,
    #                     batch_size=64,
    #                     epochs=5,
    #                     validation_data=(splitter.X_val, splitter.y_val))
    print(history.history)
    plot_accuracy_loss(history)

    print(model.evaluate(splitter.X_test, splitter.y_test))

    y_pred_prob = model.predict(splitter.X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test = np.argmax(splitter.y_test, axis=1)
    print(y_test)
    print(y_pred)

    correct_predictions = np.sum(y_pred == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(y_test, y_pred, glosses)
