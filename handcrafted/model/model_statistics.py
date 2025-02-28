import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


class ModelStatistics:
    def __init__(self, save_name: str, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir
        self._save_name = save_name

    def plot_confusion_matrix(self, y_test, y_pred, save : bool = True):
        """Plot confusion matrix."""
        cfm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test))
        df_cfm = pd.DataFrame(cfm, index=labels, columns=labels)

        plt.figure(figsize=(10, 7))
        cfm_plot = sns.heatmap(df_cfm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        if save:
            path = os.path.join(self._save_dir, f"{self._save_name}.png")
            plt.savefig(path)
            print(f"Confusion matrix saved at {path}")

        plt.show()

    @staticmethod
    def print_classification_report(y_test, y_pred):
        """Print classification report."""
        report = classification_report(y_test, y_pred, zero_division=1)
        print("Classification Report:\n", report)

    @staticmethod
    def print_accuracy(y_test, y_pred):
        """Print accuracy of the model."""
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
