"""Module to generate model statistics."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


class ModelStatistics:
    """Class to generate model statistics."""

    def __init__(self, save_name: str, save_dir: str = "plots"):
        """Initialize the ModelStatistics class.

        Parameters
        ----------
        save_name : str
            Name of the file to save the plot.
        save_dir : str, optional
            Directory to save the plot (default is "plots").

        """
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir
        self._save_name = save_name

    def plot_confusion_matrix(
        self, y_test, y_pred, save: bool = True, plot: bool = True
    ) -> None:
        """Plot confusion matrix.

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        save : bool, optional
            Whether to save the plot (default is True).
        plot : bool, optional
            Whether to display the plot (default is True).

        """
        cfm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test) | set(y_pred))
        df_cfm = pd.DataFrame(cfm, index=labels, columns=labels)

        plt.figure(figsize=(10, 7))
        _ = sns.heatmap(df_cfm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        if save:
            path = os.path.join(self._save_dir, f"{self._save_name}.png")
            plt.savefig(path)
            print(f"Confusion matrix saved at {path}")

        if plot:
            plt.show()

    @staticmethod
    def print_classification_report(y_test, y_pred) -> None:
        """Print classification report.

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        """
        report = classification_report(y_test, y_pred, zero_division=1)  # type: ignore
        print("Classification Report:\n", report)

    @staticmethod
    def print_accuracy(y_test, y_pred) -> None:
        """Print accuracy of the model.

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        """
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
