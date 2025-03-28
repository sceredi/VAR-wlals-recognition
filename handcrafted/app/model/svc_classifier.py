from sklearn.svm import SVC


class SVCClassifier:
    def __init__(self, random_state: int = 42) -> None:
        self.svc = SVC(random_state=random_state, verbose=True)

    def train(self, X_train, y_train):
        """Train the SVC model with training data."""
        print("Training SVM...")
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {len(y_train)}")

        # Fit the model
        self.svc.fit(X_train, y_train)
        print("SVM model trained.")

    def predict(self, X_test, y_test):
        """Make predictions using the trained SVC model."""
        print("Testing SVM...")
        y_pred = self.svc.predict(X_test)
        return y_pred

        # # Plot support vectors for each class
        # self.plot_support_vectors()
        #
        # # Confusion matrix visualization
        # self.plot_confusion_matrix(Y_test, Y_pred)
        #
        # # Plot classification results
        # self.plot_classification_results(X_test, Y_test, Y_pred)

    # def plot_support_vectors(self):
    #     """Plot support vectors for each class."""
    #     support_vectors = self.svc.support_vectors_
    #     n_classes = len(self.glosses)
    #
    #     plt.figure(figsize=(8, n_classes * 3))
    #     for i, cl in enumerate(self.glosses):
    #         plt.subplot(n_classes, 1, i + 1)
    #         plt.title(f"Support Vectors for class {cl}")
    #         plt.scatter(range(len(support_vectors[i])), support_vectors[i].ravel(), label=f"Class {cl}")
    #         plt.legend()
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    # def plot_confusion_matrix(self, Y_test, Y_pred):
    #     """Plot confusion matrix."""
    #     cfm = confusion_matrix(Y_test, Y_pred)
    #     df_cfm = pd.DataFrame(cfm, index=self.glosses, columns=self.glosses)
    #
    #     plt.figure(figsize=(10, 7))
    #     cfm_plot = sns.heatmap(df_cfm, annot=True, cmap="Blues", fmt="d")
    #     cfm_plot.figure.savefig("cfm.png")
    #     plt.show()
    #
    # def plot_classification_results(self, X_test, Y_test, Y_pred):
    #     """Plot the classification results."""
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(X_test[:, 0], Y_test, c=Y_pred, cmap="viridis", edgecolors="k")
    #     plt.title("Classification Results using SVC with RBF Kernel")
    #     plt.xlabel("DTW Distance")
    #     plt.ylabel("True Class")
    #     plt.colorbar(label="Predicted Class")
    #     plt.show()
