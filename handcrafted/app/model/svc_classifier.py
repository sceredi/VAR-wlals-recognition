"""Module to perform the SVC classification."""

from sklearn.svm import SVC


class SVCClassifier:
    """Class to perform SVC classification."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the SVCClassifier object.

        Parameters
        ----------
        random_state : int, optional
            The random state for reproducibility, by default 42.

        """
        self.svc = SVC(random_state=random_state, verbose=True)

    def train(self, X_train, y_train):
        """Train the SVC model.

        Parameters
        ----------
        X_train : np.ndarray
            The training data.
        y_train : np.ndarray
            The training labels.

        """
        print("Training SVM...")
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {len(y_train)}")

        # Fit the model
        self.svc.fit(X_train, y_train)
        print("SVM model trained.")

    def predict(self, X_test, y_test):
        """Predict the labels for the test data.

        Parameters
        ----------
        X_test : np.ndarray
            The test data.
        y_test : np.ndarray
            The test labels.

        Returns
        -------
        np.ndarray
            The predicted labels.

        """
        print("Testing SVM...")
        y_pred = self.svc.predict(X_test)
        return y_pred
