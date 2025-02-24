from sklearn.metrics import accuracy_score

from app.dataset.dataset import Dataset
from model.DTWClassifier import DTWClassifier

if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    glosses = dataset.glosses[1:3]

    dtw_classifier = DTWClassifier(dataset, glosses)
    dtw_classifier.train_test_set_videos()
    X_train, X_test, y_train, y_test = dtw_classifier.compute_dtw_similarity_matrix()

    y_pred = dtw_classifier.dtw_predict(X_test, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")