from sklearn.metrics import accuracy_score

from handcrafted.app.dataset.dataset import Dataset
from handcrafted.model.dtw_classifier import DTWClassifier
from handcrafted.model.svc_classifier import SVClassifier

if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    # glosses = dataset.glosses[1:3]

    dtw_classifier = DTWClassifier(dataset, dataset.glosses)
    dtw_classifier.train_test_videos()
    X_test, y_train, y_test = dtw_classifier.compute_dtw_similarity_matrix()

    y_pred = dtw_classifier.dtw_predict(X_test, y_train)
    accuracy_dtw = accuracy_score(y_test, y_pred)
    print(f"Accuracy DTW: {accuracy_dtw * 100:.2f}%")

    # svc_model = SVClassifier()
    # svc_model.train(X_train, y_train)
    # y_pred = svc_model.predict(X_test, y_test)
    # score = svc_model.svc.score(X_test, y_test)
    # print(f"Correct classification rate SVC: {score * 100:.2f}%")
