import sys

from sklearn.metrics import accuracy_score

from handcrafted.app.dataset.dataset import Dataset
from handcrafted.app.model.dtw_classifier import DTWClassifier
from handcrafted.app.model.model_statistics import ModelStatistics

if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    # glosses = dataset.glosses[1:3]

    nwords = 10
    if len(sys.argv) > 1:
        nwords = int(sys.argv[1])

    output_file = f"trad_{nwords}_class_confusion_matrx"

    dtw_classifier = DTWClassifier(dataset, dataset.glosses)
    dtw_classifier.train_test_videos(num_glosses=nwords)
    X_test, y_train, y_test = dtw_classifier.compute_dtw_similarity_matrix()

    y_pred = dtw_classifier.dtw_predict(X_test, y_train)
    accuracy_dtw = accuracy_score(y_test, y_pred)
    print(f"Accuracy DTW: {accuracy_dtw * 100:.2f}%")

    y_test_labels = [dataset.glosses[y] for y in y_test]
    y_pred_labels = [dataset.glosses[y] for y in y_pred]

    print(y_test_labels)
    print(y_pred_labels)

    model_stats = ModelStatistics(save_name=output_file)
    model_stats.plot_confusion_matrix(y_test_labels, y_pred_labels, plot=False)

    # svc_model = SVClassifier()
    # svc_model.train(X_train, y_train)
    # y_pred = svc_model.predict(X_test, y_test)
    # score = svc_model.svc.score(X_test, y_test)
    # print(f"Correct classification rate SVC: {score * 100:.2f}%")
