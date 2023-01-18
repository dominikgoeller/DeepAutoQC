import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def confusion_matrix(actual, predicted):
    assert len(actual) == len(predicted)
    confusion_matrix = metrics.confusion_matrix(y_true=actual, y_pred=predicted)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=["good", "bad"]
    )
    # tp = np.diag(confusion_matrix)
    precision = np.diag(confusion_matrix) / np.sum(
        confusion_matrix, axis=0
    )  # precision per class
    recall = np.diag(confusion_matrix) / np.sum(
        confusion_matrix, axis=1
    )  # recall per class
    print("Precision", precision)
    print("Recall", recall)
    print("Mean precision", np.mean(precision))
    print("Mean recall", np.mean(recall))
    cm_display.plot()
    plt.show()
    # return cm_display


def plot_losses():
    return None


def main():
    actual = [0, 1, 0, 1, 1]
    pred = [1, 0, 1, 0, 1]
    confusion_matrix(actual=actual, predicted=pred)


if __name__ == "__main__":
    main()
