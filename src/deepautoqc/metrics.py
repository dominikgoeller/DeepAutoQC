import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def confusion_matrix(actual, predicted):
    assert len(actual) == len(predicted)
    confusion_matrix = metrics.confusion_matrix(y_true=actual, y_pred=predicted)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[False, True]
    )
    cm_display.plot()
    plt.show()
