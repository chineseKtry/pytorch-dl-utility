import numpy as np
import sklearn


def accuracy(y, y_pred):
    return np.mean(y == np.round(y_pred))


def auroc(y, y_pred):
    return sklearn.metrics.roc_auc_score(y, y_pred)
