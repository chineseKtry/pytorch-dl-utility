import numpy as np
import scipy
import sklearn

def accuracy(y, y_pred):
    return np.mean(y == np.round(y_pred))


def auroc(y, y_pred):
    return sklearn.metrics.roc_auc_score(y, y_pred)


def mse(y, y_pred):
    return sklearn.metrics.mean_squared_error(y, y_pred).astype(np.float64)


def pearson(y, y_pred):
    return scipy.stats.stats.pearsonr(y, y_pred)[0].astype(np.float64)
