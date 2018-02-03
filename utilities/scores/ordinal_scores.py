import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr


def mean_absolute_error_per_class(y, yp):
    klasses = np.unique(y)
    return [mean_absolute_error(y[y == k], yp[y == k]) for k in klasses]


def maximum_mean_absolute_error(y, yp):
    return np.max(mean_absolute_error_per_class(y, yp))


def average_mean_absolute_error(y, yp):
    return np.mean(mean_absolute_error_per_class(y, yp))


def accuracy_for_class(k):
    def _accuracy(y, yp):
        return np.sum(np.logical_and(y == k, yp == k))
    return _accuracy


def spearman_rho(y, yp):
    return spearmanr(y, yp)[0]
