import numpy as np
from sklearn.metrics import mean_absolute_error


def mean_absolute_error_per_class(y, yp):
    y = np.asarray(y)
    yp = np.asarray(yp)
    klasses = np.unique(y)
    return [mean_absolute_error(y[y == k], yp[y == k]) for k in klasses]


def maximum_mean_absolute_error(y, yp):
    return np.max(mean_absolute_error_per_class(y, yp))


def average_mean_absolute_error(y, yp):
    return np.mean(mean_absolute_error_per_class(y, yp))
