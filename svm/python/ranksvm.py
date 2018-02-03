from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import LinearSVC
import numpy as np


def preprocess(X, y):
    K = len(np.unique(y))
    N = len(X)
    Nk = np.bincount(y)

    X1 = np.repeat(X, N, 0)
    X2 = np.tile(X.T, N).T

    y1 = np.repeat(y, N)
    y2 = np.tile(y, N)

    yy = (y1 > y2).astype(int)

    # remove y1 == y2
    diff = y1 != y2
    X1 = X1[diff]
    X2 = X2[diff]
    yy = yy[diff]

    pairs = K*(K-1)
    ww = len(X1) / (pairs * (Nk[y1[diff]]*Nk[y2[diff]]))
    return X1-X2, yy, ww


def LinearSVM():
    return LinearSVC(fit_intercept=False, penalty='l1', tol=1e-3, dual=False)


class RankSVM(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator
        if estimator is None:
            self.estimator = LinearSVM()

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # required by sklearn

        dX, dy = preprocess(X, y)
        self.model.fit(dX, dy)
        self.coefs = self.model.coef_[0]
        return self

    def predict(self, X):
        return np.sum(self.coefs * X, 1)
