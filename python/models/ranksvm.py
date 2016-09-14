# -*- coding: utf-8 -*-

# Ranking models: this is for the binary care.

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def choose_threshold(s, y):
    si = np.argsort(s)
    s = s[si]
    y = y[si]

    maxF1 = -np.inf
    bestTh = 0
    for i in range(1, len(y)):
        if y[i] != y[i-1]:
            TP = np.sum(y[i:] == 1)
            FP = np.sum(y[i:] == 0)
            FN = np.sum(y[:i] == 1)
            F1 = (2.*TP)/(2.*TP+FN+FP+1e-10)
            if F1 > maxF1:
                maxF1 = F1
                bestTh = (s[i]+s[i-1])/2.
    return bestTh


def preprocess(X, y):
    # dX,dy have classe1 times classe(0 or -1) rows
    if len(X) > 2000:  # lower space complexity
        import itertools
        len_diff = np.sum(y == 1)*np.sum(y == 0)*2
        dX = np.zeros((len_diff, X.shape[1]))
        dy = np.zeros(len_diff)
        i = 0
        for X1, y1 in itertools.izip(X, y):
            for X2, y2 in itertools.izip(X, y):
                if y1 > y2:
                    dX[i] = X1 - X2
                    dy[i] = 1
                    i += 1
                    dX[i] = X2 - X1
                    dy[i] = 0
                    i += 1
    else:  # lower time complexity
        # emparelhado: this is very fast!
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        X0 = np.repeat(X[y == 0], n1, 0)
        X1 = np.tile(X[y == 1], (n0, 1))
        dX = np.r_[X1-X0, X0-X1]
        dy = np.r_[np.ones(n0*n1), np.zeros(n0*n1)]
    return (dX, dy)


class RankSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        super(RankSVM, self).__init__()
        self.model = model
        if model is None:
            from sklearn.svm import LinearSVC
            self.model = LinearSVC(fit_intercept=False, penalty='l1', tol=1e-3,
                                   dual=False)

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # required by sklearn

        dX, dy = preprocess(X, y)
        self.model.fit(dX, dy)
        self.coefs = self.model.coef_[0]

        H = self.predict_proba(X)
        self.th = choose_threshold(H, y)
        return self

    def predict_proba(self, X):
        # these are not probabilities, but I overloaded this function because
        # it makes it nicer to have a common interface for the ROC curves
        return np.sum(self.coefs * X, axis=1)

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X) >= self.th).astype(int)
