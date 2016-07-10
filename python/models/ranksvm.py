# -*- coding: utf-8 -*-

# Ranking models.

from sklearn.base import BaseEstimator, ClassifierMixin
from extras import MyLinearSVC
from sklearn.linear_model import LogisticRegression
from utils import choose_threshold
import numpy as np


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
    def __init__(self, C=1):
        super(RankSVM, self).__init__()
        self.C = C
        self.coefs = []
        self.classes_ = (0, 1)

    def fit(self, X, y):
        # we only instantiate it here, so that sklearn can do its
        # GridSearch validation magic with get_ and set_params()
        self.estimator = MyLinearSVC(C=self.C, fit_intercept=False)
        # could also be:
        #self.estimator = LogisticRegression(C=self.C, fit_intercept=False)

        dX, dy = preprocess(X, y)
        self.estimator.fit(dX, dy)
        self.coefs = self.estimator.coef_[0]

        H = self.predict_proba(X)
        self.th = choose_threshold(H, y)
        return self

    def predict_proba(self, X):
        # these are not probabilities, but I overloaded this function because
        # it makes it nicer to have a common interface for the ROC curves
        return np.sum(self.coefs * X, axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= self.th).astype(int)

if __name__ == '__main__':
    import test
    names = []
    models = []
    for C in 2.**np.arange(-3, 11):
        names.append('RankSVM %s' % str(C))
        models.append(RankSVM(C))
    test.test(names, models)
