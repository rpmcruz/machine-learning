from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np


# implementation based on (Hsieh et al, 2008)

class DualLinearSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C, norm=1, fit_intercept=True, max_iter=1000):
        assert norm in [1, 2]
        self.C = C
        self.norm = norm
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        y[y == 0] = -1  # convert to -1, +1
        assert np.unique(y).tolist() == [-1, 1]

        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        if self.norm == 2:
            U = 1e5
            Dii = 0.5 / self.C
        else:
            U = self.C
            Dii = 0

        Qii = np.sum(X**2, 1) + Dii

        a = np.zeros(X.shape[0])
        self.w = np.zeros(X.shape[1])
        tol = 1e-3

        for k in range(self.max_iter):
            err = 0
            for i in range(len(X)):
                G = y[i]*np.sum(self.w*X[i]) - 1 + Dii*a[i]  # linear
                if a[i] < tol:
                    G = min(G, 0)
                elif a[i] > U:
                    G = max(G, 0)

                if np.abs(G) > tol:
                    _a = min(max(a[i]-G/Qii[i], 0), U)
                    self.w += (_a-a[i])*y[i]*X[i]
                    a[i] = _a
                err = max(err, np.abs(G))
            if err < tol:
                break
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        return (np.sum(X*self.w, 1) >= 0).astype(int)
