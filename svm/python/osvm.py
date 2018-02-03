from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
import collections
import numpy as np

def LinearSVM():
    return LinearSVC(fit_intercept=False, penalty='l1', tol=1e-3, dual=False)


class oSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, h=1, s=1):
        if not estimator:
            estimator = LinearSVM()
        self.estimator = estimator
        self.h = h
        self.s = s

    def fit(self, X, y):
        #y += 1  # this code assumes [1,K]

        self.K = np.max(y)+1
        Xr, yr = self.replicate_data(X, y)
        self.estimator.fit(Xr, yr)

        self.coef = np.array(self.estimator.coef_[0][: len(X[0])])
        self.b = -self.estimator.coef_[0][len(X[0]):] * self.h
        return self

    def replicate_data(self, Xaux, y):
        def u(v, K):
            ret = np.zeros((1, K - 1))
            ret[0][v - 1] = self.h
            return ret

        self.means = np.mean(Xaux, 0)
        self.stds = np.std(Xaux, 0)
        X = (Xaux - self.means) / (self.stds+1e-12)

        s = self.s
        K = self.K

        # First class (and first section)
        X1 = X[y == 0]
        n1 = X1.shape[0]

        Xleft = X[np.logical_and(1 <= y, y <= min(K, 1 + s)-1)]
        nleft = Xleft.shape[0]

        X1 = np.vstack((X1, Xleft))
        X1 = np.hstack((X1, np.repeat(u(1, K), n1 + nleft, axis=0)))

        # Last class (and last section)
        XK = X[y == K-1]
        nK = XK.shape[0]

        Xright = X[np.logical_and(max(1, K - 1 - s + 1)-1 <= y, y <= K-2)]
        nright = Xright.shape[0]

        XK = np.vstack((Xright, XK))
        XK = np.hstack((XK, np.repeat(u(K - 1, K), nK + nright, axis=0)))

        Xret = np.vstack((X1, XK))
        yret = [(0, n1), (1, nleft), (0, nright), (1, nK)]

        # Inner classes
        for q in range(2, K):
            Xil = X[np.logical_and(max(1, q - s + 1)-1 <= y, y <= q-1)]
            nil = Xil.shape[0]

            Xir = X[np.logical_and(q <= y, y <= min(K, q + s)-1)]
            nir = Xir.shape[0]

            Xinner = np.vstack((Xil, Xir))
            Xinner = np.hstack((Xinner,
                                np.repeat(u(q, K), nil + nir, axis=0)))

            Xret = np.vstack((Xret, Xinner))
            yret += [(0, nil), (1, nir)]

        yret = np.hstack(np.zeros(n) if z == 0 else np.ones(n)
                         for z, n in yret)
        yret = yret.astype(np.int)
        return Xret, yret

    def decision_function(self, X):
        Xaux = (X - self.means) / (self.stds+1e-12)
        return np.dot(Xaux, self.coef)

    def predict(self, X):
        scores = self.decision_function(X)
        return np.sum(scores[:, np.newaxis] >= self.b, 1)
