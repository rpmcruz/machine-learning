# Songfeng Zheng, "QBoost: Predicting quantiles with boosting for regression
# and binary classification" (2012)

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from scipy.stats import norm


def K(x, h):
    return norm.pdf(x, scale=h)


class ZerosDummyModel:
    def __init__(self, tau):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class QBC(BaseEstimator, ClassifierMixin):
    def __init__(self, tau, M=100, eta=0.1, h=0.1, base_estimator=None):
        self.tau = tau
        self.M = M
        self.eta = eta
        self.h = h
        self.first_estimator = ZerosDummyModel(tau)
        if base_estimator is None:
            base_estimator = DecisionTreeRegressor(max_depth=1)
        self.base_estimator = base_estimator
        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.fs = [self.first_estimator]
        # step 0
        f = self.first_estimator.fit(X, y)
        # step 1
        for m in range(self.M):
            f = self.predict_proba(X)
            # step 2
            U = (y-(1-self.tau))*K(f, self.h)
            # step 3
            g = clone(self.base_estimator).fit(X, U)
            # step 4
            self.fs.append(g)
        return self

    def predict_proba(self, X):
        f0 = self.fs[0].predict(X)
        r = np.sum([self.eta * f.predict(X) for f in self.fs[1:]], 0)
        return f0 + r

    def predict(self, X):
        return (self.predict_proba(X) >= 0).astype(int)
