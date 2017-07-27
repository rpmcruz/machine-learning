from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def I(cond):
    return cond.astype(int)


class ZerosDummyModel:
    def __init__(self, tau):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class GBoost(BaseEstimator, RegressorMixin):
    def __init__(
            self, M=100, eta=0.1, first_estimator=None, base_estimator=None,
            loss='mse', tau=None):
        self.tau = tau
        self.M = M
        self.eta = eta
        if first_estimator is None:
            # first_estimator = ZerosDummyModel(tau)
            first_estimator = TauDummyModel(tau)
        self.first_estimator = first_estimator
        if base_estimator is None:
            base_estimator = DecisionTreeRegressor(max_depth=1)
        self.base_estimator = base_estimator
        if loss == 'quantile':
            # lambda y, f: self.tau*I(y-f >= 0) - I(f-y >= 0)
            self.loss = \
                lambda y, f: I(y-f >= 0) - (1-self.tau)
        elif loss == 'mae':
            self.loss = \
                lambda y, f: I(y-f >= 0)-0.5
        elif loss == 'mse':
            self.loss = \
                lambda y, f: y-f
        else:
            raise 'Unknown loss function: %s' % loss
        self.tau = tau

    def fit(self, X, y):
        self.fs = [self.first_estimator]
        # step 0
        f = self.first_estimator.fit(X, y)
        # step 1
        for m in range(self.M):
            f = self.predict(X)
            # step 2
            U = self.loss(y, f)
            # step 3
            g = clone(self.base_estimator).fit(X, U)
            # step 4
            self.fs.append(g)
        return self

    def predict(self, X):
        f0 = self.fs[0].predict(X)
        r = np.sum([self.eta * f.predict(X) for f in self.fs[1:]], 0)
        return f0 + r

