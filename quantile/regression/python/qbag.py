import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# approach inspired by:
# http://blog.datadive.net/prediction-intervals-for-random-forests/

class QBag(BaseEstimator, RegressorMixin):
    def __init__(self, tau, base_estimator=None):
        self.tau = tau
        if base_estimator is None:
            base_estimator = RandomForestRegressor(100)
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        yp = np.zeros(len(X))
        ms = self.base_estimator.estimators_
        for i, x in enumerate(X):
            yps = [m.predict([x])[0] for m in ms]
            yp[i] = np.percentile(yps, self.tau*100)
        return yp
