from sklearn.base import BaseEstimator, ClassifierMixin
from rank2ordinal.threshold import decide_thresholds
import numpy as np


class Rank2Ordinal(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, threshold_strategy='uniform'):
        self.estimator = estimator
        self.threshold_strategy = threshold_strategy

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimator.fit(X, y)

        K = len(self.classes_)
        scores = self.estimator.predict(X)
        self.ths = decide_thresholds(scores, y, K, self.threshold_strategy)
        return self

    # this function passes the ranking score for use by some metrics
    def predict_proba(self, X):
        return self.estimator.predict(X)

    def predict(self, X):
        scores = self.estimator.predict(X)
        return np.sum(scores[:, np.newaxis] >= self.ths, 1)
