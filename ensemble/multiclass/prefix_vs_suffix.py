from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

# Algorithm from Chang, 2011
# Multi-class ensemble where classes are only compared to neighbor (ordinal)
# classes

class PrefixVsSuffix(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, alpha=1):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.ensemble = []
        for k in self.classes_[:-1]:
            Xpos = X[y > k]
            Xneg = X[y <= k]
            _X = np.r_[Xpos, Xneg]
            _y = np.r_[np.ones(len(Xpos), int), np.zeros(len(Xneg), int)]

            C = np.exp(self.alpha * np.abs(y-k-0.5))
            m = clone(self.estimator)
            try:
                m.fit(_X, _y, sample_weight=C)
            except:
                m.fit(_X, _y)
            self.ensemble.append(m)
        return self

    def predict(self, X):
        return np.sum([m.predict(X) for m in self.ensemble], 0)
