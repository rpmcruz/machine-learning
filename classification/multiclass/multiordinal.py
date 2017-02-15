from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

# Algorithm from Chang, 2011
# Multi-class ensemble where classes are only compared to neighbor (ordinal)
# classes

class MultiOrdinal(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, alpha=1):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.ensemble = []
        for k in self.classes_[:-1]:
            Xpos = X[y > k]
            Xneg = X[y <= k]
            print('Xpos:', Xpos.shape, 'Xneg:', Xneg.shape)
            _X = np.r_[Xpos, Xneg]
            _y = np.r_[np.ones(len(Xpos), int), np.zeros(len(Xneg), int)]

            C = np.exp(self.alpha * np.abs(y-k-0.5))
            print(_y)
            print('_X:', _X.shape, '_y:', _y.shape, 'C:', C.shape)
            m = clone(self.estimator).fit(_X, _y, sample_weight=C)
            self.ensemble.append(m)
        return self

    def predict(self, X):
        yps = [m.predict(X) for m in self.ensemble]
        r = 1+np.sum(yps, 0)
        return r
