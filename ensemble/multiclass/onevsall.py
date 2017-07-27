import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class MyOneVsAll(BaseEstimator, ClassifierMixin):
    # Algorithm from wikipedia:
    # https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y).astype(int)
        self.ms = [None] * len(self.classes_)
        for i, k in enumerate(self.classes_):
            _y = (y == k).astype(int)
            m = clone(self.estimator).fit(X, _y)
            self.ms[i] = m
        return self

    def predict(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        yp = np.zeros(len(X), np.int)

        for i, m in enumerate(self.ms):
            score = m.decision_function(X)
            for j, y in enumerate(yp):
                scores[j, i] = score[j]

        for i in range(len(X)):
            k = np.argmax(scores[i, :])
            # print ' '.join(["%0.4f" % x for x in scores[i, :]]),
            # print k
            yp[i] = self.classes_[k]
        assert len(yp) == len(X)
        return yp
