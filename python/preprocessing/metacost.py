# -*- coding: utf-8 -*-

# Domingos (1999)

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import balanced_class_weights
import numpy as np


def full_resamples(X, y, nresamples):
    n0 = np.sum(y == 0)  # do a stratified full resample
    n1 = np.sum(y == 1)
    _X = np.r_[X[y == 0], X[y == 1]]  # re-order to simplify things
    _y = np.r_[np.zeros(n0, int), np.ones(n1, int)]

    s = [None] * nresamples
    for i in xrange(nresamples):
        r0 = np.random.randint(0, n0, n0)  # full resample
        r1 = np.random.randint(0, n1, n1) + n0
        r = np.r_[r0, r1]
        s[i] = (_X[r], _y[r])
    return s


class MetaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, C, use_predict_proba):
        super(MetaCost, self).__init__()
        self.base_estimator = base_estimator
        self.C = C
        self.use_predict_proba = use_predict_proba
        self.ret = None
        self.classes_ = (0, 1)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.array(y, copy=True)
        if self.C == 'balanced':
            C = balanced_class_weights(y)
        else:
            C = self.C

        # number of resamples to generate
        m = 50
        # number of examples in each resample
        # (can be smaller than dataset)
        n = len(X)
        # do models produce class probabilities?
        p = self.use_predict_proba
        # are all resamples to be used for each example
        #q = True  # TODO: only True supported (recommended: False)

        # Step 1. Train everything
        M = [None] * m
        for i, (Xt, yt) in enumerate(full_resamples(X, y, m)):
            m = clone(self.base_estimator)
            M[i] = m.fit(Xt, yt)

        # Step 2. Per observation, action (i.e. relabel)
        for i in xrange(len(X)):  # observation
            if p:
                Pj = [m.predict_proba(X[[i]]) for m in M]
            else:
                Pj = [(1, 0) if m.predict(X[[i]]) == 0 else (0, 1) for m in M]
            P = np.mean(Pj, 0)
            j = np.argmax(P * C)
            y[i] = j

        # WEIRD: for whatever reason some models (LinearSVC for instance) need
        # more than one observation of each class to work
        if np.sum(y == 1) <= 1:
            self.ret = 0
        elif np.sum(y == 0) <= 1:
            self.ret = 1
        else:
            # Step 3. Train final model with new data
            self.model = clone(self.base_estimator)
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.ret is None:
            return self.model.predict_proba(X)
        return np.zeros(len(X))

    def predict(self, X):
        if self.ret is None:
            return self.model.predict(X)
        return np.repeat(self.ret, len(X))
