# -*- coding: utf-8 -*-

# GBRank as in Zheng et al (2007)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from utils import choose_threshold
import itertools
import numpy as np

# Start with an initial guess h0, for k = 1, 2, ...
#
# 1) using h_{k-1} as the current approximation of h, we separate S into two
# disjoint sets, S+ = {(x_i, y_i) e S | h_{k-1}(x_i) >= h_{k-1}(y_i)+tau} and
# S- = {(x_i, y_i) e S | h_{k-1}(x_i) < h_{k-1}(y_i)+tau}
#
# 2) fitting a regression function g_k(x) using GBT and the following training
# data {(x_i,h_{k-1}(yi)+tau), (y_i,h_{k-1}(x_i)-tau) | (xi, yi) e S-}
#
# 3) forming (with normalization of the range of h_k) h_k(x) =
# k h_{k-1}(x)+eta g_k(x), k+1 where eta is a shrinkage factor


class H0:  # NOTE: not being used
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.prod(X, axis=1)


class GBRank(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, max_its=np.inf):
        super(GBRank, self).__init__()
        self.g = []
        self.eta = 0.10  # 0.05
        self.kmax = max_its
        if base_estimator is None:
            base_estimator = GradientBoostingRegressor()
        self.estimator = base_estimator

    # Estranhamente, este código até parece mais rápido que o anterior!

    def fit(self, X, y):
        tau = 1  # ]0,1]

        Xs = []
        H = []
        for i in xrange(len(X)):
            for j in xrange(len(X)):
                if y[i] > y[j]:
                    Xs.append(X[i])  # y[i] > y[i+1]
                    Xs.append(X[j])
                    H.append(2)
                    H.append(0)
        l = np.ones(len(H), bool)

        Xs = np.asarray(Xs)  # as matrix
        H = np.asarray(H).T  # as matrix

        while self.kmax > len(self.g):
            g = clone(self.estimator).fit(Xs[l], H[l])
            self.g.append(g)

            H = self.predict_proba(Xs).T  # as matrix

            for i in xrange(len(H)-2, -1, -2):
                if l[i]:
                    assert (i % 2) == 0
                    if H[i] >= H[i+1]+tau:
                        l[i] = False
                        l[i+1] = False
                    else:
                        Hi = H[i+1]+tau
                        H[i+1] = H[i]-tau
                        H[i] = Hi
            if np.sum(l) == 0:  # converged
                break

        H = self.predict_proba(X)
        self.th = choose_threshold(H, y)
        return self

    '''
    # old code: does not converge
    def fit(self, X, y):
        tau = 1  # ]0,1]

        S = []
        for i, j in itertools.combinations(xrange(X.shape[0]), 2):
            if y[i] > y[j]:
                S.append((i, j))

        # initial guess
        self.h0 = clone(self.estimator)
        #self.h0 = DecisionTreeRegressor(max_depth=1)
        #self.h0 = H0()
        self.h0 = self.h0.fit(X, y)
        self.g.append(self.h0)

        done = False
        while not done and self.kmax > len(self.g):
            # Juntamos os passos 1 e 2
            # O passo 3 é feito por predict() e hk()

            H = self.predict_proba(X)

            Xt = []
            yt = []
            done = True
            for i, j in S:
                if H[i] < H[j]+tau:
                    Xt.append(X[i])
                    yt.append(H[j]+tau)
                    Xt.append(X[j])
                    yt.append(H[i]-tau)
                    done = False

            # if h0=g[0], then we could merge and move to top of the cycle
            if not done:
                g = clone(self.estimator)
                g = g.fit(Xt, yt)
                self.g.append(g)
        if len(self.g) == 0:
            H = self.predict_proba(X)

        self.th = choose_threshold(H, y)
        return self
    '''

    def hk(self, X, k):
        if k == 0:
            s = self.g[0].predict(X)
            return s
        g = self.g[k].predict(X)
        h = self.hk(X, k-1)
        return (k*h + self.eta*g) / float(k+1)

    def predict_proba(self, X):
        # these are not probabilities, but I overloaded this function because
        # it makes it nicer to have a common interface for the ROC curves
        X = np.asarray(X)
        return self.hk(X, len(self.g)-1)

    def predict(self, X):
        return (self.predict_proba(X) >= self.th).astype(int)
