# https://en.wikipedia.org/wiki/Extreme_learning_machine

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

class ExtremeNet(BaseEstimator, ClassifierMixin):
    def __init__(self, size, C=1, class_weight=None, method='author1'):
        self.size = size
        self.C = C  # regularization
        self.class_weight = class_weight
        self.method = method
        self.classes_ = (0, 1)

    def fit(self, X, y):
        k0 = X.shape[1]
        k1 = self.size
        STDDEV = 1
        self.w0 = STDDEV * (np.random.rand(k0+1, k1)*2-1)

        # added costs as in (2013 Zong, Huang, Chen)
        if self.class_weight:
            w = [self.class_weight if _y == 1 else 1 for _y in y]

        W = np.eye(len(X)) * w
        H = sigmoid(np.dot(X, self.w0[1:, :]) + self.w0[0, :])  # (n,k1)

        self.w1 = np.dot(np.dot(np.dot(np.linalg.inv(
            (np.eye(self.size)/self.C) + np.dot(H.T, np.dot(W, H))), H.T), W),
            T)
        return self

    def predict(self, X):
        z2 = self.predict_proba(X)
        return np.argmax(z2, 1)

    def predict_proba(self, X):
        z0 = X
        z1 = sigmoid(np.dot(z0, self.w0[1:]) + self.w0[0])  # (n,k1)
        z2 = np.dot(z1, self.w1)  # (n)
        return z2
