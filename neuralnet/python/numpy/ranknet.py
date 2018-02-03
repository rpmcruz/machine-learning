# -*- coding: utf-8 -*-

# Basic neural network (uses sigmoid on all layers)

## These have been deprecated in favor of our much faster C++
## implementation ##

from sklearn.base import BaseEstimator, ClassifierMixin
from utils import balanced_weights
import numpy as np
import sklearn.metrics


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    # dsigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    return x*(1-x)


class BaseNeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_nodes, class_weight=None):
        self.hidden_nodes = hidden_nodes
        self.eta = 0.7  # 0.7
        self.maxit = int(1e5)
        self.class_weight = class_weight
        self.classes_ = (0, 1)

    # normalization should probably be removed from this class
    def fitnorm(self, X):
        self.min = np.amin(X, 0)
        self.max = np.amax(X, 0)
        return self.applynorm(X)

    def applynorm(self, X):
        return (X - self.min) / (self.max - self.min + 1e-6)

    def build(self, k0, k1):
        rang = 0.12  # 0.7  # initial random weights on [-rang, rang]
        self.w0 = rang * (np.random.rand(k0, k1)*2-1)  # (k0,k1)
        self.b0 = rang * (np.random.rand(1, k1)*2-1)
        self.w1 = rang * (np.random.rand(k1, 1)*2-1)  # (k1,1)
        self.b1 = rang * (np.random.rand(1, 1)*2-1)

    def fprop(self, X):
        l0 = X
        l1 = sigmoid(np.dot(l0, self.w0) + self.b0)  # (n,k1)
        l2 = sigmoid(np.dot(l1, self.w1) + self.b1)  # (n)
        return (l0, l1, l2)

    def backprop(self, C, eta, l0, l1, l2):
        # nodes are updated based on how much impact they have in the
        # next layer times their unconfidence (dsigmoid)

        l2_ = dsigmoid(l2)
        delta1 = C*l2_  # (n,1)

        l1_ = dsigmoid(l1)  # (n,h)
        delta0 = delta1 * (self.w1.T * l1_)  # (n,h)

        db1 = np.sum(delta1, 0, keepdims=True)
        dw1 = np.dot(l1.T, delta1)
        db0 = np.sum(delta0, 0, keepdims=True)
        dw0 = np.dot(l0.T, delta0)

        self.b1 += eta * db1
        self.w1 += eta * dw1
        self.b0 += eta * db0
        self.w0 += eta * dw0

    def _fit(self, X, y, sample_weight):
        pass # TO OVERLOAD

    def fit(self, X, y):
        if self.class_weight == 'balanced':
            sample_weight = balanced_weights(y)[np.newaxis].T
        elif self.class_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray([self.self.class_weight[i] for i in y])

        self.build(X.shape[1], self.hidden_nodes)
        X = self.fitnorm(X)
        y = y[np.newaxis].T  # we need it to be a column vector
        self._fit(X, y, sample_weight)
        return self

    def predict_proba(self, X):
        return self.fprop(X)[2][:, 0]

    def predict(self, X):
        X = self.applynorm(X)
        return (self.predict_proba(X) >= 0.5).astype(int)



class BatchNeuralNet(BaseNeuralNet):
    def _fit(self, X, y, sample_weight):
        eta = self.eta / len(X)
        for t in xrange(self.maxit):
            l0, l1, l2 = self.fprop(X)

            C = (l2 - y) #* sample_weight[i]
            self.backprop(C, -eta, l0, l1, l2)
            error = np.mean(np.abs(C))

            #if t % (self.maxit*10) == 0:
            #    print error, sklearn.metrics.f1_score(y, (l2 >= 0.5).astype(int))
            if error < 1e-2:  # has converged
                #print 'converged!'
                break


class StochasticNeuralNet(BaseNeuralNet):
    def _fit(self, X, y, sample_weight):
        eta = self.eta
        # stochastic descent
        error = 0
        for t in xrange(self.maxit):
            oldb1 = np.copy(self.b1)
            oldw1 = np.copy(self.w1)
            oldb0 = np.copy(self.b0)
            oldw0 = np.copy(self.w0)
            olderror = error
            error = 0

            for i in xrange(len(X)):
                l0, l1, l2 = self.fprop(X[[i]])

                C = (l2 - y[i]) * sample_weight[i]
                self.backprop(C, -eta, l0, l1, l2)
                error += np.abs(C)
            error = error/len(X)

            # http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/
            if error < olderror:
                eta += 0.01*eta
            elif t > 0:
                self.b1 = oldb1
                self.w1 = oldw1
                self.b0 = oldb0
                self.w0 = oldw0
                oldeta = eta
                eta = 0.50*eta
                print oldeta, '->', eta

            print error
            if error < 1e-2:  # has converged
                print 'converged!'
                break


###########


class BaseRankNet(BaseNeuralNet):
    def _fit(self, X, P, idx, jdx):
        pass

    def fit(self, X, y):
        idx = np.repeat(np.arange(len(X)), len(X))
        jdx = np.tile(np.arange(len(X)), len(X))
        
        dx = np.r_[idx, jdx]

        y = y[np.newaxis].T  # we need it to be a column vector
        P = (y[idx] - y[jdx] + 1)/2.

        self.build(X.shape[1], self.hidden_nodes)
        X = self.fitnorm(X)

        self._fit(X, P, idx, jdx)
        H = self.predict_proba(X)
        self.th = choose_threshold(H, y)
        return self

    def predict(self, X):
        X = self.applynorm(X)
        return (self.predict_proba(X) >= self.th).astype(int)


class BatchRankNet(BaseRankNet):
    def _fit(self, X, P, idx, jdx):
        # batch gradient descent
        eta = self.eta / (len(X)**2)
        for t in xrange(int(self.maxit)):
            l0, l1, l2 = self.fprop(X)

            s = np.r_[l2[[idx]] - l2[[jdx]], l2[[jdx]] - l2[[idx]]]
            exp_s = np.exp(s)
            C = exp_s/(exp_s+1) - np.r_[P, P]

            self.backprop(C, -eta, l0[[dx]], l1[[dx]], l2[[dx]])

            error = np.mean(np.abs(C))
            #if t % (self.maxit*10) == 0:
            #    print error
            if error < 1e-2:  # has converged
                #print 'converged!'
                break


class StochasticRankNet(BaseRankNet):
    def _fit(self, X, P, idx, jdx):
        # stochastic gradient descent
        for t in xrange(self.maxit):
            error = 0
            for i in xrange(len(idx)):
                l0_1, l1_1, l2_1 = self.fprop(X[[idx[i]]])
                l0_2, l1_2, l2_2 = self.fprop(X[[jdx[i]]])

                s = l2_1 - l2_2
                C = np.exp(s)/(np.exp(s)+1) - P[i]

                self.backprop(C, -1, l0_1, l1_1, l2_1)
                self.backprop(C, +1, l0_2, l1_2, l2_2)
                error += np.abs(C)
            print error/len(idx)
            if error/len(idx) < 1e-2:
                break





if __name__ == '__main__':  # test
    import test
    test.test(('NeuralNet',), (StochasticNeuralNet(8),), 1)
