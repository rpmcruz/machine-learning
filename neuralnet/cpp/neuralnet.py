# -*- coding: utf-8 -*-

# Wrapper around cpp/neuralnet.cpp

import os
os.system('cpp/compile.sh')
print 'compiled neuralnet.cpp -> libneuralnet.so'

import ctypes
lib = ctypes.cdll.LoadLibrary('cpp/libneuralnet.so')

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_nodes, balanced, ranker=False, maxit=1000):
        self.obj = None
        self.hidden_nodes = hidden_nodes
        self.balanced = balanced
        self.ranker = ranker
        self.maxit = maxit
        self.classes_ = (0, 1)

    def __del__(self):
        if self.obj is not None:
            lib.NeuralNet_delete(self.obj)

    def fit(self, X, y):
        # it's ugly, but better to allocate nnet c++ here
        if self.obj is not None:
            lib.NeuralNet_delete(self.obj)

        X = np.asarray(X, np.float64, 'F')
        y = np.asarray(y, np.int32)
        Xptr = ctypes.c_void_p(X.ctypes.data)
        yptr = ctypes.c_void_p(y.ctypes.data)

        if self.ranker:
            self.obj = lib.RankNet_new(self.hidden_nodes)
        else:
            self.obj = lib.NeuralNet_new(self.hidden_nodes, self.balanced)
        lib.NeuralNet_fit(self.obj, X.shape[1], X.shape[0], Xptr, yptr, self.maxit)
        return self

    def predict(self, X):
        X = np.asarray(X, np.float64, 'F')
        y = np.zeros(len(X), np.int32)
        Xptr = ctypes.c_void_p(X.ctypes.data)
        yptr = ctypes.c_void_p(y.ctypes.data)
        lib.NeuralNet_predict(self.obj, X.shape[1], X.shape[0], Xptr, yptr)
        return y

    def predict_proba(self, X):
        X = np.asarray(X, np.float64, 'F')
        s = np.zeros(len(X), np.float64)
        Xptr = ctypes.c_void_p(X.ctypes.data)
        sptr = ctypes.c_void_p(s.ctypes.data)
        lib.NeuralNet_scores(self.obj, X.shape[1], X.shape[0], Xptr, sptr)
        return s


class RankNet(NeuralNet):
    def __init__(self, hidden_nodes, maxit=1000):
        NeuralNet.__init__(self, hidden_nodes, False, True, maxit)
