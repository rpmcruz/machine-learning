from keras.models import Model
from keras.layers import Input, Dense, Subtract
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

def preprocess(X, y):
    K = len(np.unique(y))
    N = len(X)
    Nk = np.bincount(y)

    X1 = np.repeat(X, N, 0)
    X2 = np.tile(X.T, N).T

    y1 = np.repeat(y, N)
    y2 = np.tile(y, N)
    yy = (y1 > y2) + (y1 == y2)*0.5

    pairs = K*(K-1)
    ww = len(X1) / (pairs * (Nk[y1]*Nk[y2]))
    return X1, X2, yy, ww


def create_model(nfeatures, nhidden, l2):
    reg = regularizers.l2(l2) if l2 else None

    input1 = Input([nfeatures])
    input2 = Input([nfeatures])

    hidden = Dense(nhidden, activation='tanh', kernel_regularizer=reg)
    
    output1 = hidden(input1)
    output2 = hidden(input2)
    diff = Subtract()([output1, output2])
    output = Dense(1, activation='sigmoid')(diff)
    
    model = Model([input1, input2], output)
    model.compile('adam', 'binary_crossentropy')
    #model.summary()
    return model


class RankNet(BaseEstimator, RegressorMixin):
    def __init__(self, nhidden, l2=0):
        self.nhidden = nhidden
        self.l2 = l2

    def fit(self, X, y):
        self.model = create_model(X.shape[1], self.nhidden, self.l2)
        X1, X2, yy, ww = preprocess(X, y)
        cb = EarlyStopping('loss', 0.001, 10)
        self.logs = self.model.fit(
            [X1, X2], yy, 128, 10000, 0, callbacks=[cb], sample_weight=ww)
        return self

    def predict(self, X):
        return self.model.predict([X, np.zeros_like(X)])[:, 0]
