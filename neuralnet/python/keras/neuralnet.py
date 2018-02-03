from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Ordinal network encoding:
# http://orca.st.usm.edu/~zwang/files/rank.pdf

def create_model(nfeatures, nhidden, l2, K):
    reg = regularizers.l2(l2) if l2 else None

    input_layer = Input([nfeatures])
    hidden = Dense(
        nhidden, activation='tanh', kernel_regularizer=reg)(input_layer)
    output = Dense(K, activation='sigmoid')(hidden)
    
    model = Model(input_layer, output)
    model.compile('adam', 'binary_crossentropy')
    #model.summary()
    return model

def class_weight(y):
    klasses = np.unique(y)
    count = np.bincount(y)[klasses]
    return len(y) / (len(klasses)*count)


class MultiClassNet(BaseEstimator, ClassifierMixin):
    def __init__(self, nhidden, l2=0, balanced=False):
        self.nhidden = nhidden
        self.l2 = l2
        self.balanced = balanced

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yy = OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
        self.model = create_model(X.shape[1], self.nhidden, self.l2, K)
        cb = EarlyStopping('loss', 0.001, 1)
        ww = class_weight(y) if self.balanced else None
        self.logs = self.model.fit(
            X, yy, 512, 10000, 0, callbacks=[cb], class_weight=ww)
        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return np.argmax(self.model.predict(X), 1)


class OrdinalNet(BaseEstimator, ClassifierMixin):
    def __init__(self, nhidden, l2=0, balanced=False):
        self.nhidden = nhidden
        self.l2 = l2
        self.balanced = balanced

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yy = np.zeros((len(y), K-1), int)  # ordinal encoding
        for i,_y in enumerate(y):
            yy[i, 0:_y] = 1
        self.model = create_model(X.shape[1], self.nhidden, self.l2, K-1)
        cb = EarlyStopping('loss', 0.001, 1)
        ww = class_weight(y) if self.balanced else None
        self.logs = self.model.fit(
            X, yy, 512, 10000, 0, callbacks=[cb], class_weight=ww)
        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return np.sum(self.model.predict(X) > 0.5, 1)
