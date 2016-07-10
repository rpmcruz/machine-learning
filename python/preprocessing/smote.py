# -*- coding: utf-8 -*-

# Algoritmos de resampling:
# * undersample: escolhe aleatoriamente sem repetição alguns dos pontos
# * oversample: mantém os primeiros e escolhe aleatoriamente com repetição
#               o resto
# * smote: oversample usando (Chawla, 2002), usa vizinhos. A nossa versão é
#          um pouco diferente: a original funciona apenas para múltiplos de
#          100.
# * msmote: variação em que:
#      all neighbors = 1 -> safe -> qualquer vizinho
#      all neighbors = 0 -> noise -> nada
#      else              -> border -> nearest neighbor

from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
import collections
import numpy as np

K_NEIGHBORS = 5

def smote(X, y, N, msmote, k):
    X0 = X[y == 0]
    T = X[y == 1]
    # based on http://comments.gmane.org/gmane.comp.python.scikit-learn/5278

    knn = NearestNeighbors(n_neighbors=min(len(T), k))
    knn.fit(T)

    if msmote:
        types = np.zeros(len(T))
        # 0 noise, 1 border, 2 safe
        nn = knn.kneighbors(T, return_distance=False)
        for i in xrange(len(T)):
            n1 = np.sum(y[nn[i]])
            if n1 == k:  # safe
                types[i] = 2
            elif n1 == 0:  # noise
                types[i] = 0
            else:
                types[i] = 1
        if np.sum(types == 0) == len(T):
            #raise ValueError('MSMOTE: all points are noise (k=%d)!' % k)
            msmote = False  # using regular SMOTE

    S = np.zeros((N, T.shape[1]))
    n = 0
    while n < N:
        i = np.random.randint(len(T))
        if not msmote or types[i] == 1:
            nn = knn.kneighbors(T[[i]], return_distance=False)
            # repeat until knn returns neighbor different than T[i]
            while True:
                nni = np.random.choice(nn[0])
                if nni != i:
                    break
        elif types[i] == 2:
            dist, nn = knn.kneighbors(T[[i]])
            nni = nn[0][np.argmin(dist)]
        else:
            continue
        dif = T[nni] - T[i]
        gap = np.random.random()
        S[n] = T[i] + gap*dif
        n += 1
    return (np.r_[X0, T, S], np.r_[np.zeros(len(X0)), np.ones(len(T)+N)])


class SMOTE(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, msmote=False):
        super(SMOTE, self).__init__()
        self.base_estimator = base_estimator
        self.msmote = msmote
        self.classes_ = (0, 1)

    def fit(self, X, y):
        y0 = np.sum(y == 0)
        y1 = np.sum(y == 1)
        factor = int(np.round(float(y0) / y1))
        X, y = smote(X, y, y1*(factor-1), self.msmote, K_NEIGHBORS)
        return self.base_estimator.fit(X, y)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        return self.base_estimator.predict(X)


class MSMOTE(SMOTE):
    def __init__(self, base_estimator):
        SMOTE.__init__(self, base_estimator, True)


if __name__ == '__main__':
    import matplotlib.pyplot as plot
    plot.ioff()

    def gauss(n=1000, r=8, stddev=2):
        import scipy.stats
        X = (np.random.rand(n, 2)*2-1)*r
        y = np.random.rand(n) < \
            scipy.stats.norm.pdf(X[:, 0], 0, stddev) * \
            scipy.stats.norm.pdf(X[:, 1], 0, stddev) / \
            (scipy.stats.norm.pdf(1, 0, stddev)**2)
        return (X, y)

    X, y = gauss(250)

    # normal
    plot.subplot(1, 4, 1)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('Normal')

    factor = 4

    # smote
    extra = len(y) - np.sum(y)
    Xn, yn = smote(X, y, np.sum(y)*factor, False, 8)
    plot.subplot(1, 4, 2)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        Xn[extra:, 0], Xn[extra:, 1], 'go',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('SMOTE %dx' % factor)

    # msmote
    Xn, yn = smote(X, y, np.sum(y)*factor, True, 8)
    plot.subplot(1, 4, 3)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        Xn[extra:, 0], Xn[extra:, 1], 'go',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('MSMOTE %dx' % factor)

    # normal
    X, y = gauss(250*factor)
    plot.subplot(1, 4, 4)
    plot.plot(
        X[y == 0, 0], X[y == 0, 1], 'bo',
        X[y == 1, 0], X[y == 1, 1], 'ro',
        markersize=4)
    plot.title('Normal %dx' % factor)

    plot.show()
