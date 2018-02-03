from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Boosting algorithm which uses another metric for success.
# Algorithm from Wu et al (2008)
# Freund (2003)
# https://fr.wikipedia.org/wiki/RankBoost

# NOTE: I convert y from {0,1} to {-1,+1} and then back again because
# it makes it easier for the learning method :P

class RankBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, T, base_estimator=None):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.estimator = base_estimator
        self.T = T
        self.classes_ = (0, 1)

    def fit(self, X, y):
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        D = np.repeat(1./(n0*n1), n0*n1)

        # emparelhado
        #X0 = np.repeat(X[y == 0], n1, 0)
        #X1 = np.tile(X[y == 1], (n0, 1))

        #Xs = np.r_[X0, X1]
        ys = np.r_[np.zeros(n0*n1), np.ones(n0*n1)]

        nn = np.sum(y == 1)*np.sum(y == 0)
        len_diff = nn*2
        Xs = np.zeros((len_diff, X.shape[1]))
        #ys = np.zeros(len_diff)

        for i, x0 in enumerate(X[y==0]):
            for j in range(n1):
                Xs[i*n1 + j] = x0

        for i, x1 in enumerate(X[y==1]):
            for j in range(n0):
                Xs[nn + i + n1*j] = x1

        """
        for X1, y1 in zip(X, y):
            for X2, y2 in zip(X, y):
                if y1 > y2:
                    Xs[i] = X2
                    #ys[i] = 0
                    Xs[i+nn] = X1
                    ys[i+nn] = 1
                    i += 1
        """

        self.h = [None]*self.T
        self.a = [0]*self.T
        epsilon = 1e-6  # to avoid division by zero (Schapire and Singer, 1999)

        for t in range(self.T):
            # Train weak ranker ft based on distribution Dt
            Ds = np.r_[D, D]/2
            self.h[t] = clone(self.estimator).fit(Xs, ys, Ds)

            # Choose alpha
            # there are apparently several approaches for this; we are using
            # one for a classification base estimator
            f_X0 = self.h[t].predict(X[y == 0])
            f_X1 = self.h[t].predict(X[y == 1])

            """
            Wneg = 0.
            Wpos = 0.
            i = 0
            for f0 in f_X0:
                for f1 in f_X1:
                    next_df = f0 - f1
                    if next_df == -1:
                        Wneg += D[i]
                    else:
                        Wpos += D[i]
                    i += 1
            """
            df = np.repeat(f_X0, n1) - np.tile(f_X1, n0)
            """
            f_X0 = self.h[t].predict(X0)
            f_X1 = self.h[t].predict(X1)

            df = f_X0 - f_X1  # (n0*n1)
            """

            Wneg = np.sum(D[df == -1])  # right
            Wpos = np.sum(D[df == +1])  # wrong

            self.a[t] = 0.5*np.log((Wneg+epsilon)/(Wpos+epsilon))

            # Update D
            D = D*np.exp(self.a[t]*df)
            D = D/np.sum(D)  # normalize distribution
        return self

    def predict(self, X):
        return np.sum([a*h.predict(X) for a, h in zip(self.a, self.h)], 0)
