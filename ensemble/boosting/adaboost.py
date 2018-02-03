from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# NOTE: I convert y from {0,1} to {-1,+1} and then back again because
# it makes it easier for the learning method :P

class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, T, base_estimator=None, balanced=False):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.estimator = base_estimator
        self.T = T
        self.balanced = balanced
        self.classes_ = (0, 1)

    def fit(self, X, y):
        if self.balanced:
            cw = (2./np.sum(y == 0), 2./np.sum(y == 1))
            D = np.asarray([cw[_y] for _y in y])
        else:
            D = np.repeat(1./len(X), len(X))

        y = y*2-1  # change domain to {-1,+1}
        self.h = [None]*self.T
        self.a = [0]*self.T
        epsilon = 1e-6  # to avoid division by zero (Schapire and Singer, 1999)

        for t in range(self.T):
            self.h[t] = clone(self.estimator).fit(X, y, D)
            yp = self.h[t].predict(X)

            err = 1-np.sum((yp*y > 0)*D)
            self.a[t] = 0.5*np.log((1-err+epsilon)/(err+epsilon))
            D = D*np.exp(-self.a[t]*y*yp)
            D = D/np.sum(D)  # normalize distribution
        return self

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        # thresholds: we overload this function for easy integration with the
        # other models.
        return np.sum([a*h.predict(X) for a, h in zip(self.a, self.h)], 0)

    def predict(self, X):
        s = self.predict_proba(X)
        return (np.sign(s)+1)/2  # change domain back to {0,1}
