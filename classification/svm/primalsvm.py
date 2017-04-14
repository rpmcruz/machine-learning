from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


# implementation based on Pegasos (Shwartz et al, 2007)

class PrimalLinearSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_, batch_size, fit_intercept=True, max_iter=1000):
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        y[y == 0] = -1  # convert to -1, +1
        assert np.unique(y).tolist() == [-1, 1]

        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        k = min(self.batch_size, len(X))

        self.w = np.zeros(X.shape[1])
        for t in range(1, self.max_iter+1):
            # choose samples
            ix = np.random.choice(len(X), k, False)
            Ap = [y[i]*X[i] for i in ix if y[i]*np.sum(self.w*X[i]) < 1]
            eta = 1/(self.lambda_*t)
            # update weights
            self.w = (1-eta*self.lambda_)*self.w + (eta/k)*np.sum(Ap, 0)
            _min = min(1, (1/np.sqrt(self.lambda_))/np.linalg.norm(self.w))
            self.w = _min*self.w
            # calculate loss
            margin = 1 - (y * np.sum(X * w, 1))
            ix = margin > 0
            loss = 0.5*self.lambda_*np.sum(w**2) + \
                (1/len(X))*np.sum(margin[margin > 0])
            print(loss)

        return self

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        return (np.sum(X*self.w, 1) >= 0).astype(int)
