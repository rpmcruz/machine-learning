from statsmodels.regression.quantile_regression import QuantReg


class SkQuantReg:
    def __init__(self, tau):
        self.tau = tau

    def fit(self, X, y):
        self.m = QuantReg(y, X).fit(self.tau)
        return self

    def predict(self, X):
        return self.m.predict(X)
