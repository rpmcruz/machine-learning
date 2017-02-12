from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class MyRandomForestClassifier:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.trees = [DecisionTreeClassifier(random_state=1).fit(X, y)
                      for _ in range(self.n_estimators)]
        return self

    def predict(self, X):
        yp = [tree.predict(X) for tree in self.trees]
        return ((np.sum(yp, 0) / len(self.trees)) > 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
