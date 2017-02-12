from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from qbc import QBC
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

print('Testing quantile regression...')

_datasets = [
    (datasets.load_iris, 1),
    (lambda _: datasets.load_digits(10, True), 5),
]

quantiles = [0.1, 0.9]

n_estimators = 100
eta = 0.1

models = [
    ('dummy', lambda tau: DummyClassifier('most_frequent')),
    ('gboost', lambda _: GradientBoostingClassifier(
        learning_rate=eta, n_estimators=n_estimators, max_depth=1)),
    ('qbc', lambda tau: QBC(tau, n_estimators, eta)),
]

for dataset, th in _datasets:
    print('# dataset %s' % dataset.__name__)
    X, y = dataset(True)
    y = (y >= th).astype(int)
    yps = [[np.zeros(len(y)) for _ in quantiles] for _ in models]
    Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.8)
    for q, tau in enumerate(quantiles):
        print('## quantile %.2f' % tau)
        for i, (name, model) in enumerate(models):
            m = model(tau).fit(Xtr, ytr)
            yp = m.predict(Xts)

            (TNR, FPR), (FNR, TPR) = confusion_matrix(yts, yp)/len(yts)
            print('%10s: FP: %.3f, FN: %.3f' % (name, FPR, FNR))
