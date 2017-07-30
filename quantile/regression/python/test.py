from sklearn.model_selection import train_test_split
from score import pinball_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from qbr import QBR, TauDummyModel
from qbag import QBag
from skquantreg import SkQuantReg
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

print('Testing quantile regression...')

datasets = [
    datasets.load_boston,
    datasets.load_diabetes,
]

quantiles = [0.1, 0.9]

n_estimators = 1000
eta = 0.1

models = [
    ('dummy', lambda tau: TauDummyModel(tau)),
    ('qreg', lambda tau: SkQuantReg(tau)),
    ('gboost', lambda _: GradientBoostingRegressor(
        'ls', eta, n_estimators, max_depth=1)),
    ('qgboost', lambda tau: GradientBoostingRegressor(
        'quantile', eta, n_estimators, max_depth=1, alpha=tau)),
    ('forest', lambda _: RandomForestRegressor(n_estimators)),
    ('qbag', lambda tau: QBag(tau, RandomForestRegressor(n_estimators))),
    ('qbr', lambda tau: QBR(tau, n_estimators, eta)),
]

for dataset in datasets:
    print('# dataset %s' % dataset.__name__)
    X, y = dataset(True)
    yps = [[np.zeros(len(y)) for _ in quantiles] for _ in models]
    Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.8)

    for q, tau in enumerate(quantiles):
        print('## quantile %.2f' % tau)
        for i, (name, model) in enumerate(models):
            m = model(tau).fit(Xtr, ytr)
            yp = m.predict(Xts)
            score = pinball_score(yts, yp, tau)
            if i == 0:
                base_score = score
            score /= base_score
            print('%10s: %.3f' % (name, score))
            yps[i][q] = yp

x = range(len(yts))
o = np.argsort(yts)
plt.scatter(x, yts[o], s=2, color='black')
plt.title('y')
plt.show()

for q, tau in enumerate(quantiles):
    plt.scatter(x, yts[o], s=2, color='black')
    for i in [0, 1, 3, 6]:
        color = 'gray' if i == 0 else None
        plt.plot(x, yps[i][q][o], label=models[i][0], color=color)
    plt.ylim(np.min(yts), np.max(yts))
    plt.legend()
    plt.title('tau=%.2f' % tau)
    plt.show()
