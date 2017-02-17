# introduce delay in the target variable

import numpy as np

def shift(y, delay):
    X = np.asarray([np.roll(y, d) for d in range(1, delay+1)]).T
    X = X[delay:]
    y = y[delay:]
    t = np.arange(delay, len(y)+delay, dtype=int)
    return t, X, y
