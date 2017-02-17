import numpy as np


def pinball_score(y, yp, tau):
    # quantile score, also known as the check function (Koenker, 2005)
    r = y-yp
    I = (r > 0).astype(int)
    return np.mean(r*(I+tau))
