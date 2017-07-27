import numpy as np

class SlidingWindow:
    def __init__(self, n_splits=3, test_size=1):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, y=None):
        train_size = len(X)-self.test_size
        train_blocks = np.linspace(0, train_size, self.n_splits+1, dtype=int)
        train_i = train_blocks[:-1]
        train_j = train_blocks[1:]
        for i, j in zip(train_blocks[:-1], train_blocks[1:]):
            yield np.arange(i, j), np.arange(j, j+self.test_size)
