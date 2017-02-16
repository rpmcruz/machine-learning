import numpy as np

class GrowingWindow:
    def __init__(self, n_splits=3, test_size=1):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, y=None):
        train_size = len(X)-self.test_size
        train_blocks = np.linspace(0, train_size, self.n_splits+1, dtype=int)
        for j in train_blocks[1:]:
            yield np.arange(0, j), np.arange(j, j+self.test_size)
