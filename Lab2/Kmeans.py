import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, num_bases: int, max_epochs: int = 100):
        self.num_bases = num_bases
        self.max_epochs = max_epochs

    def train(self, data: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        self.bases = np.random.permutation(data)[:self.num_bases]
        error = []
        prev_bases = []

        while not np.array_equal(prev_bases, self.bases):
            prev_bases = self.bases
            labels = np.argmin(np.square(data - self.bases[:, np.newaxis]), 0)
            self.bases = np.bincount(labels, data) / np.bincount(labels)

            e = np.min(np.square(data - self.bases[:, np.newaxis]), 0)
            error_sum = np.sum(e)
            error.append(error_sum)

        e = np.min(np.square(data - self.bases[:, np.newaxis]), 0)
        self.variance = np.bincount(labels, e) / np.bincount(labels)

        return self.bases, self.variance, error
