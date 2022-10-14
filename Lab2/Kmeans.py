import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, num_bases: int, simple=False):
        self.num_bases = num_bases
        self.simple = simple

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

        if self.simple:
            d_max = -np.inf
            for i in range(self.bases.size):
                for j in range(i + 1, self.bases.size):
                    d_max = max(d_max, abs(self.bases[i] - self.bases[j]))
            self.variance = np.full(
                self.num_bases, d_max ** 2 / (2 * self.num_bases))
        else:
            e = np.min(np.square(data - self.bases[:, np.newaxis]), 0)
            self.variance = np.bincount(labels, e) / np.bincount(labels)

        self.variance += 10 ** -100
        return self.bases, self.variance, error
