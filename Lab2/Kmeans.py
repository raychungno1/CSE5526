import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, num_bases: int, max_epochs: int = 100):
        self.num_bases = num_bases
        self.max_epochs = max_epochs

    def train(self, data: ndarray, labels: ndarray):
        self.bases = np.random.choice(data, self.num_bases)
        error = []

        for i in range(self.max_epochs):
            prev_bases = self.bases
            pred = np.argmin(np.square(data - self.bases[:, np.newaxis]), 0)
            self.bases = np.bincount(pred, data) / np.bincount(pred)

            if np.array_equal(prev_bases, self.bases):
                break

            e = np.min(np.square(data - self.bases[:, np.newaxis]), 0)
            error_sum = np.sum(e)
            error.append(error_sum)
            print(f"Epoch: {i}\tCost: {error_sum}")

        print(np.bincount(pred))
        self.variance = np.bincount(pred, e) / np.bincount(pred)

        return self.bases, self.variance, error
