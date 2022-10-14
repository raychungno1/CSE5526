import numpy as np
from numpy import ndarray


class RBF:
    def __init__(self, bases: ndarray, variance: ndarray):
        self.bases = bases
        self.variance = variance
        self.w = np.random.rand(1, bases.size) * 2 - 1
        self.b = np.random.rand() * 2 - 1

    def train(self, data: ndarray, labels: ndarray, lr: float) -> ndarray:
        error_hist = []

        for _ in range(100):
            sse = 0

            for x, d in zip(data, labels):
                y = self.predict(x)
                e = (d - y)

                sse += e ** 2
                self.w += lr * e * self.x_t
                self.b += lr * e

            error_hist.append(sse)

        return error_hist

    def predict(self, x: float) -> float:
        self.x_t = np.exp(-np.divide(np.square(x - self.bases), (2 * self.variance)))
        return self.w @ self.x_t + self.b
