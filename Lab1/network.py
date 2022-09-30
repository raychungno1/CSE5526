import numpy as np
from activation import sigmoid, sigmoid_prime

from error import sse, sse_prime


class Network:
    def __init__(self):
        self.w1 = np.random.rand(4, 4) * 2 - 1
        self.b1 = np.random.rand(4, 1) * 2 - 1
        self.w2 = np.random.rand(1, 4) * 2 - 1
        self.b2 = np.random.rand() * 2 - 1

    def train(self, train, labels, learning_rate):
        model_passes = False
        epoch = 0

        while not model_passes:  # and epoch < 1000:
            error = 0
            model_passes = True
            abs_err = 0
            for x, d in zip(train, labels):
                self.predict(x)
                error += sse(d, self.y2)
                abs_err += abs(d - self.y2[0][0])
                if abs(d - self.y2[0][0]) >= 0.05:
                    model_passes = False

                delta2 = sse_prime(d, self.y2) * sigmoid_prime(self.y2)
                delta1 = sigmoid_prime(self.y1) * (delta2 * self.w2.T)

                self.w2 = self.w2 - learning_rate * delta2 @ self.y1.T
                self.b2 = self.b2 - learning_rate * delta2
                self.w1 = self.w1 - learning_rate * delta1 @ self.y0.T
                self.b1 = self.b1 - learning_rate * delta1

            # print(
            #     f"Epoch: {epoch}\tError: {error}\tAvg Abs Error: {abs_err/16}")
            epoch += 1

        return epoch

    def predict(self, x):
        self.y0 = np.reshape(x, (4, 1))
        self.y1 = sigmoid(self.w1 @ self.y0 + self.b1)
        self.y2 = sigmoid(self.w2 @ self.y1 + self.b2)
        return self.y2

    def set_debug(self):
        for i in range(0, 4):
            self.w1[i, :] = self.w2[:, i] = 0.1 * i + 0.1
        self.b1[:, 0] = self.b2 = 0.5


if __name__ == "__main__":
    n = Network()
    n.set_debug()

    n.predict([1, 1, 1, 1])

    print("----- y0 -----")
    print(n.y0)
    print("----- y1 -----")
    print(n.y1)
    print("----- y2 -----")
    print(n.y2)
    print("----- Expected -----")
    print(0.792)

    n.train([[1, 1, 1, 1]], [0], 0.5)

    print("----- w1 -----")
    print(n.w1)
    print("----- b1 -----")
    print(n.b1)
    print("----- w2 -----")
    print(n.w2)
    print("----- b2 -----")
    print(n.b2)
