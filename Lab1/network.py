import numpy as np
from activation import sigmoid, sigmoid_prime

from error import sse, sse_prime


class Network:
    def __init__(self):
        self.w1 = np.random.rand(4, 5) * 2 - 1
        self.w2 = np.random.rand(1, 5) * 2 - 1

    def train(self, train, labels, learning_rate):
        model_passes = False
        epoch = 0

        while not model_passes:
        
            error = 0
            error_prime = 0
            # model_passes = True
            abs_err = 0

            for x, d in zip(train, labels):
                self.predict(x)
                error += sse(d, self.y2)
                error_prime += sse_prime(d, self.y2)
                
                abs_err += abs(d - self.y2)
                # if abs(d - self.y2) >= 0.05:
                    # model_passes = False

            # (1, 1) * (1, 1) = (1, 1)
            delta2 = error_prime * sigmoid_prime(self.v2)

            # (1, 1) * (1, 1) * (1, 5) = (1, 5)
            delta1 = sigmoid_prime(self.v1) * delta2 * self.w2

            # (1, 1) * (1, 5) = (1, 5)
            de_dw2 = delta2 @ self.y1.T
            self.w2 = self.w2 - learning_rate * de_dw2
            # print(de_dw2)

            # (4, 1) * (1, 5) = (4, 5)
            de_dw1 = self.y0.T * delta1
            self.w1 = self.w1 - learning_rate * de_dw1

            print(f"Epoch: {epoch}\tError: {error}\t Abs Error: {abs_err}")
            epoch += 1
        # print(de_dw1)

            # (1, 1)
            # dy2_dv2 = sigmoid_prime(self.v2)

            # # (1, 1) * (1, 1) = (1, 1)
            # delta2 = dy2_dv2 * de_dy2

            # # (1, 1) * (1, 5) = (1, 5)
            # de_dw2 = delta2 @ self.y1.T

            # # (1, 1) * (1, 5) = (1, 5)
            # de_dy1 = delta2 @ self.w2

            # # (4, 1)
            # dy1_dv1 = sigmoid_prime(self.v1)

            # # (4, 1) * (1, 5) = (4, 5)
            # delta1 = dy1_dv1 @ de_dy1

            # # (4, 5) * (5, 4)
            # de_dw1 = delta1 @ np.repeat(self.y0.T, 4, axis = 1)
            # print(np.repeat(self.y0.T, 4, axis = 1))

    def predict(self, x):
        # (5, 1)
        self.y0 = np.append(np.reshape(x, (4, 1)), [[1]], 0) 

        #(4, 5) * (5, 1) = (4, 1)
        self.v1 = self.w1 @ self.y0

        # (5, 1)
        self.y1 = np.append(sigmoid(self.v1), [[1]], 0)
        
        #(1, 5) * (5, 1) = (1, 1)
        self.v2 = self.w2 @ self.y1

        # (1, 1)
        self.y2 = sigmoid(self.v2)
        return self.y2

    def set_debug(self):
        for i in range(0, 4):
            self.w1[i,:] = 0.1 * i + 0.1
            self.w2[:,i] = 0.1 * i + 0.1

        self.w1[:,-1] = 0.5
        self.w2[:,-1] = 0.5

# n = Network()
# n.set_debug()
# print("----- W1 -----")
# print(n.w1)
# print("----- W2 -----")
# print(n.w2)

# print(n.predict([1, 1, 1, 1]))

# print("----- y0 -----")
# print(n.y0)
# print("----- v1 -----")
# print(n.v1)
# print("----- y1 -----")
# print(n.y1)
# print("----- v2 -----")
# print(n.v2)
# print("----- y2 -----")
# print(n.y2)
