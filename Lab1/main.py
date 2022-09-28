import numpy as np
from network import Network
from data_gen import data_gen

np.random.seed(0)
np.set_printoptions(precision=3)

if __name__ == "__main__":

    train, labels = data_gen()
    print(len(train))
    n = Network()
    # n.set_debug()

    print("----- W1 -----")
    print(n.w1)
    print("----- W2 -----")
    print(n.w2)
    n.train(train, labels, 0.5)
    print("----- W1 -----")
    print(n.w1)
    print("----- W2 -----")
    print(n.w2)
    # n.train([train[0]], [labels[0]], 1, 1)
    # for i in range(0, 4):
    #     n.layers[0][i, :] = i + 2
    # n.layers[0][:,-1] = 0
    # n.layers[1][0, :] = 2
    # n.layers[1][0, -1] = 0
    # print(n.layers)
    # p = n.predict([1, 1, 1, 1])
    # print(f"Expected: 40\tPredicted: {p}")
