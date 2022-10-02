import time
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from data_gen import data_gen

if __name__ == "__main__":

    train, labels = data_gen()

    np.random.seed(0)
    n = Network()
    print("----- INITIAL WEIGHTS -----\n")
    print("----- w1 -----")
    print(n.w1)
    print("----- b1 -----")
    print(n.b1.T)
    print("----- w2 -----")
    print(n.w2)
    print("----- b2 -----")
    print(n.b2)
    print("\n---------------------------\n")

    for lr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        np.random.seed(0)
        n = Network()

        start = time.time()
        epoch, error_hist = n.train(train, labels, lr, True)
        end = time.time()

        print(f"Learning Rate: {lr} | Epoch: {epoch} | Time: {(end - start)}")

        plt.clf()
        plt.plot(error_hist)
        plt.title(f"SSE | Learning Rate: {lr}")
        plt.savefig(f"./sse-plots/momentum-{lr}.png")

    print("\n----- FINAL WEIGHTS -----\n")
    print("----- w1 -----")
    print(n.w1)
    print("----- b1 -----")
    print(n.b1.T)
    print("----- w2 -----")
    print(n.w2)
    print("----- b2 -----")
    print(n.b2)
    print("\n-------------------------\n")

    print("----- VERIFYING NN -----")
    for x, d in zip(train, labels):
        y = n.predict(x)[0][0]
        print(f"Data: {str(x)} | Pred: {y} | Lbl: {d} | Diff: {abs(d - y)}")
