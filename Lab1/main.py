import time
import numpy as np
from network import Network
from data_gen import data_gen

np.random.seed(0)
np.set_printoptions(precision=3)

if __name__ == "__main__":

    train, labels = data_gen()

    n = Network()
    w1 = np.copy(n.w1)
    b1 = np.copy(n.b1)
    w2 = np.copy(n.w2)
    b2 = np.copy(n.b2)

    print("----- w1 -----")
    print(n.w1)
    print("----- b1 -----")
    print(n.b1)
    print("----- w2 -----")
    print(n.w2)
    print("----- b2 -----")
    print(n.b2)
    
    for lr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:

        start = time.time()
        epoch = n.train(train, labels, lr)
        end = time.time()

        print(
            f"Learning Rate: {lr}\tEpoch: {epoch}\t Time: {(end - start)}")
        n.w1 = np.copy(w1)
        n.b1 = np.copy(b1)
        n.w2 = np.copy(w2)
        n.b2 = np.copy(b2)
        
    print("----- w1 -----")
    print(n.w1)
    print("----- b1 -----")
    print(n.b1)
    print("----- w2 -----")
    print(n.w2)
    print("----- b2 -----")
    print(n.b2)
