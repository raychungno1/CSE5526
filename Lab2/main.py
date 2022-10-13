import numpy as np
import matplotlib.pyplot as plt
from Kmeans import Kmeans

from data_gen import data_gen

if __name__ == "__main__":
    np.random.seed(0)
    train, labels = data_gen()

    model = Kmeans(5)
    bases, variance, error = model.train(train, labels)
    print(bases)
    print(variance)
    print(error)

    plt.clf()
    x = np.arange(0, 1, 0.01)
    y = 0.5 + 0.4 * np.sin(2 * np.pi * x)
    plt.vlines(bases, 0, 1, alpha=0.2)
    plt.vlines(bases + variance, 0, 1, linestyle="--", alpha=0.2)
    plt.vlines(bases - variance, 0, 1, linestyle="--", alpha=0.2)

    plt.plot(x, y, linestyle="--")
    plt.plot(train, labels, linestyle="None", marker="o")
    plt.plot(train, np.zeros(train.shape), linestyle="None", marker="o")
    plt.title(f"No Noise")
    plt.show()
    # plt.savefig(f"./sse-plots/momentum-{lr}.png")
