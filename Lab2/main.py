import numpy as np
import matplotlib.pyplot as plt
from Kmeans import Kmeans
from RBF import RBF

from data_gen import data_gen

if __name__ == "__main__":
    train, labels = data_gen()
    num_bases = [2, 4, 7, 11, 16]
    learning_rates = [0.01, 0.02]

    for b in num_bases:
        for lr in learning_rates:
            np.random.seed(0)

            kmm = Kmeans(b)
            bases, variance, error = kmm.train(train)

            rbf = RBF(bases, variance)
            rbf.train(train, labels, lr)

            x = np.arange(0, 1, 0.01)
            y = 0.5 + 0.4 * np.sin(2 * np.pi * x)
            prediction = [rbf.predict(p) for p in x]

            # for c, v, w in zip(bases, variance, rbf.w[0, :]):
            #     gaussian = [w * np.exp(-np.square(p - c) / (2 * v)) + rbf.b for p in x]
            #     plt.plot(x, gaussian, color="k", alpha=0.2)

            plt.clf()
            plt.plot(x, y, linestyle="--", label="Original Function")
            plt.plot(x, prediction, label="RBF Function")
            plt.plot(train, labels, linestyle="None", marker="o",
                     markersize=4, label="Data Point")
            plt.legend()
            plt.title(f"Bases: {b} | Learning Rate: {lr}")
            plt.savefig(f"./plots/base-{b}-lr-{lr}.png")
            plt.show
