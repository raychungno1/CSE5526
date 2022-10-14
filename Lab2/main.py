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
            bases, variance, kmm_error = kmm.train(train)

            rbf = RBF(bases, variance)
            rbf_error = rbf.train(train, labels, lr)

            plt.clf()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(kmm_error)
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("SSE")
            ax1.set_title("K-Means Error")
            ax2.plot(rbf_error)
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("SSE")
            ax2.set_title("RBF Error")
            plt.suptitle(f"Bases: {b} | Learning Rate: {lr}")
            plt.savefig(f"./plots/base-{b}-lr-{lr}-error.png")

            x = np.arange(0, 1, 0.01)
            y = 0.5 + 0.4 * np.sin(2 * np.pi * x)
            prediction = [rbf.predict(p) for p in x]

            plt.clf()
            plt.plot(x, y, linestyle="--", label="Original Function")
            plt.plot(x, prediction, label="RBF Function")
            plt.plot(train, labels, linestyle="None", marker="o",
                     markersize=4, label="Data Point")
            plt.legend()
            plt.title(f"Bases: {b} | Learning Rate: {lr}")
            plt.savefig(f"./plots/base-{b}-lr-{lr}.png")
