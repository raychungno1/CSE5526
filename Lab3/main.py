import os
import random
import numpy as np
from libsvm.svmutil import *
import matplotlib.pyplot as plt


data_dir = os.path.join(os.path.dirname(__file__), "data")
train_file = os.path.join(data_dir, "ncRNA_s.train.txt")
test_file = os.path.join(data_dir, "ncRNA_s.test.txt")
plots_dir = os.path.join(os.path.dirname(__file__), "plots")

x = [i for i in range(-4, 9)]
C = A = [2**i for i in x]

train_y, train_x = svm_read_problem(train_file)
test_y, test_x = svm_read_problem(test_file)


###
# Part 1: Linear SVM
###

def train(c):
    m = svm_train(train_y, train_x, f"-t 0 -c {c} -q")
    _, p_acc, _ = svm_predict(test_y, test_x, m, "-q")
    return p_acc[0]


accuracy = [train(c) for c in C]

fig, ax = plt.subplots(1)
ax.plot(x, accuracy, marker="o")
ax.set_xlabel("C")
ax.set_xticks(x)
ax.set_xticklabels([r"$2^{" + str(i) + r"}$"for i in x])
ax.set_ylabel("Accuracy (%)")
ax.set_title("Linear SVM Classification Accuracy")
plt.savefig(os.path.join(plots_dir, "linear-svm-accuracy.png"))


###
# Part 2: RBF SVM w/ Cross Validation
###

random.seed(0)
valid_idx = random.sample(range(len(train_x)), 1000)
valid_x = [train_x[i] for i in valid_idx]
valid_y = [train_y[i] for i in valid_idx]

accuracy = np.zeros((len(C), len(A)))
num_models = 0
for i, c in enumerate(C):
    for j, a in enumerate(A):
        avg_acc = 0
        for k in range(0, 1000, 200):
            t_x = valid_x[0:k] + valid_x[(k + 200):]
            t_y = valid_y[0:k] + valid_y[(k + 200):]

            v_x = valid_x[k:(k + 200)]
            v_y = valid_y[k:(k + 200)]

            m = svm_train(t_y, t_x, f"-t 2 -c {c} -g {a} -q")
            _, p_acc, _ = svm_predict(v_y, v_x, m, "-q")
            avg_acc += p_acc[0]
            num_models += 1

        accuracy[i, j] = avg_acc / 5

max_acc = accuracy.max()
max_c_idx, max_a_idx = np.unravel_index(accuracy.argmax(), accuracy.shape)
max_c = C[max_c_idx]
max_a = A[max_a_idx]

m = svm_train(train_y, train_x, f"-t 2 -c {max_c} -g {max_a} -q")
_, p_acc, _ = svm_predict(test_y, test_x, m, "-q")

print(accuracy)
print(f"Best Model:\tc={max_c}\ta={max_a}")
print(f"Validation Dataset Accuracy: {max_acc}")
print(f"Full Dataset Accuracy: {p_acc[0]}")
