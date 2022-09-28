from itertools import product
import numpy as np

def data_gen():
    train_data = np.array(list(product([0, 1], repeat=4)))
    train_lbls = [1 if np.sum(data) % 2 else 0 for data in train_data]
    return train_data, train_lbls
