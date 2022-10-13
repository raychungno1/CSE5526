import numpy as np
from numpy import ndarray


def data_gen() -> tuple[ndarray, ndarray]:
    train_data = np.array([np.random.rand() for _ in range(75)])
    train_lbls = np.array([(np.random.rand() * 0.2 - 0.1) + 0.5 + 0.4 *
                           np.sin(2 * np.pi * x) for x in train_data])
    return train_data, train_lbls
