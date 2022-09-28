import numpy as np

def sse(d, y):
    return np.sum(np.square(d - y)) / 2

def sse_prime(d, y):
    return -np.sum(d - y)
