import numpy as np


data = np.array([0.1, 0.3, 0.7, 0.9])
bases = np.array([0.3, 0.7])
labels = np.argmin(np.square(data - bases[:, np.newaxis]), 0)
new_bases = np.bincount(labels, data) / np.bincount(labels)

print(data)
print(labels)
print(new_bases)
print(np.sum(np.min(np.square(data - new_bases[:, np.newaxis]), 0)))

# dist = np.square(data - bases[:, np.newaxis])
# print(np.argmin(dist, 0))
