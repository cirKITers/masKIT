from pennylane import numpy as np


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])
