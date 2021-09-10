from pennylane import numpy as np
from typing import NamedTuple


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])


class Data(NamedTuple):
    train_data: np.ndarray
    train_target: np.ndarray
    test_data: np.ndarray
    test_target: np.ndarray
