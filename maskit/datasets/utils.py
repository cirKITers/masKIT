from pennylane import numpy as np
from typing import NamedTuple, Optional


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])


class DataSet(NamedTuple):
    train_data: Optional[np.ndarray]
    train_target: Optional[np.ndarray]
    test_data: Optional[np.ndarray]
    test_target: Optional[np.ndarray]
