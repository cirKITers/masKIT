from pennylane import numpy as np
from typing import NamedTuple, Optional


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])


def pad_data(data: np.ndarray, axis: int, padding: int) -> np.ndarray:
    """
    Function pads 0 data to the end of the given axis.

    :param data: The data to pad 0 data to
    :param axis: Axis to pad on
    :param padding: How many elements to pad
    """
    size = data.shape[0]
    if size == 0:
        return data.reshape((size, data.shape[axis] + padding))
    return np.append(data, [[0] * padding for _ in range(size)], axis)


class DataSet(NamedTuple):
    train_data: Optional[np.ndarray]
    train_target: Optional[np.ndarray]
    validation_data: Optional[np.ndarray]
    validation_target: Optional[np.ndarray]
    test_data: Optional[np.ndarray]
    test_target: Optional[np.ndarray]
