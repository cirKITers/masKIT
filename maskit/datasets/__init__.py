from typing import Tuple
from pennylane import numpy as np

from maskit.datasets.circles import circles
from maskit.datasets.iris import iris
from maskit.datasets.mnist import mnist
from maskit.datasets.utils import DataSet


def load_data(
    dataset: str,
    train_size: int = 100,
    test_size: int = 50,
    shuffle: bool = True,
    classes: Tuple[int, ...] = (6, 9),
    wires: int = 4,
    target_length: int = 16,
) -> DataSet:
    """
    Returns the data for the requested ``dataset``.

    :param dataset: Name of the chosen dataset.
        Available datasets are: iris, mnist and circles.
    :param train_size: Size of the training dataset
    :param test_size: Size of the testing dataset
    :param shuffle: if the dataset should be shuffled
    :param classes: which numbers of the mnist dataset should be included
    :param wires: number of wires in the circuit
    :param target_length: Normalised length of target arrays

    :raises ValueError: Raised if a not supported dataset is requested
    """
    if dataset == "iris":
        result = iris(train_size, test_size, shuffle)
    elif dataset == "mnist":
        result = mnist(wires, classes, train_size, test_size, shuffle)
    elif dataset == "circles":
        result = circles(train_size, test_size, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    difference = target_length - result.train_target.shape[1]
    assert difference >= 0, (
        f"Target length ({target_length}) must support at least "
        f"{result.train_target.shape[1]} classes"
    )
    if difference > 0:
        # extend train and test target arrays
        size = result.train_target.shape[0]
        try:
            new_train_target = np.append(
                result.train_target,
                [[0] * difference for _ in range(size)],
                1,
            )
        except ValueError:
            new_train_target = result.train_target
        size = result.test_target.shape[0]
        try:
            new_test_target = np.append(
                result.test_target,
                [[0] * difference for _ in range(size)],
                1,
            )
        except ValueError:
            new_test_target = result.test_target
        result = DataSet(
            result.train_data, new_train_target, result.test_data, new_test_target
        )
    return result
