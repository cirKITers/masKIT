from typing import Tuple

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

    :raises ValueError: Raised if a not supported dataset is requested
    """
    if dataset == "simple":
        return DataSet(None, None, None, None)
    elif dataset == "iris":
        return iris(train_size, test_size, shuffle)
    elif dataset == "mnist":
        return mnist(wires, classes, train_size, test_size, shuffle)
    elif dataset == "circles":
        return circles(train_size, test_size, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
