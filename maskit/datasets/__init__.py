from typing import Tuple, Optional, Union

from maskit.datasets.circles import circles
from maskit.datasets.iris import iris
from maskit.datasets.mnist import mnist
from maskit.datasets.utils import DataSet, pad_data


def load_data(
    dataset: str,
    train_size: int = 100,
    test_size: int = 50,
    validation_size: int = 0,
    shuffle: Union[bool, int] = 1337,
    classes: Tuple[int, ...] = (6, 9),
    wires: int = 4,
    target_length: Optional[int] = None,
) -> DataSet:
    """
    Returns the data for the requested ``dataset``.

    :param dataset: Name of the chosen dataset.
        Available datasets are: iris, mnist and circles.
    :param train_size: Size of the training dataset
    :param test_size: Size of the testing dataset
    :param shuffle: if the dataset should be shuffled, used also as a seed
    :param classes: which numbers of the mnist dataset should be included
    :param wires: number of wires in the circuit
    :param target_length: Normalised length of target arrays

    :raises ValueError: Raised if a not supported dataset is requested
    """
    if dataset == "iris":
        result = iris(train_size=train_size, test_size=test_size, shuffle=shuffle)
    elif dataset == "mnist":
        result = mnist(
            wires=wires,
            classes=classes,
            train_size=train_size,
            test_size=test_size,
            validation_size=validation_size,
            shuffle=shuffle,
        )
    elif dataset == "circles":
        result = circles(train_size=train_size, test_size=test_size, shuffle=shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if target_length is None:
        target_length = 2**wires
    difference = target_length - result.train_target.shape[1]
    assert difference >= 0, (
        f"Target length ({target_length}) must support at least "
        f"{result.train_target.shape[1]} classes"
    )
    if difference > 0:
        # extend train, validation and test target arrays
        new_train_target = pad_data(result.train_target, 1, difference)
        new_test_target = pad_data(result.test_target, 1, difference)
        if result.validation_target is not None and len(result.validation_target) > 0:
            new_validation_target = pad_data(result.validation_target, 1, difference)
        else:
            new_validation_target = None
        result = DataSet(
            result.train_data,
            new_train_target,
            result.validation_data,
            new_validation_target,
            result.test_data,
            new_test_target,
        )
    return result
