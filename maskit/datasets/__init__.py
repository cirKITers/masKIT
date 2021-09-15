from maskit.datasets.circles import circles
from maskit.datasets.iris import iris
from maskit.datasets.mnist import mnist
from maskit.datasets.utils import Data


def load_data(dataset: str, **kwargs):
    """
    Returns the data for the requested ``dataset``.

    :param dataset: Name of the chosen dataset.
        Available datasets are: iris, mnist and circles.
    :param kwargs: Further arguments for dataset selection. Possible arguments:
        train_size (int): Size of the training dataset
        test_size (int): Size of the testing dataset
        shuffle (bool): if the dataset should be shuffled
        classes (list[int]): which numbers of the mnist dataset should be included
        wires (int): number of wires in the circuit
    :raises ValueError: Raised if a not supported dataset is requested
    """
    train_size = kwargs.get("train_size", 100)
    test_size = kwargs.get("test_size", 50)
    shuffle = kwargs.get("shuffle", True)
    classes = kwargs.get("classes", [6, 9])
    wires = kwargs.get("wires", 4)

    if dataset == "simple":
        return Data(None, None, None, None)
    elif dataset == "iris":
        return iris(train_size, test_size, shuffle)
    elif dataset == "mnist":
        return mnist(wires, classes, train_size, test_size, shuffle)
    elif dataset == "circles":
        return circles(train_size, test_size, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
