from maskit.examples.circles import circles
from maskit.examples.iris import iris
from maskit.examples.mnist import mnist


def load_data(dataset, **kwargs):
    train_size = kwargs.get("train_size", 100)
    test_size = kwargs.get("test_size", 50)
    shuffle = kwargs.get("shuffle", True)
    classes = kwargs.get("classes", [6, 9])
    wires = kwargs.get("wires", 4)

    if dataset == "simple":
        return [None, None, None, None]
    elif dataset == "iris":
        return iris(train_size, test_size, shuffle)
    elif dataset == "mnist":
        return mnist(wires, classes, train_size, test_size, shuffle)
    elif dataset == "circles":
        return circles(train_size, test_size, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
