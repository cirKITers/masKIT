from maskit.examples.load_circles import load_circles
from maskit.examples.load_iris import load_iris
from maskit.examples.load_mnist import load_mnist


def load_data(dataset, **kwargs):
    train_size = kwargs.get("train_size", 100)
    test_size = kwargs.get("test_size", 50)
    shuffle = kwargs.get("shuffle", True)
    classes = kwargs.get("classes", [6, 9])
    wires = kwargs.get("wires", 4)

    if dataset == "iris":
        return load_iris(train_size, test_size, shuffle)
    elif dataset == "mnist":
        return load_mnist(wires, classes, train_size, test_size, shuffle)
    elif dataset == "circles":
        return load_circles(train_size, test_size, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
