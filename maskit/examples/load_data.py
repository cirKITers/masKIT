from maskit.examples.load_iris import load_iris
from maskit.examples.load_mnist import load_mnist


def load_data(dataset, wires=None, embedding=None, params=None):
    if dataset == "iris":
        return load_iris(wires, params)
    elif dataset == "mnist":
        return load_mnist(wires, params)
    elif dataset == "circles":
        pass
    else:
        return [None, None, None, None]
