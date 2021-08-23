from sklearn import datasets
from pennylane import numpy as np
from maskit.examples.utils import one_hot

DEFAULT_TRAIN_SIZE = 150
DEFAULT_TEST_SIZE = 50


def load_circles(wires, params):
    train_size, test_size = DEFAULT_TRAIN_SIZE, DEFAULT_TEST_SIZE
    shuffle = True
    if "train_size" in params:
        train_size = params["train_size"]
    if "test_size" in params:
        test_size = params["test_size"]
    if "shuffle" in params:
        shuffle = params["shuffle"]

    # TODO: currently hardcoded to 4 to fit with iris
    # num_classes = 2
    num_classes = 4

    x, y = datasets.make_circles(n_samples=train_size + test_size, shuffle=shuffle)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    y_train, y_test = np.reshape(y_train, [len(y_train), 1]), np.reshape(
        y_test, [len(y_test), 1]
    )
    y_train, y_test = one_hot(y_train, num_classes), one_hot(y_test, num_classes)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    data_params = {"wires": 10, "embedding": None, "classes": [6, 9], "train_size": 120}
    train_data, train_target, test_data, test_target = load_circles(10, data_params)
    print(train_data)
    print(train_target)
