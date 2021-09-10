from sklearn import datasets
from pennylane import numpy as np
from maskit.examples.utils import one_hot, Data


def load_circles(train_size, test_size, shuffle):
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

    data = Data(x_train, y_train, x_test, y_test)

    return data
