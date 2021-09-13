from pennylane import numpy as np
from sklearn import datasets
from maskit.datasets.utils import one_hot, Data

MAX_SAMPLES = 150


def iris(train_size=100, test_size=50, shuffle=True):
    train_size = min(train_size, MAX_SAMPLES)
    if train_size + test_size > MAX_SAMPLES:
        test_size = MAX_SAMPLES - train_size
    data, target = datasets.load_iris(return_X_y=True)
    target = target.reshape((train_size + test_size, 1))
    dataset = np.concatenate((data, target), axis=1)

    if shuffle:
        np.random.shuffle(dataset)

    train, test = (
        dataset[:train_size, :],
        dataset[train_size:, :],
    )

    x_train, y_train = np.split(train, [4], axis=1)
    x_test, y_test = np.split(test, [4], axis=1)

    y_train = one_hot(y_train, 4)
    y_test = one_hot(y_test, 4)

    data = Data(x_train, y_train, x_test, y_test)

    return data
