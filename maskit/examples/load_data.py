from pennylane import numpy as np
from sklearn import datasets

np.random.seed(1337)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])


def load_iris(train_size=120):
    train_size = min(train_size, 150)
    data, target = datasets.load_iris(return_X_y=True)
    target = target.reshape((150, 1))
    dataset = np.concatenate((data, target), axis=1)

    np.random.shuffle(dataset)

    train, test = dataset[:train_size, :], dataset[train_size:, :]

    x_train, y_train = np.split(train, [4], axis=1)
    x_test, y_test = np.split(test, [4], axis=1)

    y_train = one_hot(y_train, 4)
    y_test = one_hot(y_test, 4)

    return x_train, y_train, x_test, y_test


def load_data(dataset, wires=None, embedding=None):
    if dataset == "iris":
        return load_iris()
    elif dataset == "mnist":
        pass
    elif dataset == "circles":
        pass
    else:
        return [None, None, None, None]
