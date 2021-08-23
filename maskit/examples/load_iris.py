from pennylane import numpy as np
from sklearn import datasets
from maskit.examples.utils import one_hot

np.random.seed(42)


def load_iris(wires, params):
    train_size = 150
    if "train_size" in params:
        train_size = min(params["train_size"], 150)
    data, target = datasets.load_iris(return_X_y=True)
    target = target.reshape((150, 1))
    dataset = np.concatenate((data, target), axis=1)

    if "shuffle" in params and params["shuffle"]:
        np.random.shuffle(dataset)

    train, test = dataset[:train_size, :], dataset[train_size:, :]

    x_train, y_train = np.split(train, [4], axis=1)
    x_test, y_test = np.split(test, [4], axis=1)

    print(y_train)
    y_train = one_hot(y_train, 4)
    print(y_train)
    y_test = one_hot(y_test, 4)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    data_params = {"wires": 10, "embedding": None, "classes": [6, 9], "train_size": 120}
    train_data, train_target, test_data, test_target = load_iris(10, data_params)
    print(train_data)
    print(train_target)
