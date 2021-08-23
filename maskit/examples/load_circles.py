from sklearn import datasets

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

    x, y = datasets.make_circles(n_samples=train_size + test_size, shuffle=shuffle)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    data_params = {"wires": 10, "embedding": None, "classes": [6, 9], "train_size": 120}
    train_data, train_target, test_data, test_target = load_circles(10, data_params)
    print(train_data)
    print(train_target)
