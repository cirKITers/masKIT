from sklearn import datasets


def load_circles(wires, params):
    train_size, test_size = params["train_size"], params["test_size"]
    x, y = datasets.make_circles(
        n_samples=train_size + test_size, shuffle=params["shuffle"]
    )
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    return x_train, y_train, x_test, y_test
