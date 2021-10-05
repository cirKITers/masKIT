from sklearn import datasets
from maskit.datasets.utils import one_hot, DataSet


def circles(train_size=100, test_size=50, shuffle=True) -> DataSet:
    x, y = datasets.make_circles(n_samples=train_size + test_size, shuffle=shuffle)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    y_train, y_test = one_hot(y_train, 2), one_hot(y_test, 2)

    return DataSet(x_train, y_train, None, None, x_test, y_test)
