from sklearn import datasets
from maskit.examples.utils import one_hot, Data


def load_circles(train_size=100, test_size=50, shuffle=True):
    # TODO: currently hardcoded to 4 to fit with iris
    # num_classes = 2
    num_classes = 4

    x, y = datasets.make_circles(n_samples=train_size + test_size, shuffle=shuffle)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    y_train, y_test = one_hot(y_train, num_classes), one_hot(y_test, num_classes)

    data = Data(x_train, y_train, x_test, y_test)

    return data
