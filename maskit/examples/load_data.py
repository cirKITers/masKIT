from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
import utils
import tensorflow as tf

np.random.seed(1337)


def load_iris(train_size=120):
    train_size = min(train_size, 150)
    data, target = datasets.load_iris(return_X_y=True)
    target = target.reshape((150, 1))
    dataset = np.concatenate((data, target), axis=1)

    np.random.shuffle(dataset)

    train, test = dataset[:train_size, :], dataset[train_size:, :]

    x_train, y_train = np.split(train, [4], axis=1)
    x_test, y_test = np.split(test, [4], axis=1)

    y_train = utils.one_hot(y_train, 4)
    y_test = utils.one_hot(y_test, 4)

    return x_train, y_train, x_test, y_test


def load_mnist(wires, params):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    classes = []
    if "classes" in params:
        classes = params["classes"]

    x_train, y_train = zip(*((x, y) for x, y in zip(x_train, y_train) if y in classes))
    x_test, y_test = zip(*((x, y) for x, y in zip(x_test, y_test) if y in classes))

    x_train = [utils.reduce_image(x) for x in x_train]
    x_test = [utils.reduce_image(x) for x in x_test]

    x_train, y_train = utils.remove_contradicting(x_train, y_train)
    x_test, y_test = utils.remove_contradicting(x_test, y_test)

    x_train = utils.convert_to_binary(x_train)
    x_test = utils.convert_to_binary(x_test)
    y_train = [utils.convert_label(y, classes) for y in y_train]
    y_test = [utils.convert_label(y, classes) for y in y_test]

    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    )
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    pca = utils.apply_PCA(wires, x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    c = list(zip(x_train, y_train))
    np.random.shuffle(c)
    x_train, y_train = zip(*c)

    n_x_train = len(x_train)
    data_combined = np.concatenate((np.array(x_train), np.array(x_test)))

    data_combined = minmax_scale(data_combined, (0, 2 * np.pi))
    x_train = data_combined[:n_x_train][:]
    x_test = data_combined[n_x_train:][:]

    return x_train, y_train, x_test, y_test


def load_data(dataset, wires=None, embedding=None, params=None):
    if dataset == "iris":
        return load_iris()
    elif dataset == "mnist":
        load_mnist(wires, params)
    elif dataset == "circles":
        pass
    else:
        return [None, None, None, None]
