import tensorflow as tf
import collections
import math
from sklearn.decomposition import PCA
from pennylane import numpy as np
from sklearn.preprocessing import minmax_scale
from maskit.datasets.utils import Data

MAX_TRAIN_SAMPLES = 11471
MAX_TEST_SAMPLES = 1952


def reduce_image(x: np.ndarray) -> np.ndarray:
    x = np.reshape(x, [1, 28, 28, 1])
    x = tf.image.resize(x, [4, 4])
    x = np.reshape(x, [4, 4])
    return x / 255


def remove_contradicting(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    mapping = collections.defaultdict(set)
    for x, y in zip(xs, ys):
        mapping[str(x)].add(y)

    return zip(*((x, y) for x, y in zip(xs, ys) if len(mapping[str(x)]) == 1))


def convert_to_binary(x: np.ndarray) -> np.ndarray:
    x_new = []
    for a in x:
        c = (a < 0.5).astype(int)
        c = np.expand_dims(c, axis=0)
        x_new.append(c)

    x_new = np.concatenate(x_new)
    return x_new


def convert_label(y: int, classes: list) -> list:
    assert y in classes
    # Measuring n qubits gets 2^n results to compare against this vector
    num_classes = nearest_power_of_two(len(classes))
    a = [0.0 for i in range(num_classes)]
    a[classes.index(y)] = 1.0
    return a


def apply_PCA(wires: int, x_train: np.ndarray):
    pca = PCA(n_components=wires)
    pca.fit(x_train)
    return pca


def nearest_power_of_two(x: int) -> int:
    return 2 ** (math.ceil(math.log(x, 2)))


def mnist(wires=4, classes=(6, 9), train_size=100, test_size=50, shuffle=True) -> Data:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_size = min(train_size, MAX_TRAIN_SAMPLES)
    test_size = min(test_size, MAX_TEST_SAMPLES)

    x_train, y_train = zip(*((x, y) for x, y in zip(x_train, y_train) if y in classes))
    x_test, y_test = zip(*((x, y) for x, y in zip(x_test, y_test) if y in classes))

    x_train = [reduce_image(x) for x in x_train]
    x_test = [reduce_image(x) for x in x_test]

    x_train, y_train = remove_contradicting(x_train, y_train)
    x_test, y_test = remove_contradicting(x_test, y_test)

    x_train, y_train = x_train[:train_size], y_train[:train_size]
    x_test, y_test = x_test[:test_size], y_test[:test_size]

    x_train = convert_to_binary(x_train)
    x_test = convert_to_binary(x_test)
    y_train = [convert_label(y, classes) for y in y_train]
    y_test = [convert_label(y, classes) for y in y_test]

    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    )
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    pca = apply_PCA(wires, x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    if shuffle:
        c = list(zip(x_train, y_train))
        np.random.shuffle(c)
        x_train, y_train = zip(*c)

    n_x_train = len(x_train)
    data_combined = np.concatenate((np.array(x_train), np.array(x_test)))

    data_combined = minmax_scale(data_combined, (0, 2 * np.pi))
    x_train = data_combined[:n_x_train][:]
    x_test = data_combined[n_x_train:][:]

    y_train, y_test = np.array(y_train), np.array(y_test)

    return Data(x_train, y_train, x_test, y_test)
