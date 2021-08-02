import tensorflow as tf
import collections
from sklearn.decomposition import PCA
from pennylane import numpy as np
from sklearn.preprocessing import minmax_scale

np.random.seed(42)

MAX_TRAIN_SAMPLES = 11471


def reduce_image(x):
    x = np.reshape(x, [1, 28, 28, 1])
    x = tf.image.resize(x, [4, 4])
    x = np.reshape(x, [4, 4])
    return x / 255


def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    for x, y in zip(xs, ys):
        mapping[str(x)].add(y)

    return zip(*((x, y) for x, y in zip(xs, ys) if len(mapping[str(x)]) == 1))


def convert_to_binary(x):
    x_new = []
    for a in x:
        c = (a < 0.5).astype(int)
        c = np.expand_dims(c, axis=0)
        x_new.append(c)

    x_new = np.concatenate(x_new)
    print(x_new.shape)
    return x_new


def convert_label(y, classes):
    if y == classes[1]:
        return 1.0
    else:
        return 0.0


def apply_PCA(wires, x_train):
    pca = PCA(n_components=wires)
    pca.fit(x_train)
    return pca


def load_mnist(wires, params):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_size = min(params["train_size"], MAX_TRAIN_SAMPLES)

    classes = []
    if "classes" in params:
        classes = params["classes"]

    x_train, y_train = zip(*((x, y) for x, y in zip(x_train, y_train) if y in classes))
    x_test, y_test = zip(*((x, y) for x, y in zip(x_test, y_test) if y in classes))

    x_train = [reduce_image(x) for x in x_train]
    x_test = [reduce_image(x) for x in x_test]

    x_train, y_train = remove_contradicting(x_train, y_train)
    x_test, y_test = remove_contradicting(x_test, y_test)

    x_train, y_train = x_train[:train_size], y_train[:train_size]

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

    c = list(zip(x_train, y_train))
    np.random.shuffle(c)
    x_train, y_train = zip(*c)

    n_x_train = len(x_train)
    data_combined = np.concatenate((np.array(x_train), np.array(x_test)))

    data_combined = minmax_scale(data_combined, (0, 2 * np.pi))
    x_train = data_combined[:n_x_train][:]
    x_test = data_combined[n_x_train:][:]

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    data_params = {"wires": 10, "embedding": None, "classes": [6, 9], "train_size": 120}
    train_data, train_target, test_data, test_target = load_mnist(10, data_params)
