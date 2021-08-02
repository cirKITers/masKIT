import collections
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])


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
