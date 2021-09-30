from typing import List, Tuple
import tensorflow as tf
from sklearn.decomposition import PCA
from pennylane import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.utils import shuffle as skl_shuffle

from maskit.datasets.utils import DataSet

# ignore filtering on classes
MAX_TRAIN_SAMPLES = 60000
MAX_TEST_SAMPLES = 10000


def reduce_image(x: np.ndarray) -> np.ndarray:
    x = np.reshape(x, [1, 28, 28, 1])
    x = tf.image.resize(x, [4, 4])
    x = np.reshape(x, [4, 4])
    return x / 255


def convert_label(y: int, classes: List[int]) -> List[float]:
    assert y in classes
    return [1.0 if the_class == y else 0.0 for the_class in classes]


def apply_PCA(wires: int, x_train: np.ndarray):
    pca = PCA(n_components=wires)
    pca.fit(x_train)
    return pca


def downscale(x_data, y_data, size) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function does several things including the reduction of image size,
    conversion to binary as well as removal of duplicates.

    :param x_data: Input data
    :param y_data: Target data
    :param size: Maximum number of data to prepare
    """
    data_index: List[bool] = np.zeros(len(y_data), dtype=bool)
    result_x: List[np.ndarray] = []
    result_x_set = set()
    for index, image in enumerate(x_data):
        if len(result_x) >= size:
            break
        reduced_image = reduce_image(image)
        previous_len = len(result_x_set)
        result_x_set.add(str(reduced_image))
        if len(result_x_set) <= previous_len:  # next if contradicting
            continue
        result_x.append(reduced_image)
        data_index[index] = True
    return (np.array(result_x), y_data[data_index])


def prepare_data(
    x_data: np.ndarray, y_data: np.ndarray, size, classes
) -> Tuple[np.ndarray, np.ndarray]:
    data_index = np.isin(y_data, classes)
    x_data, y_data = x_data[data_index], y_data[data_index]
    x_data, y_data = downscale(x_data, y_data, size)
    y_data = np.array([convert_label(y, classes) for y in y_data])
    try:
        x_data = np.reshape(
            x_data, (x_data.shape[0], x_data.shape[1] * x_data.shape[2])
        )
    except IndexError:
        pass
    return (x_data, y_data)


def mnist(
    wires: int = 4,
    classes=(6, 9),
    train_size: int = 100,
    validation_size: int = 0,
    test_size: int = 50,
    shuffle: bool = True,
) -> DataSet:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_size = min(train_size, MAX_TRAIN_SAMPLES - validation_size)
    validation_size = min(validation_size, MAX_TRAIN_SAMPLES - train_size)
    test_size = min(test_size, MAX_TEST_SAMPLES)

    # split validation set from train set
    split_point = len(x_train) - (5000 if validation_size > 0 else 0)
    x_train, y_train, x_validation, y_validation = (
        x_train[:split_point],
        y_train[:split_point],
        x_train[split_point:],
        y_train[split_point:],
    )

    x_train, y_train = prepare_data(x_train, y_train, train_size, classes)
    x_validation, y_validation = prepare_data(
        x_validation, y_validation, validation_size, classes
    )
    x_test, y_test = prepare_data(x_test, y_test, test_size, classes)

    if shuffle:
        x_train, y_train = skl_shuffle(x_train, y_train)
        # TODO: also shuffle validation data?

    pca = apply_PCA(wires, x_train)
    x_train = pca.transform(x_train)
    x_train = minmax_scale(x_train, (0, 2 * np.pi))
    if len(x_validation) > 0:
        x_validation = pca.transform(x_validation)
        x_validation = minmax_scale(x_validation, (0, 2 * np.pi))
    x_test = pca.transform(x_test)
    x_test = minmax_scale(x_test, (0, 2 * np.pi))

    return DataSet(x_train, y_train, x_validation, y_validation, x_test, y_test)


if __name__ == "__main__":
    import timeit

    print(
        timeit.timeit(
            "mnist(validation_size=50)", setup="from __main__ import mnist", number=1
        )
    )
