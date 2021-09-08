from pennylane import numpy as np

from maskit.optimizers import ExtendedOptimizers


def check_params(train_params):
    assert train_params["dataset"] in ["simple", "iris"]
    assert isinstance(train_params["optimizer"], ExtendedOptimizers)


def cross_entropy(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Cross entropy calculation between :py:attr:`targets` (encoded as one-hot vectors)
    and :py:attr:`predictions`.

    .. note::

        The implementation of this function is based on the discussion on
        `StackOverflow <https://stackoverflow.com/a/47398312/10138546>`_.

        Due to ArrayBoxes that are required for automatic differentiation, we currently
        use this implementation instead of implementations provided by sklearn for
        example.

    :param predictions: Predictions in same order as targets. In case predictions for
        several samples are given, the weighted cross entropy is returned.
    :param targets: Ground truth labels for supplied samples.
    :param epsilon: Amount to clip predictions as log is not defined for `0` and `1`.
    """
    assert predictions.shape == targets.shape
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    sample_count = 1 if predictions.ndim == 1 else predictions.shape[0]
    return -np.sum(targets * np.log(predictions)) / sample_count
