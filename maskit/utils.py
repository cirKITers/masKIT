from pennylane import numpy as np

from maskit.optimizers import ExtendedOptimizers


def check_params(train_params):
    assert train_params["dataset"] in ["simple", "iris", "mnist", "circles"]
    assert isinstance(train_params["optimizer"], ExtendedOptimizers)
    if "interpret" not in train_params:
        train_params["interpret"] = tuple(
            range(2 ** len(train_params.get("wires_to_measure", (0,))))
        )


def cross_entropy(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Cross entropy calculation between :py:attr:`targets` (encoded as one-hot vectors)
    and :py:attr:`predictions`. Predictions are normalized to sum up to `1.0`.

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
    assert (
        predictions.shape == targets.shape
    ), f"Shape of predictions {predictions.shape} must match targets {targets.shape}"
    current_sum = np.sum(predictions, axis=predictions.ndim - 1)

    if predictions.ndim == 1:
        sample_count = 1
        predictions = predictions / current_sum
    else:
        sample_count = predictions.shape[0]
        predictions = predictions / current_sum[:, np.newaxis]

    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -np.sum(targets * np.log(predictions)) / sample_count
