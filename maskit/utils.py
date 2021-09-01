from pennylane import numpy as np

from maskit.optimizers import ExtendedOptimizers


def check_params(train_params):
    assert train_params["dataset"] in ["simple", "iris"]
    assert isinstance(train_params["optimizer"], ExtendedOptimizers)


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) or (k) ndarray
           targets with the same dimensions as predictions
    Returns: scalar
    """
    assert(predictions.shape == targets.shape)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0] if len(predictions.shape) == 2 else 1
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce
