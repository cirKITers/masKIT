import numpy as np
import pytest
from sklearn.metrics import log_loss

from maskit.utils import cross_entropy


@pytest.mark.parametrize(
    "predictions,targets",
    [
        pytest.param(
            np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]]),
            np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        ),
        pytest.param(np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])),
        pytest.param(
            np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.0, 0.0, 1.0, 0.0])
        ),
        pytest.param(
            np.array([0.01, 0.01, 0.01, 0.96]), np.array([0.0, 0.0, 0.0, 1.0])
        ),
    ],
)
def test_cross_entropy(predictions, targets):
    if predictions.ndim == 1:
        expected_value = log_loss([targets], [predictions])
    else:
        expected_value = log_loss(targets, predictions)
    calculated_value = cross_entropy(predictions=predictions, targets=targets)
    assert np.isclose(expected_value, calculated_value)
