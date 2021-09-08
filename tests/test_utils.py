import numpy as np

from maskit.utils import cross_entropy


def test_cross_entropy():
    predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]])
    targets = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    expected_value = 0.71355817782  # Correct answer
    calculated_value = cross_entropy(predictions=predictions, targets=targets)
    assert np.isclose(expected_value, calculated_value)
