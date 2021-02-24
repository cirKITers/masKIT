import random
import pennylane.numpy as np

random.seed(1337)


class MaskedParameters(object):
    def __init__(self, params):
        self._params = params
        self._mask = np.zeros_like(params, dtype=bool)

    @property
    def params(self):
        return self._params

    @property
    def mask(self):
        return self._mask

    @params.setter
    def params(self, values):
        self._params = values

    def copy(self) -> 'MaskedParameters':
        clone = object.__new__(MaskedParameters)
        clone._params = self._params.copy()
        clone._mask = self._mask.copy()
        return clone

    def perturb(self, amount=None):
        count = amount if amount is not None else random.randrange(0, self._params.size)
        if amount and amount < 0:
            indices = np.argwhere(self._mask)
        else:
            indices = np.random.choice(self._params.size, count, replace=False)
        self._mask[np.unravel_index(indices, self._mask.shape)] = ~self._mask[np.unravel_index(indices, self._mask.shape)]
