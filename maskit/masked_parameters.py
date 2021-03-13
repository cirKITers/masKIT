import random as rand
import pennylane.numpy as np
from enum import Enum

rand.seed(1337)


class PerturbationAxis(Enum):
    #: Perturbation affects whole wires
    WIRES = 0
    #: Perturbation affects whole layers
    LAYERS = 1
    #: Perturbation affects random locations
    RANDOM = 2


class PerturbationMode(Enum):
    #: Adding new holes to the mask
    ADD = 0
    #: Removing holes from the mask
    REMOVE = 1
    #: Invert current state of the mask
    INVERT = 2


class MaskedParameters(object):
    """
    TODO: currently only works for 2d arrays
    TODO: interpretation of wires and layers is not strict and depends on user
        interpretation how the different parameters are mapped to the circuit
    """

    def __init__(
        self,
        params,
        perturbation_axis: PerturbationAxis = PerturbationAxis.RANDOM,
        seed=1337,
    ):
        rand.seed(seed)
        self._params = params
        self._mask = None
        self.perturbation_axis = perturbation_axis
        self.reset()

    @property
    def params(self):
        return self._params

    @property
    def mask(self):
        return self._mask

    @params.setter
    def params(self, values):
        self._params = values

    def reset(self):
        """
        Resets the mask to all-False.
        """
        self._mask = np.zeros_like(self._params, dtype=bool, requires_grad=False)

    def copy(self) -> "MaskedParameters":
        clone = object.__new__(MaskedParameters)
        clone._params = self._params.copy()
        clone._mask = self._mask.copy()
        clone.perturbation_axis = self.perturbation_axis
        return clone

    def perturb(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
        random: bool = True,
    ):
        assert (
            amount is None or amount >= 0
        ), "Negative values are not supported, please use PerturbationMode.REMOVE"
        if amount == 0:
            return
        if self.perturbation_axis == PerturbationAxis.WIRES:
            self._perturb_wires(amount, mode, random)
        elif self.perturbation_axis == PerturbationAxis.LAYERS:
            self._perturb_layers(amount, mode, random)
        elif self.perturbation_axis == PerturbationAxis.RANDOM:
            self._perturb_random(amount, mode, random)
        else:
            raise NotImplementedError(
                f"The perturbation {self.perturbation_axis} is not supported"
            )

    def _perturb_layers(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
        random: bool = True,
    ):
        wire_count = self._params.shape[1]
        count = abs(amount) if amount is not None else rand.randrange(0, wire_count)
        if mode == PerturbationMode.REMOVE:
            indices = [index for index, value in enumerate(self._mask[:, 0]) if value]
        elif mode == PerturbationMode.ADD:
            indices = [
                index for index, value in enumerate(self._mask[:, 0]) if not value
            ]
        else:
            indices = np.arange(wire_count)
        if len(indices) == 0:
            return
        if random:
            indices = np.random.choice(indices, min(count, len(indices)), replace=False)
        else:
            indices = indices[:count]
        self._mask[indices] = ~self._mask[indices]

    def _perturb_wires(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
        random: bool = True,
    ):
        layer_count = self._params.shape[0]
        count = abs(amount) if amount is not None else rand.randrange(0, layer_count)
        if mode == PerturbationMode.REMOVE:
            indices = [index for index, value in enumerate(self._mask[0]) if value]
        elif mode == PerturbationMode.ADD:
            indices = [index for index, value in enumerate(self._mask[0]) if not value]
        else:
            indices = np.arange(layer_count)
        if len(indices) == 0:
            return
        if random:
            layer_indices = [
                slice(None, None, None),
                np.random.choice(indices, min(count, len(indices)), replace=False),
            ]
        else:
            layer_indices = indices[:count]
        self._mask[layer_indices] = ~self._mask[layer_indices]

    def _perturb_random(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
        random: bool = True,
    ):
        count = (
            abs(amount) if amount is not None else rand.randrange(0, self._params.size)
        )
        if mode == PerturbationMode.REMOVE:
            indices = np.argwhere(self._mask)
            if len(indices) == 0:
                return
            if random:
                random_indices = tuple(
                    zip(
                        *indices[
                            np.random.choice(
                                len(indices), min(count, len(indices)), replace=False
                            )
                        ]
                    )
                )
            else:
                random_indices = tuple(zip(*indices[:count]))
        elif mode == PerturbationMode.ADD:
            indices = np.argwhere(~self._mask)
            if len(indices) == 0:
                return
            if random:
                random_indices = tuple(
                    zip(
                        *indices[
                            np.random.choice(
                                len(indices), min(count, len(indices)), replace=False
                            )
                        ]
                    )
                )
            else:
                random_indices = tuple(zip(*indices[:count]))
        else:
            indices = np.arange(self._params.size)
            if len(indices) == 0:
                return
            if random:
                selection = np.random.choice(
                    indices, min(count, len(indices)), replace=False
                )
            else:
                selection = indices[:count]
            random_indices = np.unravel_index(selection, self._mask.shape)
        self._mask[random_indices] = ~self._mask[random_indices]

    def apply_mask(self, params):
        params = np.array(params[0])
        return params[~self._mask]


if __name__ == "__main__":
    parameter = MaskedParameters(np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])))
