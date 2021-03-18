import random as rand
import pennylane.numpy as np
from enum import Enum
from typing import Optional

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


class MaskedObject(object):
    __slots__ = "_mask"

    def __setitem__(self, key, value: bool):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            self._mask[key] = value
        else:
            raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            return self._mask[key]
        raise NotImplementedError

    @property
    def mask(self):
        return self._mask

    def apply_mask(self, values):
        raise NotImplementedError

    def reset(self):
        """Resets the mask to not mask anything."""
        self._mask = np.zeros_like(self._mask, dtype=bool, requires_grad=False)

    def perturb(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        assert (
            amount is None or amount >= 0
        ), "Negative values are not supported, please use PerturbationMode.REMOVE"
        if amount == 0:
            return
        count = (
            abs(amount) if amount is not None else rand.randrange(0, self._mask.size)
        )
        if mode == PerturbationMode.ADD:
            indices = np.argwhere(~self._mask)
        elif mode == PerturbationMode.INVERT:
            indices = np.array([list(index) for index in np.ndindex(*self._mask.shape)])
        elif mode == PerturbationMode.REMOVE:
            indices = np.argwhere(self._mask)
        else:
            raise NotImplementedError(f"The perturbation mode {mode} is not supported")
        if len(indices) == 0:
            return
        indices = tuple(
            zip(
                *indices[
                    np.random.choice(
                        len(indices), min(count, len(indices)), replace=False
                    )
                ]
            )
        )
        self._mask[indices] = ~self._mask[indices]

    def copy(self) -> "MaskedObject":
        clone = object.__new__(type(self))
        clone._mask = self._mask.copy()
        return clone


class MaskedParameter(MaskedObject):
    def __init__(self, parameters):
        super().__init__()
        self._parameters = parameters
        self._mask = np.zeros_like(parameters, dtype=bool, requires_grad=False)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        self._parameters = values

    def copy(self) -> "MaskedParameter":
        clone = super().copy()
        clone._parameters = self._parameters.copy()
        return clone


class MaskedLayer(MaskedObject):
    def __init__(self, layers: int):
        super().__init__()
        self._mask = np.zeros(layers, dtype=bool, requires_grad=False)


class MaskedWire(MaskedObject):
    def __init__(self, wires: int):
        super().__init__()
        self._mask = np.zeros(wires, dtype=bool, requires_grad=False)


class MaskedCircuit(object):
    """
    A MaskedCircuit supports masking of different components including wires,
    layers, and parameters.
    """

    def __init__(self, parameters, layers: int, wires: int):
        assert (
            layers == parameters.shape[0]
        ), "First dimension of parameters shape must be equal to number of layers"
        assert (
            wires == parameters.shape[1]
        ), "Second dimension of parameters shape must be equal to number of wires"
        self._parameters = MaskedParameter(parameters)
        self._layers = MaskedLayer(layers=layers)
        self._wires = MaskedWire(wires=wires)

    @property
    def parameters(self):
        return self._parameters.parameters

    @parameters.setter
    def parameters(self, values):
        self._parameters.parameters = values

    @property
    def mask(self):
        mask = self.parameter_mask.copy()
        mask[self.layer_mask] = True
        mask[:, self.wire_mask] = True
        return mask

    @property
    def layer_mask(self):
        return self._layers.mask

    @property
    def wire_mask(self):
        return self._wires.mask

    @property
    def parameter_mask(self):
        return self._parameters.mask

    def perturb(
        self,
        axis: PerturbationAxis,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        assert (
            amount is None or amount >= 0
        ), "Negative values are not supported, please use PerturbationMode.REMOVE"
        assert mode in list(PerturbationMode), "The selected mode is not supported."
        if amount == 0:
            return
        if axis == PerturbationAxis.LAYERS:
            mask_fn = self._mask_layers
        elif axis == PerturbationAxis.WIRES:
            mask_fn = self._mask_wires
        elif axis == PerturbationAxis.RANDOM:  # Axis is on parameters
            mask_fn = self._mask_parameters
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")
        mask_fn(amount=amount, mode=mode)

    def _mask_layers(self, amount, mode):
        self._layers.perturb(amount=amount, mode=mode)

    def _mask_wires(self, amount, mode):
        self._wires.perturb(amount=amount, mode=mode)

    def _mask_parameters(self, amount, mode):
        self._parameters.perturb(amount=amount, mode=mode)

    def reset(self):
        """Resets all masks."""
        self._layers.reset()
        self._wires.reset()
        self._parameters.reset()

    def apply_mask(self, params):
        """
        Applies the masks for wires, layers, and parameters to the given instance of
        parameters.

        :param params: Parameters to apply the mask to
        :type params: [type]
        """
        assert params.shape == self.parameter_mask.shape, "The given shape must match"
        return params[~self.mask]

    def copy(self) -> "MaskedCircuit":
        clone = object.__new__(type(self))
        clone._parameters = self._parameters.copy()
        clone._layers = self._layers.copy()
        clone._wires = self._wires.copy()
        return clone


if __name__ == "__main__":
    parameter = MaskedParameter(np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])))
