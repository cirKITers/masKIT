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
    #: Perturbation affects random locations in parameter mask
    RANDOM = 2


class PerturbationMode(Enum):
    #: Adding new holes to the mask
    ADD = 0
    #: Removing holes from the mask
    REMOVE = 1
    #: Invert current state of the mask
    INVERT = 2


class MaskedObject(object):
    """
    A MaskedObject encapsulates a :py:attr:`~.mask` storing boolean value if
    a specific value is masked or not. In case a specific position is `True`,
    the according value is masked, otherwise it is not.
    """

    __slots__ = ("_mask",)

    def __setitem__(self, key, value: bool):
        """
        Convenience function to set the value of a specific position of the
        encapsulated :py:attr:`~.mask`.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            self._mask[key] = value
        else:
            raise NotImplementedError(f"key {key}")

    def __getitem__(self, key):
        """
        Convenience function to get the value of a specific position of the
        encapsulated :py:attr:`~.mask`.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            return self._mask[key]
        raise NotImplementedError(f"key {key}")

    @property
    def mask(self):
        """
        Returns the encapsulated :py:attr:`~.mask`
        """
        return self._mask

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask` to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the mask to not mask anything."""
        self._mask = np.zeros_like(self._mask, dtype=bool, requires_grad=False)

    def perturb(
        self,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        """
        Perturbs the MaskedObject by the given ``mode`` of type
        :py:class:`~.PerturbationMode` ``amount`` times. If no amount is given,
        that is ``amount=None``, a random ``amount`` is determined given by the
        actual size of the py:attr:`~.mask`. The ``amount`` is automatically
        limited to the actual size of the py:attr:`~.mask`.

        :param amount: Number of items to perturb, defaults to None
        :param mode: How to perturb, defaults to PerturbationMode.INVERT
        :raises AssertionError: Raised for negative amounts
        :raises NotImplementedError: Raised in case of an unknown mode
        """
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
        """Returns a copy of the current MaskedObject."""
        clone = object.__new__(type(self))
        clone._mask = self._mask.copy()
        return clone


class MaskedParameter(MaskedObject):
    """
    A MaskedParameter encapsulates not only the :py:attr:`~.mask` but also the
    according :py:attr:`~.parameters` being masked.
    """
    __slots__ = ("_parameters",)
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
    """
    Logical representation of a MaskedLayer.
    """

    def __init__(self, layers: int):
        super().__init__()
        self._mask = np.zeros(layers, dtype=bool, requires_grad=False)


class MaskedWire(MaskedObject):
    """
    Logical representation of a MaskedWire.
    """

    def __init__(self, wires: int):
        super().__init__()
        self._mask = np.zeros(wires, dtype=bool, requires_grad=False)


class MaskedCircuit(object):
    """
    A MaskedCircuit supports masking of different components including wires,
    layers, and parameters.
    """

    def __init__(self, parameters: np.ndarray, layers: int, wires: int):
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
    def mask(self) -> np.ndarray:
        """
        Accumulated mask of layer, wire, and parameter masks.
        Note that this mask is readonly.
        """
        mask = self.parameter_mask.copy()
        mask[self.layer_mask] = True
        mask[:, self.wire_mask] = True
        return mask

    @property
    def layer_mask(self):
        """Returns the encapsulated layer mask."""
        return self._layers.mask

    @property
    def wire_mask(self):
        """Returns the encapsulated wire mask."""
        return self._wires.mask

    @property
    def parameter_mask(self):
        """Returns the encapsulated parameter mask."""
        return self._parameters.mask

    def perturb(
        self,
        axis: PerturbationAxis,
        amount: Optional[int] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        """
        Perturbs the MaskedCircuit for a given ``axis`` that is of type
        :py:class:`~.PerturbationAxis`. The perturbation is applied ``amount``times
        and depends on the given ``mode`` of type :py:class:`~.PerturbationMode`.
        If no amount is given, that is ``amount=None``, a random ``amount`` is
        determined given by the actual size of the py:attr:`~.mask`. The ``amount``
        is automatically limited to the actual size of the py:attr:`~.mask`.

        :param amount: Number of items to perturb, defaults to None
        :param axis: Which mask to perturb
        :param mode: How to perturb, defaults to PerturbationMode.INVERT
        :raises AssertionError: Raised for negative amounts
        :raises NotImplementedError: Raised in case of an unknown mode
        """
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

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask`s to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        :raises AssertionError: In case the shape of values and mask don't match.
        """
        assert values.shape == self.parameter_mask.shape, "The given shape must match"
        return values[~self.mask]

    def copy(self) -> "MaskedCircuit":
        """Returns a copy of the current MaskedCircuit."""
        clone = object.__new__(type(self))
        clone._parameters = self._parameters.copy()
        clone._layers = self._layers.copy()
        clone._wires = self._wires.copy()
        return clone


if __name__ == "__main__":
    parameter = MaskedParameter(np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])))
