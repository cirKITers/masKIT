import random as rand
import pennylane.numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


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


class Mask(object):
    """
    A Mask encapsulates a :py:attr:`~.mask` storing boolean value if a specific value
    is masked or not. In case a specific position is `True`, the according value is
    masked, otherwise it is not.
    """

    __slots__ = ("mask",)

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.mask = np.zeros(shape, dtype=bool, requires_grad=False)

    def __setitem__(self, key, value: bool):
        """
        Convenience function to set the value of a specific position of the
        encapsulated :py:attr:`~.mask`.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            self.mask[key] = value
        else:
            raise NotImplementedError(f"key {key}")

    def __getitem__(self, key):
        """
        Convenience function to get the value of a specific position of the
        encapsulated :py:attr:`~.mask`.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            return self.mask[key]
        raise NotImplementedError(f"key {key}")

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask` to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        return values[~self.mask]

    def clear(self) -> None:
        """Resets the mask to not mask anything."""
        self.mask = np.zeros_like(self.mask, dtype=bool, requires_grad=False)

    def perturb(
        self,
        amount: Optional[Union[int, float]] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        """
        Perturbs the Mask by the given ``mode`` of type :py:class:`~.PerturbationMode`
        ``amount`` times. If no amount is given or ``amount=None``, a random ``amount``
        is determined given by the actual size of the py:attr:`~.mask`. If ``amount``
        is smaller than `1`, it is interpreted as the fraction of the py:attr:`~.mask`s
        size.
        Note that the ``amount`` is automatically limited to the actual size of the
        py:attr:`~.mask`.

        :param amount: Number of items to perturb given either by an absolute amount
            when amount >= 1 or a fraction of the mask, defaults to None
        :param mode: How to perturb, defaults to PerturbationMode.INVERT
        :raises NotImplementedError: Raised in case of an unknown mode
        """
        assert (
            amount is None or amount >= 0
        ), "Negative values are not supported, please use PerturbationMode.REMOVE"
        if amount is not None:
            if amount < 1:
                amount *= self.mask.size
            amount = round(amount)
        count = abs(amount) if amount is not None else rand.randrange(0, self.mask.size)
        if count == 0:
            return
        if mode == PerturbationMode.ADD:
            indices = np.argwhere(~self.mask)
        elif mode == PerturbationMode.INVERT:
            indices = np.array([list(index) for index in np.ndindex(*self.mask.shape)])
        elif mode == PerturbationMode.REMOVE:
            indices = np.argwhere(self.mask)
        else:
            raise NotImplementedError(f"The perturbation mode {mode} is not supported")
        if indices.size == 0:
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
        self.mask[indices] = ~self.mask[indices]

    def shrink(self, amount: int = 1):
        index = np.argwhere(self.mask)
        index = index[:amount]
        if index.size > 0:
            self.mask[tuple(zip(*index))] = False

    def copy(self) -> "Mask":
        """Returns a copy of the current Mask."""
        clone = object.__new__(type(self))
        clone.mask = self.mask.copy()
        return clone


class MaskedCircuit(object):
    """
    A MaskedCircuit supports masking of different components including wires, layers,
    and parameters.
    """

    __slots__ = (
        "_layer_mask",
        "_wire_mask",
        "_parameter_mask",
        "parameters",
    )

    def __init__(self, parameters: np.ndarray, layers: int, wires: int):
        assert (
            layers == parameters.shape[0]
        ), "First dimension of parameters shape must be equal to number of layers"
        assert (
            wires == parameters.shape[1]
        ), "Second dimension of parameters shape must be equal to number of wires"
        self.parameters = parameters
        self._parameter_mask = Mask(shape=parameters.shape)
        self._layer_mask = Mask(shape=(layers,))
        self._wire_mask = Mask(shape=(wires,))

    @property
    def mask(self) -> np.ndarray:
        """
        Accumulated mask of layer, wire, and parameter masks.
        Note that this mask is readonly.
        """
        mask = self.parameter_mask.copy()
        mask[self.layer_mask, :] = True
        mask[:, self.wire_mask] = True
        return mask

    def active(self) -> int:
        """Number of active gates in the circuit."""
        mask = self.mask
        return mask.size - np.sum(mask)

    @property
    def layer_mask(self):
        """Returns the encapsulated layer mask."""
        return self._layer_mask.mask

    @property
    def wire_mask(self):
        """Returns the encapsulated wire mask."""
        return self._wire_mask.mask

    @property
    def parameter_mask(self):
        """Returns the encapsulated parameter mask."""
        return self._parameter_mask.mask

    def perturb(
        self,
        axis: PerturbationAxis = PerturbationAxis.RANDOM,
        amount: Optional[Union[int, float]] = None,
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
        :raises NotImplementedError: Raised in case of an unknown mode
        """
        assert mode in list(
            PerturbationMode
        ), f"The selected perturbation mode {mode} is not supported."
        if amount == 0:
            return
        if axis == PerturbationAxis.LAYERS:
            self._layer_mask.perturb(amount=amount, mode=mode)
        elif axis == PerturbationAxis.WIRES:
            self._wire_mask.perturb(amount=amount, mode=mode)
        elif axis == PerturbationAxis.RANDOM:  # Axis is on parameters
            self._parameter_mask.perturb(amount=amount, mode=mode)
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")

    def shrink(self, axis: PerturbationAxis = PerturbationAxis.LAYERS, amount: int = 1):
        if axis == PerturbationAxis.LAYERS:
            self._layer_mask.shrink(amount)
        elif axis == PerturbationAxis.WIRES:
            self._wire_mask.shrink(amount)
        elif axis == PerturbationAxis.RANDOM:
            self._parameter_mask.shrink(amount)
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")

    def clear(self):
        """Resets all masks."""
        self._layer_mask.clear()
        self._wire_mask.clear()
        self._parameter_mask.clear()

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask`s to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        return values[~self.mask]

    def copy(self) -> "MaskedCircuit":
        """Returns a copy of the current MaskedCircuit."""
        clone = object.__new__(type(self))
        clone._parameter_mask = self._parameter_mask.copy()
        clone._layer_mask = self._layer_mask.copy()
        clone._wire_mask = self._wire_mask.copy()
        clone.parameters = self.parameters.copy()
        return clone

    @staticmethod
    def execute(masked_circuit: "MaskedCircuit", operations: List[Dict]):
        # TODO: add check for supported operations and error handling
        result = masked_circuit
        if operations is not None:
            for operation_dict in operations:
                for operation, parameters in operation_dict.items():
                    value = result.__getattribute__(operation)(**parameters)
                    if value is not None:
                        result = value
        return result

    def __repr__(self) -> str:
        def format_value(value):
            return f"{value: .8f}"

        length = 0
        first_layer = True
        result = ["["]
        for layer, layer_hidden in enumerate(self.layer_mask):
            if first_layer:
                result.append("[")
                first_layer = False
            else:
                result.append("\n [")
            first_wire = True
            first_value = True
            for wire, wire_hidden in enumerate(self.wire_mask):
                if isinstance(self.parameter_mask[layer][wire].unwrap(), np.ndarray):
                    if first_wire:
                        result.append("[")
                        first_wire = False
                    else:
                        result.append("\n  [")
                    first_value = True
                    for parameter, parameter_hidden in enumerate(
                        self.parameter_mask[layer][wire]
                    ):
                        if not (layer_hidden or wire_hidden or parameter_hidden):
                            value = format_value(
                                self.parameters[layer][wire][parameter]
                            )
                            length = len(value)
                        else:
                            value = "{placeholder}"
                        if first_value:
                            result.append(value)
                            first_value = False
                        else:
                            result.append(f" {value}")
                    result += "]"
                else:
                    if not (
                        layer_hidden or wire_hidden or self.parameter_mask[layer][wire]
                    ):
                        value = format_value(self.parameters[layer][wire])
                        length = len(value)
                    else:
                        value = "{placeholder}"
                    if first_value:
                        result.append(value)
                        first_value = False
                    else:
                        result.append(f" {value}")
            result.append("]")
        result.append("]")
        return "".join(result).format(placeholder="-" * length)


if __name__ == "__main__":
    parameter = MaskedCircuit(
        np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])), 3, 3
    )
    parameter.wire_mask[1] = True
    print(parameter)
