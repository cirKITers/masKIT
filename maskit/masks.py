import random as rand
import pennylane.numpy as np
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar

Self = TypeVar("Self")


class PerturbationAxis(Enum):
    #: Perturbation affects whole wires
    WIRES = 0
    #: Perturbation affects whole layers
    LAYERS = 1
    #: Perturbation affects random locations in parameter mask
    RANDOM = 2
    #: Perturbation affects entangling gates
    ENTANGLING = 3


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

    __slots__ = ("mask", "_parent")

    def __init__(
        self, shape: Tuple[int, ...], parent: Optional["MaskedCircuit"] = None
    ):
        super().__init__()
        self.mask = np.zeros(shape, dtype=bool, requires_grad=False)
        self._parent = parent

    def __len__(self) -> int:
        """Returns the len of the encapsulated :py:attr:`~.mask`"""
        return len(self.mask)

    @property
    def shape(self) -> Any:
        """Returns the shape of the encapsulated :py:attr:`~.mask`"""
        return self.mask.shape

    @property
    def size(self) -> Any:
        """Returns the size of the encapsulated :py:attr:`~.mask`"""
        return self.mask.size

    def __setitem__(self, key, value: bool):
        """
        Convenience function to set the value of a specific position of the
        encapsulated :py:attr:`~.mask`.

        Attention: when working with multi-dimensional masks please use tuple
        convention for accessing the elements as otherwise changes are not
        recognised and a `MaskedCircuit` cannot be informed about changes.

        Instead of

            .. code:
                mask[2][2] = True

        please use

            .. code:
                mask[2, 2] = True
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, tuple):
            before = self.mask.copy()
            self.mask[key] = value
            delta_indices = np.argwhere(before != self.mask)
            if self._parent is not None:
                self._parent.mask_changed(self, delta_indices)
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
        self[indices] = ~self.mask[indices]

    def shrink(self, amount: int = 1):
        index = np.argwhere(self.mask)
        index = index[:amount]
        if index.size > 0:
            self[tuple(zip(*index))] = False

    def copy(self, parent: Optional["MaskedCircuit"] = None) -> "Mask":
        """Returns a copy of the current Mask."""
        clone = object.__new__(type(self))
        clone.mask = self.mask.copy()
        clone._parent = parent
        return clone


class MaskedCircuit(object):
    """
    A MaskedCircuit supports masking of different components including wires, layers,
    and parameters.

    :param entangling_mask: The mask to apply for entangling gates within the circuit,
        defaults to None
    """

    __slots__ = (
        "_layer_mask",
        "_wire_mask",
        "_parameter_mask",
        "_entangling_mask",
        "parameters",
        "default_value",
    )

    def __init__(
        self,
        parameters: np.ndarray,
        layers: int,
        wires: int,
        default_value: Optional[float] = None,
        entangling_mask: Optional[Mask] = None,
    ):
        assert (
            layers == parameters.shape[0]
        ), "First dimension of parameters shape must be equal to number of layers"
        assert (
            wires == parameters.shape[1]
        ), "Second dimension of parameters shape must be equal to number of wires"
        self.parameters = parameters
        self._parameter_mask = Mask(shape=parameters.shape, parent=self)
        self._layer_mask = Mask(shape=(layers,), parent=self)
        self._wire_mask = Mask(shape=(wires,), parent=self)
        self.default_value = default_value
        if entangling_mask:
            assert layers == entangling_mask.shape[0]
        self._entangling_mask = entangling_mask

    @property
    def differentiable_parameters(self) -> np.ndarray:
        """Subset of parameters that are not masked and therefore differentiable."""
        return self.parameters[~self.mask]

    @differentiable_parameters.setter
    def differentiable_parameters(self, value) -> None:
        """
        Provides a setter for the differentiable parameters. It is ensured that the
        updated values are written into the underlying :py:attr:`~.parameters`.
        """
        self.parameters[~self.mask] = value

    @property
    def mask(self) -> np.ndarray:
        """
        Accumulated mask of layer, wire, and parameter masks.
        Note that this mask is readonly.
        """
        mask = self.parameter_mask.mask.copy()
        mask[self.layer_mask.mask, :] = True
        mask[:, self.wire_mask.mask] = True
        return mask

    def active(self) -> int:
        """
        Number of active gates in the circuit based on layer, wire, and parameter mask.
        Entangling gates are not included.
        """
        mask = self.mask
        return mask.size - np.sum(mask)

    @property
    def layer_mask(self):
        """Returns the encapsulated layer mask."""
        return self._layer_mask

    @property
    def wire_mask(self):
        """Returns the encapsulated wire mask."""
        return self._wire_mask

    @property
    def parameter_mask(self):
        """Returns the encapsulated parameter mask."""
        return self._parameter_mask

    @property
    def entangling_mask(self):
        """Returns the encapsulated mask of entangling gates."""
        return self._entangling_mask

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
        elif axis == PerturbationAxis.ENTANGLING:
            if self._entangling_mask:
                self._entangling_mask.perturb(amount=amount, mode=mode)
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")

    def shrink(self, axis: PerturbationAxis = PerturbationAxis.LAYERS, amount: int = 1):
        if axis == PerturbationAxis.LAYERS:
            self._layer_mask.shrink(amount)
        elif axis == PerturbationAxis.WIRES:
            self._wire_mask.shrink(amount)
        elif axis == PerturbationAxis.RANDOM:
            self._parameter_mask.shrink(amount)
        elif axis == PerturbationAxis.ENTANGLING:
            if self._entangling_mask:
                self._entangling_mask.shrink(amount)
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")

    def clear(self):
        """Resets all masks."""
        self._layer_mask.clear()
        self._wire_mask.clear()
        self._parameter_mask.clear()
        if self.entangling_mask:
            self._entangling_mask.clear()

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask`s to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        return values[~self.mask]

    def mask_changed(self, mask: Mask, indices: np.ndarray):
        """
        Callback function that is used whenever one of the encapsulated masks does
        change. In case the mask does change and adds a parameter back into the circuit,
        the configured :py:attr:`~.default_value` is applied.

        :raises NotImplementedError: In case an unimplemented mask reports change
        """
        if len(indices) == 0 or self.default_value is None:
            return
        np_indices = tuple(zip(*indices))
        if not np.all(mask.mask[np_indices]):
            if self.wire_mask is mask:
                self.parameters[:, np_indices] = self.default_value
            elif self.layer_mask is mask:
                self.parameters[np_indices, :] = self.default_value
            elif self.parameter_mask is mask:
                self.parameters[np_indices] = self.default_value
            else:
                raise NotImplementedError(f"The mask {mask} is not supported")

    def copy(self: Self) -> Self:
        """Returns a copy of the current MaskedCircuit."""
        clone = object.__new__(type(self))
        clone._parameter_mask = self._parameter_mask.copy(clone)
        clone._layer_mask = self._layer_mask.copy(clone)
        clone._wire_mask = self._wire_mask.copy(clone)
        clone.parameters = self.parameters.copy()
        clone.default_value = self.default_value
        if self._entangling_mask:
            clone._entangling_mask = self._entangling_mask.copy(clone)
        else:
            clone._entangling_mask = None
        return clone

    def expanded_parameters(self, changed_parameters: np.ndarray) -> np.ndarray:
        """
        This method helps building a circuit with a current instance of differentiable
        parameters. Differentiable parameters are contained within a box for autograd
        e.g. for proper tracing. As from those parameters the structure of the
        circuit cannot be implied, this method takes care to expand on these parameters
        by giving a view that is a combination of parameters and the differentiable
        parameters.
        Note that the returned parameters are based on a copy of the underlying
        parameters and therefore should not be changed manually.

        :param changed_parameters: Current set of differentiable parameters
        """
        result = self.parameters.astype(object)
        result[~self.mask] = changed_parameters.flatten()
        return result

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


class FreezableMaskedCircuit(MaskedCircuit):
    """
    A FreezableMaskedCircuit not only supports masking of different components
    including wires, layers, and parameters but also supports freezing a subset
    of parameters defined again on the different components wires, layers, and
    parameters.
    """

    __slots__ = "_layer_freeze_mask", "_wire_freeze_mask", "_parameter_freeze_mask"

    def __init__(
        self,
        parameters: np.ndarray,
        layers: int,
        wires: int,
        default_value: Optional[float] = None,
        entangling_mask: Optional[Mask] = None,
    ):
        super().__init__(
            parameters,
            layers,
            wires,
            default_value=default_value,
            entangling_mask=entangling_mask,
        )
        self._parameter_freeze_mask = Mask(shape=parameters.shape)
        self._layer_freeze_mask = Mask(shape=(layers,))
        self._wire_freeze_mask = Mask(shape=(wires,))

    @property
    def mask(self) -> np.ndarray:
        """
        Accumulated mask of layers, wires, and parameters for both masking and freezing.
        Note that this mask is readonly.
        """
        base = super().mask
        base[self._parameter_freeze_mask.mask] = True
        base[self._layer_freeze_mask, :] = True
        base[:, self._wire_freeze_mask.mask] = True
        return base

    @property
    def parameter_freeze_mask(self) -> Mask:
        """Returns the encapsulated freezing parameter mask."""
        return self._parameter_freeze_mask

    @property
    def wire_freeze_mask(self) -> Mask:
        """Returns the encapsulated freezing wire mask."""
        return self._wire_freeze_mask

    @property
    def layer_freeze_mask(self) -> Mask:
        """Returns the encapsulated freezing layer mask."""
        return self._layer_freeze_mask

    def freeze(
        self,
        axis: PerturbationAxis = PerturbationAxis.LAYERS,
        amount: Optional[Union[int, float]] = None,
        mode: PerturbationMode = PerturbationMode.ADD,
    ):
        """
        Freezes the parameter values for a given ``axis`` that is of type
        :py:class:`~.PerturbationAxis`. The freezing is applied ``amount``times
        and depends on the given ``mode`` of type :py:class:`~.PerturbationMode`.
        If no amount is given, that is ``amount=None``, a random ``amount`` is
        determined given by the actual size of the py:attr:`~.mask`. The ``amount``
        is automatically limited to the actual size of the py:attr:`~.mask`.

        :param amount: Number of items to freeze, defaults to None
        :param axis: Which mask to freeze, defaults to PerturbationAxis.LAYERS
        :param mode: How to freeze, defaults to PerturbationMode.ADD
        :raises NotImplementedError: Raised in case of an unknown mode
        """
        assert mode in list(
            PerturbationMode
        ), f"The selected perturbation mode {mode} is not supported."
        if amount == 0:
            return
        if axis == PerturbationAxis.LAYERS:
            self._layer_freeze_mask.perturb(amount=amount, mode=mode)
        elif axis == PerturbationAxis.WIRES:
            self._wire_freeze_mask.perturb(amount=amount, mode=mode)
        elif axis == PerturbationAxis.RANDOM:  # Axis is on parameters
            self._parameter_freeze_mask.perturb(amount=amount, mode=mode)
        else:
            raise NotImplementedError(f"The perturbation {axis} is not supported")

    def copy(self: Self) -> Self:
        """Returns a copy of the current FreezableMaskedCircuit."""
        clone = super().copy()
        clone._parameter_freeze_mask = self._parameter_freeze_mask.copy()
        clone._layer_freeze_mask = self._layer_freeze_mask.copy()
        clone._wire_freeze_mask = self._wire_freeze_mask.copy()
        return clone


if __name__ == "__main__":
    parameter = MaskedCircuit(
        np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])), 3, 3
    )
    parameter.wire_mask[1] = True
    print(parameter)
