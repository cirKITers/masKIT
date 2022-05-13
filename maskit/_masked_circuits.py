import pennylane.numpy as np

from typing import Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from maskit._masks import (
    Mask,
    DropoutMask,
    PerturbationAxis as Axis,
    PerturbationMode as Mode,
)

Self = TypeVar("Self")


class MaskedCircuit(object):
    """
    A MaskedCircuit supports masking of different components including wires, layers,
    and parameters.
    Masking naturally removes active parameters from a circuit. However, some optimisers
    expect the array of parameters to remain stable across iteration steps;
    use ``dynamic_parameters=False`` to force the mask to always yield the full set of
    parameters in such cases.
    The mask will still prevent modification of inactive parameters.

    :param parameters: Initial parameter set for circuit
    :param layers: Number of layers
    :param wires: Number of wires
    :param dynamic_parameters: Whether the array of differentiable parameters may
        change size/order
    :param default_value: Default value for gates that are added back in. In case of
        `None` that is also the default, the last known value is assumed
    :param masks: List of tuples of `PerturbationAxis` and `Mask` types to specify
        which masks are supported, defaults to `None`
    """

    __slots__ = (
        "parameters",
        "default_value",
        "_dynamic_parameters",
        "masks",
        "wires",
        "layers",
    )

    def __init__(
        self,
        parameters: np.ndarray,
        layers: int,
        wires: int,
        masks: Optional[Iterable[Tuple[Axis, Type[Mask]]]] = None,
        dynamic_parameters: bool = True,
        default_value: Optional[float] = None,
    ):
        assert (
            layers == parameters.shape[0]
        ), "First dimension of parameters shape must be equal to number of layers"
        assert (
            wires == parameters.shape[1]
        ), "Second dimension of parameters shape must be equal to number of wires"
        self.wires = wires
        self.layers = layers
        self.parameters = parameters
        self.default_value = default_value
        self._dynamic_parameters = dynamic_parameters

        self.masks: Dict[Axis, Dict[Type[Mask], Mask]] = {}
        if masks is not None:
            for axis, mask_type in masks:
                if axis == Axis.PARAMETERS:
                    shape = parameters.shape
                elif axis == Axis.WIRES:
                    shape = (wires,)
                elif axis == Axis.LAYERS:
                    shape = (layers,)
                else:
                    # ENTANGLING axis currently is not supported as the shape cannot
                    #   be inferred automatically
                    raise NotImplementedError(
                        f"The perturbation {axis} is not supported"
                    )
                self.masks.setdefault(axis, {})[mask_type] = mask_type(
                    shape=shape, parent=self
                )  # TODO: initial mask is missing

    def register_mask(self, axis: Axis, mask: Mask) -> None:
        """
        Registers a given `mask` for a given ``PerturbationAxis`` and the appropriate
        type of `Mask`.

        :param axis: The `PerturbationAxis` the mask is applied to
        :param mask: The mask to register
        """
        assert type(mask) not in self.masks.get(
            axis, {}
        ), f"{type(mask)} for {axis} already set: {self.masks[axis][type(mask)]}"
        self.masks.setdefault(axis, {})[type(mask)] = mask

    def mask(self, axis: Axis, mask_type: Type[Mask] = DropoutMask) -> Mask:
        """
        Returns the stored mask for a given `axis` and `mask_type`.

        :param axis: The axis the mask is performed on
        :param mask_type: The type of `Mask` to return
        """
        try:
            return self.masks[axis][mask_type]
        except KeyError as err:
            mask_types = list(self.masks[axis].keys())
            if len(mask_types) > 0:
                return mask_type(shape=self.masks[axis][mask_types[0]].shape)
            if axis == Axis.WIRES:
                return mask_type(shape=(self.wires,))
            elif axis == Axis.LAYERS:
                return mask_type(shape=(self.layers,))
            elif axis == Axis.PARAMETERS:
                return mask_type(shape=self.parameters.shape)
            else:
                raise ValueError from err

    def _accumulated_mask(self, for_differentiable=True) -> np.ndarray:
        result = np.zeros(
            shape=self.parameters.shape,
            dtype=bool if for_differentiable is True else float,
        )
        for mask_type in {
            key
            for value in self.masks.values()
            for key in value
            if key.relevant_for_differentiation is for_differentiable
        }:
            result = result + self.full_mask(mask_type)
        return result

    @property
    def differentiable_parameters(self) -> np.ndarray:
        """Subset of parameters that are not masked and therefore differentiable."""
        if self._dynamic_parameters:
            return self.parameters[~self._accumulated_mask()]
        return self.parameters

    @differentiable_parameters.setter
    def differentiable_parameters(self, value) -> None:
        """
        Provides a setter for the differentiable parameters. It is ensured that the
        updated values are written into the underlying :py:attr:`~.parameters`.
        """
        mask = self._accumulated_mask()
        if self._dynamic_parameters:
            self.parameters[~mask] = value
        else:
            self.parameters[~mask] = value[~mask]

    def full_mask(self, mask_type: Type[Mask]) -> np.ndarray:
        """
        Accumulated mask of layer, wire, and parameter masks for a given type of Mask.
        Note that this mask is readonly.
        """
        try:
            result = self.masks[Axis.PARAMETERS][mask_type].mask.copy()
        except KeyError:
            result = np.zeros(
                self.parameters.shape,
                dtype=mask_type.dtype,
                requires_grad=False,
            )
        shape = result.shape
        if Axis.WIRES in self.masks and mask_type in self.masks[Axis.WIRES]:
            wires = self.masks[Axis.WIRES][mask_type].mask
            # create new axes to match the shape to properly broadcast values
            result = (
                result
                + wires[(np.newaxis, slice(None), *[np.newaxis] * (len(shape) - 2))]
            )
        if Axis.LAYERS in self.masks and mask_type in self.masks[Axis.LAYERS]:
            layers = self.masks[Axis.LAYERS][mask_type].mask
            # create new axes to match the shape to properly broadcast values
            result = result + layers[(slice(None), *[np.newaxis] * (len(shape) - 1))]
        return result

    def active(self) -> int:
        """
        Number of active gates in the circuit based on layer, wire, and parameter mask.
        Entangling gates are not included.
        """
        mask = self.full_mask(mask_type=DropoutMask)
        return mask.size - np.sum(mask)

    def perturb(
        self,
        axis: Axis = Axis.PARAMETERS,
        amount: Optional[Union[int, float]] = None,
        mode: Mode = Mode.INVERT,
        mask: Type[Mask] = DropoutMask,
    ):
        """
        Perturbs the MaskedCircuit for a given ``axis`` that is of type
        :py:class:`~.Axis`. The perturbation is applied ``amount``times
        and depends on the given ``mode`` of type :py:class:`~.PerturbationMode`.
        If no amount is given, that is ``amount=None``, a random ``amount`` is
        determined given by the actual size of the py:attr:`~.mask`. The ``amount``
        is automatically limited to the actual size of the py:attr:`~.mask`.

        :param amount: Number of items to perturb, defaults to None
        :param axis: On which axis to perturb
        :param mode: How to perturb, defaults to PerturbationMode.INVERT
        :param mask: On which mask type to perturb
        :raises NotImplementedError: Raised in case of an unknown mode
        """
        assert mode in list(
            Mode
        ), f"The selected perturbation mode {mode} is not supported."
        if amount == 0:
            return
        try:
            self.masks[axis][mask].perturb(amount=amount, mode=mode)
        except KeyError:
            raise ValueError(
                f"The mask {mask} on axis {axis} is not supported"
            ) from None

    def shrink(
        self, axis: Axis = Axis.LAYERS, amount: int = 1, mask: Type[Mask] = DropoutMask
    ):
        """
        Shrinks the `mask` by a specified `amount` on a given `axis`.

        :param axis: The `PerturbationAxis` the mask is applied on
        :param amount: The number of elements to shrink
        :param mask: The type of `Mask` to consider
        """
        try:
            self.masks[axis][mask].shrink(amount)
        except KeyError:
            raise ValueError(
                f"The mask {mask} on axis {axis} is not supported"
            ) from None

    def clear(self):
        """Resets all masks."""
        for masks in self.masks.values():
            for mask in masks.values():
                mask.clear()

    def apply_mask(self, values: np.ndarray):
        """
        Applies the encapsulated py:attr:`~.mask`s to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        value_mask = self._accumulated_mask(for_differentiable=False)
        diff_mask = self._accumulated_mask()
        result = values + value_mask
        return result[~diff_mask]

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
            for axis, registered_masks in self.masks.items():
                for registered_mask in registered_masks.values():
                    if mask == registered_mask:
                        if axis == Axis.WIRES:
                            self.parameters[:, np_indices] = self.default_value
                        elif axis == Axis.LAYERS:
                            self.parameters[np_indices, :] = self.default_value
                        elif axis == Axis.PARAMETERS:
                            self.parameters[np_indices] = self.default_value
                        else:
                            raise ValueError(f"The mask {mask} is not registered")

    def copy(self: Self) -> Self:
        """Returns a copy of the current MaskedCircuit."""
        clone = object.__new__(type(self))
        clone.wires = self.wires
        clone.layers = self.layers
        clone.parameters = self.parameters.copy()
        clone.default_value = self.default_value
        clone._dynamic_parameters = self._dynamic_parameters
        clone.masks = {}
        for axis, masks in self.masks.items():
            for mask_type, mask in masks.items():
                try:
                    clone.masks[axis][mask_type] = mask.copy(clone)
                except KeyError:
                    clone.masks[axis] = {mask_type: mask.copy(clone)}
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
        diff_mask = self._accumulated_mask()
        value_mask = self._accumulated_mask(for_differentiable=False)
        result = self.parameters.astype(object)
        if self._dynamic_parameters:
            result[~diff_mask] = changed_parameters.flatten()
        else:
            result[~diff_mask] = changed_parameters[~diff_mask]
        result = result + value_mask
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
        for layer, layer_hidden in enumerate(self.mask(Axis.LAYERS)):
            if first_layer:
                result.append("[")
                first_layer = False
            else:
                result.append("\n [")
            first_wire = True
            first_value = True
            for wire, wire_hidden in enumerate(self.mask(Axis.WIRES)):
                if isinstance(
                    self.mask(Axis.PARAMETERS)[layer][wire].unwrap(),
                    np.ndarray,
                ):
                    if first_wire:
                        result.append("[")
                        first_wire = False
                    else:
                        result.append("\n  [")
                    first_value = True
                    for parameter, parameter_hidden in enumerate(
                        self.mask(Axis.PARAMETERS)[layer][wire]
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
                        layer_hidden
                        or wire_hidden
                        or self.mask(Axis.PARAMETERS)[layer][wire]
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

    @classmethod
    def full_circuit(
        cls,
        parameters: np.ndarray,
        layers: int,
        wires: int,
        dynamic_parameters: bool = True,
        default_value: Optional[float] = None,
        entangling_mask: Optional[Mask] = None,
        wire_mask: Optional[Mask] = None,
        layer_mask: Optional[Mask] = None,
        parameter_mask: Optional[Mask] = None,
        mask_type: Type[Mask] = DropoutMask,
    ):
        """Helper to create a full circuit with `DropoutMasks`."""
        initializable_masks = [axis for axis in Axis if axis != Axis.ENTANGLING]
        for axis, mask in (
            (Axis.WIRES, wire_mask),
            (Axis.LAYERS, layer_mask),
            (Axis.PARAMETERS, parameter_mask),
        ):
            if mask is not None:
                initializable_masks.remove(axis)
        circuit = cls(
            parameters=parameters,
            layers=layers,
            wires=wires,
            masks=[(axis, mask_type) for axis in initializable_masks],
            dynamic_parameters=dynamic_parameters,
            default_value=default_value,
        )
        for axis, mask in (
            (Axis.WIRES, wire_mask),
            (Axis.LAYERS, layer_mask),
            (Axis.PARAMETERS, parameter_mask),
        ):
            if mask is not None:
                circuit.register_mask(axis=axis, mask=mask)
        if entangling_mask is not None:
            assert layers == entangling_mask.shape[0]
            circuit.register_mask(Axis.ENTANGLING, mask=entangling_mask)
        return circuit


if __name__ == "__main__":
    parameter = MaskedCircuit.full_circuit(
        np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])), 3, 3
    )
    parameter.mask(Axis.WIRES)[1] = True
    print(parameter)
