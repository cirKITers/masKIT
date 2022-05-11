import random as rand
import pennylane.numpy as np
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Optional, Tuple, TypeVar, Union, Type

if TYPE_CHECKING:
    from maskit._masked_circuits import MaskedCircuit

T = TypeVar("T")


class PerturbationAxis(Enum):
    #: Perturbation affects whole wires
    WIRES = 0
    #: Perturbation affects whole layers
    LAYERS = 1
    #: Perturbation affects random locations in parameter mask
    PARAMETERS = 2
    #: Perturbation affects entangling gates
    ENTANGLING = 3


class PerturbationMode(Enum):
    #: Set a new value on a mask
    SET = 0
    #: Remove a value from the mask
    RESET = 1
    #: Invert current state of the mask
    INVERT = 2


class Mask(Generic[T]):
    """
    A Mask encapsulates a :py:attr:`~.mask` storing boolean value if a specific value
    is masked or not. In case a specific position is `True`, the according value is
    masked, otherwise it is not.

    :param shape: The shape of the mask
    :param parent: `MaskedCircuit` that owns the mask
    :param mask: Preset of values that is taken by mask
    """

    relevant_for_differentiation: bool = False
    #: The underlying datatype of the encapsulated mask
    dtype: Type[T]
    __slots__ = ("mask", "_parent")

    def __init__(
        self,
        shape: Tuple[int, ...],
        parent: Optional["MaskedCircuit"] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.mask = np.zeros(shape, dtype=self.dtype, requires_grad=False)
        if mask is not None:
            assert mask.dtype == self.dtype, f"Mask must be of type {self.dtype}"
            assert (
                mask.shape == shape
            ), f"Shape of mask ({mask.shape}) must be equal to {shape}"
            self.mask[:] = mask
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

    def __setitem__(self, key, value: T):
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
            if self._parent is not None:
                before = self.mask.copy()
                self.mask[key] = value
                delta_indices = np.argwhere(before != self.mask)
                self._parent.mask_changed(self, delta_indices)
            else:
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

    @abstractmethod
    def apply_mask(self, values: np.ndarray) -> np.ndarray:
        """
        Applies the encapsulated py:attr:`~.mask` to the given ``values``.
        Note that the values should have the same shape as the py:attr:`~.mask`.

        :param values: Values where the mask should be applied to
        """
        return NotImplemented

    def clear(self) -> None:
        """Resets the mask to not mask anything."""
        self.mask = np.zeros_like(self.mask, dtype=self.mask.dtype, requires_grad=False)

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
    def shrink(self, amount: int = 1):
        """
        Shrinks a Mask for ``amount`` given positions. Shrinking can be interpreted
        as setting the ``amount`` leftmost values of the Mask that are different
        from its default to its default value, e.g. to `False` in case of a boolean
        `dtype` or `0` in case of floats.

        :param amount: Number of items to reset to `0`, defaults to `1`
        """
        return NotImplemented

    def copy(self, parent: Optional["MaskedCircuit"] = None) -> "Mask":
        """Returns a copy of the current Mask."""
        clone = object.__new__(type(self))
        clone.mask = self.mask.copy()
        clone._parent = parent
        return clone

    def _count_for_amount(self, amount: Optional[Union[int, float]]) -> int:
        """
        Returns a valid number of elements with respect to the current size of the
        encapsulated :py:attr:`~.mask`. The number of elements is dependent on the
        value of ``amount``: 1) either a random number is returned in case of `None`;
        2) a specific percentage in case of an amount in the range of `[0, 1[`;
        or 3) the absolute value is given.

        :param amount: Amount of elements to consider, random if `None`, percentage
            if in the range `[0, 1[`, or absolute value otherwise
        """
        assert (
            amount is None or amount >= 0
        ), "Negative values are not supported, please use PerturbationMode.REMOVE"
        if amount is not None:
            if amount < 1:
                amount *= self.mask.size
            amount = round(amount)
        return (
            min(abs(amount), self.mask.size)
            if amount is not None
            else rand.randrange(0, self.mask.size)
        )


class DropoutMask(Mask[bool]):
    """
    A DropoutMask marks the positions that should be dropped from a given array
    and therefore an associated gate in a quantum circuit.
    A position that will not be dropped is marked as `False`, and `True` otherwise.
    """

    dtype = bool
    relevant_for_differentiation = True

    def apply_mask(self, values: np.ndarray):
        return values[~self.mask]

    def perturb(
        self,
        amount: Optional[Union[int, float]] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
    ):
        count = self._count_for_amount(amount=amount)
        if count == 0:
            return
        if mode == PerturbationMode.SET:
            indices = np.argwhere(~self.mask)
        elif mode == PerturbationMode.INVERT:
            indices = np.array([list(index) for index in np.ndindex(*self.mask.shape)])
        elif mode == PerturbationMode.RESET:
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


class FreezeMask(DropoutMask):
    """
    A FreezeMask provides the same functionality as a :py:class:`~.DropoutMask`.
    However, the meaning is a bit different. Marked positions are interpreted as
    being frozen and underlying values therefore are not intended to be changed.
    """

    pass


class ValueMask(Mask[float]):
    """
    A ValueMask shifts given values the mask is applied to with the stored values
    inside the mask.
    """

    dtype = float
    relevant_for_differentiation = False

    def apply_mask(self, values: np.ndarray):
        return values + self.mask

    def perturb(
        self,
        amount: Optional[Union[int, float]] = None,
        mode: PerturbationMode = PerturbationMode.INVERT,
        value: Optional[float] = np.pi,
    ):
        count = self._count_for_amount(amount=amount)
        if count == 0:
            return
        if mode == PerturbationMode.SET:  # INVERT might be added here
            indices = np.array([list(index) for index in np.ndindex(*self.mask.shape)])
        elif mode == PerturbationMode.RESET:
            indices = np.argwhere(self.mask != 0)
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
        if mode == PerturbationMode.SET and value is not None:
            self[indices] = value
        elif mode == PerturbationMode.RESET:
            self[indices] = 0

    def shrink(self, amount: int = 1):
        index = np.argwhere(self.mask != 0)
        index = index[:amount]
        if index.size > 0:
            self[tuple(zip(*index))] = 0
