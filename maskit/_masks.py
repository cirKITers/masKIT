import random as rand
import pennylane.numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

if TYPE_CHECKING:
    from maskit._masked_circuits import MaskedCircuit


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

    :param shape: The shape of the mask
    :param parent: `MaskedCircuit` that owns the mask
    :param mask: Preset of values that is taken by mask
    """

    __slots__ = ("mask", "values", "_parent")

    def __init__(
        self,
        shape: Tuple[int, ...],
        parent: Optional["MaskedCircuit"] = None,
        mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.mask = np.zeros(shape, dtype=bool, requires_grad=False)
        if mask is not None:
            assert mask.dtype == bool, "Mask must be of type bool"
            assert mask.shape == shape, "Shape of mask must be equal to shape"
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
