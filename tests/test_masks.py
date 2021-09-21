import random
import pytest

import pennylane.numpy as pnp

from maskit._masks import DropoutMask, ValueMask, PerturbationMode


class TestDropoutMask:
    def test_init(self):
        size = 3
        with pytest.raises(AssertionError):
            DropoutMask((size,), mask=pnp.array([True, True, False, False]))
        with pytest.raises(AssertionError):
            DropoutMask((size,), mask=pnp.array([0, 1, 3]))
        preset = [False, True, False]
        mp = DropoutMask((size,), mask=pnp.array(preset))
        assert pnp.array_equal(mp.mask, preset)

    def test_setting(self):
        size = 3
        mp = DropoutMask((size,))
        assert mp
        assert len(mp.mask) == mp.mask.size
        assert pnp.sum(mp.mask) == 0
        mp[1] = True
        assert mp[1] == True  # noqa: E712
        with pytest.raises(IndexError):
            mp[size] = True
        assert pnp.sum(mp.mask) == 1
        mp.clear()
        assert pnp.sum(mp.mask) == 0
        mp[:] = True
        result = mp[:]
        assert len(result) == size
        assert pnp.all(result)
        assert pnp.sum(mp.mask) == size
        mp.clear()
        with pytest.raises(IndexError):
            mp[1, 2] = True

    def test_wrong_mode(self):
        mp = DropoutMask((3,))
        with pytest.raises(NotImplementedError):
            mp.perturb(mode=10, amount=1)

    def test_perturbation(self):
        size = 3
        mp = DropoutMask((size,))

        for i in range(1, size + 1):
            mp.perturb(i)
            mp.perturb(i, mode=PerturbationMode.RESET)
            assert pnp.sum(mp.mask) == 0

    def test_percentage_perturbation(self):
        size = 3
        mp = DropoutMask((size,))

        for i in [0.01, 0.1, 0.5, 0.9]:
            mp.perturb(amount=i)
            assert pnp.sum(mp.mask) == round(i * mp.mask.size)
            mp.clear()

    def test_wrong_percentage_perturbation(self):
        size = 3
        mp = DropoutMask((size,))

        for i in [1.1, 1.5, 3.1]:
            mp.perturb(amount=i)
            assert pnp.sum(mp.mask) == round(i)
            mp.clear()

    def test_negative_perturbation(self):
        mp = DropoutMask((3,))
        with pytest.raises(AssertionError):
            mp.perturb(amount=-1)

    def test_perturbation_remove_add(self):
        size = 3
        mp = DropoutMask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.RESET)
            assert pnp.sum(mp.mask) == 0
            mp.perturb(amount=amount, mode=PerturbationMode.SET)
            assert pnp.sum(mp.mask) == min(amount, size)
            mp.clear()

    def test_perturbation_invert_remove(self):
        size = 3
        mp = DropoutMask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.INVERT)
            reversed_amount = pnp.sum(mp.mask).unwrap()  # unwrap tensor
            mp.perturb(amount=reversed_amount, mode=PerturbationMode.RESET)
            assert pnp.sum(mp.mask) == 0

    def test_perturbation_add_remove(self):
        size = 3
        mp = DropoutMask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.SET)
            assert pnp.sum(mp.mask) == min(amount, size)
            mp.perturb(amount=amount, mode=PerturbationMode.RESET)
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "mode",
        [
            (PerturbationMode.SET, PerturbationMode.INVERT),
            (PerturbationMode.INVERT, PerturbationMode.INVERT),
        ],
    )
    def test_perturbation_mode(self, mode):
        size = 3
        mp = DropoutMask((size,))

        for amount in [0, size, size + 1]:
            mp.perturb(amount=amount, mode=mode[0])
            mp.perturb(amount=amount, mode=mode[1])
            assert pnp.sum(mp.mask) == 0

    def test_shrink(self):
        size = 3
        mp = DropoutMask((size,))

        for amount in range(size + 1):
            mp[:] = True
            mp.shrink(amount)
            assert pnp.sum(mp.mask) == size - amount

    def test_shrink_nd(self):
        size = 3
        mp = DropoutMask((size, size - 1))
        for amount in range(mp.mask.size + 1):
            mp[:] = True
            mp.shrink(amount)
            assert pnp.sum(mp.mask) == mp.mask.size - amount

    def test_copy(self):
        size = 3
        mp = DropoutMask((size,))
        new_mp = mp.copy()
        mp[0] = True
        assert pnp.sum(mp.mask) > pnp.sum(new_mp.mask)
        assert pnp.sum(new_mp.mask) == 0

    def test_apply_mask(self):
        size = 3
        mp = DropoutMask((size,))
        with pytest.raises(IndexError):
            mp.apply_mask(pnp.ones((size - 1, size)))
        mp.mask[1] = True
        result = mp.apply_mask(pnp.ones((size,), dtype=bool))
        assert pnp.sum(mp.mask) == 1
        assert pnp.sum(result) == size - 1


class TestValueMask:
    def test_init(self):
        vm = ValueMask((3,))
        assert vm
        assert vm.mask.dtype == float

    def test_apply_mask(self):
        size = 3
        mp = ValueMask((size,))
        with pytest.raises(ValueError):
            mp.apply_mask(pnp.ones((size - 1,)))
        mp.mask[1] = 1
        result = mp.apply_mask(pnp.ones((size,), dtype=float))
        assert pnp.sum(mp.mask) == 1
        assert pnp.sum(result) == size + 1
        mp.mask[1] = -1
        result = mp.apply_mask(pnp.ones((size,), dtype=float))
        assert pnp.sum(result) == size - 1

    def test_perturbation(self):
        size = 3
        mp = ValueMask((size,))

        for i in range(1, size + 1):
            mp.perturb(i, mode=PerturbationMode.SET, value=1)
            mp.perturb(i, mode=PerturbationMode.RESET)
            assert pnp.sum(mp.mask) == 0

    def test_shrink(self):
        size = 3
        mp = ValueMask((size,))

        for amount in range(size + 1):
            mp[:] = 1
            mp.shrink(amount)
            assert pnp.sum(mp.mask) == size - amount
