import random
import pytest

import pennylane.numpy as pnp

from maskit.masks import (
    MaskedCircuit,
    MaskedLayer,
    MaskedWire,
    MaskedParameter,
    PerturbationAxis,
    PerturbationMode,
)

# TODO: unit test für None
# TODO: unit test für Entfernung der Maske, wenn keine angesetzt ist
# TODO: test len indices == 0
# TODO: check if it is the row/column as selected
# TODO: test auf None


class TestMaskedCircuits:
    def test_init(self):
        mp = self._create_circuit(3)
        assert mp
        assert len(mp.wire_mask) == 3
        assert len(mp.layer_mask) == 3
        assert mp.parameter_mask.shape == (3, 3)

        size = 3
        with pytest.raises(AssertionError):
            MaskedCircuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size - 1,
                wires=size,
            )
        with pytest.raises(AssertionError):
            MaskedCircuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size + 1,
                wires=size,
            )
        with pytest.raises(AssertionError):
            MaskedCircuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size,
                wires=size - 1,
            )
        with pytest.raises(AssertionError):
            MaskedCircuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size,
                wires=size + 1,
            )

    def test_wrong_mode(self):
        mp = self._create_circuit(3)
        with pytest.raises(AssertionError):
            mp.perturb(axis=PerturbationAxis.LAYERS, mode=10)

    def test_wrong_axis(self):
        mp = self._create_circuit(3)
        with pytest.raises(NotImplementedError):
            mp.perturb(axis=10)

    @pytest.mark.parametrize("axis", list(PerturbationAxis))
    def test_reset(self, axis):
        size = 3
        mp = self._create_circuit(size)
        mp.perturb(axis=axis, amount=size)
        assert (
            pnp.sum(mp.layer_mask) + pnp.sum(mp.wire_mask) + pnp.sum(mp.parameter_mask)
            == size
        )
        mp.reset()
        assert (
            pnp.sum(mp.layer_mask) + pnp.sum(mp.wire_mask) + pnp.sum(mp.parameter_mask)
            == 0
        )

    @pytest.mark.parametrize("axis", list(PerturbationAxis))
    def test_zero_amount(self, axis):
        mp = self._create_circuit(3)
        pre_sum = (
            pnp.sum(mp.wire_mask) + pnp.sum(mp.layer_mask) + pnp.sum(mp.parameter_mask)
        )
        mp.perturb(axis=axis, amount=0)
        assert pre_sum == pnp.sum(mp.wire_mask) + pnp.sum(mp.layer_mask) + pnp.sum(
            mp.parameter_mask
        )

    def test_apply_mask(self):
        size = 3
        mp = self._create_circuit(size)
        with pytest.raises(AssertionError):
            mp.apply_mask(pnp.ones((size, size - 1)))
        mp.wire_mask[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == size
        assert pnp.sum(result) == size
        mp.layer_mask[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == 1
        assert pnp.sum(result) == 1
        mp.parameter_mask[(size - 1, size - 1)] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == 0
        assert pnp.sum(result) == 0

    def test_copy(self):
        size = 3
        mp = self._create_circuit(size)
        mp.layer_mask[0] = True
        new_mp = mp.copy()

        mp.wire_mask[0] = True
        mp.parameter_mask[0, 0] = True

        assert pnp.sum(mp.mask) > pnp.sum(new_mp.mask)
        assert pnp.sum(new_mp.wire_mask) == 0
        assert pnp.sum(new_mp.layer_mask) == pnp.sum(mp.layer_mask)
        assert pnp.sum(new_mp.parameter_mask) == 0

    def test_parameters(self):
        size = 3
        mp = self._create_circuit(size)

        new_random_values = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size)
        )
        assert (mp.parameters != new_random_values).all()
        mp.parameters = new_random_values
        assert (mp.parameters == new_random_values).all()

    def _create_circuit(self, size):
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
        return MaskedCircuit(parameters=parameters, layers=size, wires=size)


class TestMaskedObject:
    @pytest.mark.parametrize("masked_object", [MaskedLayer, MaskedWire])
    def test_setting(self, masked_object):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)
        assert mp
        assert len(mp.mask) == mp.mask.size
        assert pnp.sum(mp.mask) == 0
        mp[1] = True
        print(mp[1])
        assert mp[1] == True  # noqa: E712
        with pytest.raises(IndexError):
            mp[size] = True
        assert pnp.sum(mp.mask) == 1
        mp.reset()
        assert pnp.sum(mp.mask) == 0
        mp[:] = True
        result = mp[:]
        assert len(result) == size
        assert pnp.all(result)
        assert pnp.sum(mp.mask) == max_size
        mp.reset()
        with pytest.raises(IndexError):
            mp[1, 2] = True

    def test_setting_parameter(self):
        size = 3
        max_size, mp = self._create_masked_object(MaskedParameter, size)
        assert mp
        assert len(mp.mask) == size
        assert mp.mask.size == max_size
        assert pnp.sum(mp.mask) == 0
        mp[1] = True
        result = mp[1]
        assert len(result) == size
        assert pnp.all(result)
        assert pnp.sum(mp.mask) == size
        mp.reset()
        assert pnp.sum(mp.mask) == 0
        mp[(0, 0)] = True
        assert mp[(0, 0)] == True  # noqa: E712
        assert pnp.sum(mp.mask) == 1
        with pytest.raises(IndexError):
            mp[size] = True
        mp[:] = True
        result = mp[:]
        assert len(result) == size
        assert pnp.all(result)
        assert pnp.sum(mp.mask) == max_size
        mp.reset()

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_wrong_mode(self, masked_object):
        _, mp = self._create_masked_object(masked_object, 3)
        with pytest.raises(NotImplementedError):
            mp.perturb(mode=10)

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_perturbation(self, masked_object):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)

        for i in range(1, max_size + 1):
            mp.perturb(i)
            mp.perturb(i, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_negative_perturbation(self, masked_object):
        _, mp = self._create_masked_object(masked_object, 3)
        with pytest.raises(AssertionError):
            mp.perturb(amount=-1)

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_perturbation_remove_add(self, masked_object):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)

        for amount in [random.randrange(max_size), 0, max_size, max_size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0
            mp.perturb(amount=amount, mode=PerturbationMode.ADD)
            assert pnp.sum(mp.mask) == min(amount, max_size)
            mp.reset()

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_perturbation_invert_remove(self, masked_object):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)

        for amount in [random.randrange(max_size), 0, max_size, max_size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.INVERT)
            reversed_amount = pnp.sum(mp.mask)
            mp.perturb(amount=reversed_amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_perturbation_add_remove(self, masked_object):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)

        for amount in [random.randrange(max_size), 0, max_size, max_size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.ADD)
            assert pnp.sum(mp.mask) == min(amount, max_size)
            mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    @pytest.mark.parametrize(
        "mode",
        [
            (PerturbationMode.ADD, PerturbationMode.INVERT),
            (PerturbationMode.INVERT, PerturbationMode.INVERT),
        ],
    )
    def test_perturbation_mode(self, masked_object, mode):
        size = 3
        max_size, mp = self._create_masked_object(masked_object, size)

        for amount in [0, max_size, max_size + 1]:
            mp.perturb(amount=amount, mode=mode[0])
            mp.perturb(amount=amount, mode=mode[1])
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "masked_object", [MaskedLayer, MaskedWire, MaskedParameter]
    )
    def test_copy(self, masked_object):
        size = 3
        _, mp = self._create_masked_object(masked_object, size)
        new_mp = mp.copy()
        mp[0] = True
        assert pnp.sum(mp.mask) > pnp.sum(new_mp.mask)
        assert pnp.sum(new_mp.mask) == 0

    def _create_masked_object(self, masked_object, size):
        if masked_object == MaskedParameter:
            parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
            return size * size, masked_object(parameters)
        return size, masked_object(size)
