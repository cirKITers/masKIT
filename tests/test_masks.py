from maskit.circuits import variational_circuit as original_variational_circuit
from tests.utils import cost, create_freezable_circuit, device, variational_circuit
import random
import pytest

import pennylane as qml
import pennylane.numpy as pnp

from maskit._masks import (
    Mask,
    PerturbationAxis as Axis,
    PerturbationMode,
)
from maskit._masked_circuits import FreezableMaskedCircuit, MaskedCircuit


class TestMaskedCircuits:
    def test_init(self):
        mp = self._create_circuit(3)
        assert mp
        assert len(mp.mask_for_axis(Axis.WIRES)) == 3
        assert len(mp.mask_for_axis(Axis.LAYERS)) == 3
        assert mp.mask_for_axis(Axis.PARAMETERS).shape == (3, 3)

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
        with pytest.raises(AssertionError):
            MaskedCircuit.full_circuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size,
                wires=size,
                entangling_mask=Mask(shape=(size + 1, size)),
            )
        mc = MaskedCircuit.full_circuit(
            parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
            layers=size,
            wires=size,
            wire_mask=pnp.ones((size,), dtype=bool),
        )
        assert pnp.array_equal(
            mc.mask_for_axis(Axis.WIRES), pnp.ones((size,), dtype=bool)
        )

    def test_wrong_mode(self):
        mp = self._create_circuit_with_entangling_gates(3)
        with pytest.raises(AssertionError):
            mp.perturb(axis=Axis.LAYERS, mode=10)

    def test_wrong_axis(self):
        mp = self._create_circuit_with_entangling_gates(3)
        with pytest.raises(NotImplementedError):
            mp.perturb(axis=10)

    @pytest.mark.parametrize("axis", list(Axis))
    def test_clear(self, axis):
        size = 3
        mp = self._create_circuit_with_entangling_gates(size)
        mp.perturb(axis=axis, amount=size)
        assert (
            pnp.sum(mp.mask_for_axis(Axis.LAYERS))
            + pnp.sum(mp.mask_for_axis(Axis.WIRES))
            + pnp.sum(mp.mask_for_axis(Axis.PARAMETERS))
            + pnp.sum(mp.mask_for_axis(Axis.ENTANGLING))
            == size
        )
        mp.clear()
        assert (
            pnp.sum(mp.mask_for_axis(Axis.LAYERS))
            + pnp.sum(mp.mask_for_axis(Axis.WIRES))
            + pnp.sum(mp.mask_for_axis(Axis.PARAMETERS))
            + pnp.sum(mp.mask_for_axis(Axis.ENTANGLING))
            == 0
        )

    def test_perturb_entangling(self):
        size = 3
        mp = self._create_circuit(size)
        with pytest.raises(NotImplementedError):
            mp.perturb(axis=Axis.ENTANGLING, amount=1)
        # ensure nothing has happened as entangling mask is None
        assert (
            pnp.sum(mp.mask_for_axis(Axis.LAYERS))
            + pnp.sum(mp.mask_for_axis(Axis.WIRES))
            + pnp.sum(mp.mask_for_axis(Axis.PARAMETERS))
            == 0
        )

        mp = self._create_circuit_with_entangling_gates(size)
        mp.perturb(axis=Axis.ENTANGLING, amount=1)
        assert pnp.sum(mp.mask_for_axis(Axis.ENTANGLING)) == 1

    @pytest.mark.parametrize("axis", list(Axis))
    def test_zero_amount(self, axis):
        mp = self._create_circuit_with_entangling_gates(3)
        pre_sum = (
            pnp.sum(mp.mask_for_axis(Axis.WIRES))
            + pnp.sum(mp.mask_for_axis(Axis.LAYERS))
            + pnp.sum(mp.mask_for_axis(Axis.PARAMETERS))
            + pnp.sum(mp.mask_for_axis(Axis.ENTANGLING))
        )
        mp.perturb(axis=axis, amount=0)
        assert pre_sum == pnp.sum(mp.mask_for_axis(Axis.WIRES)) + pnp.sum(
            mp.mask_for_axis(Axis.LAYERS)
        ) + pnp.sum(mp.mask_for_axis(Axis.PARAMETERS)) + pnp.sum(
            mp.mask_for_axis(Axis.ENTANGLING)
        )

    def test_apply_mask(self):
        size = 3
        mp = self._create_circuit(size)
        with pytest.raises(IndexError):
            mp.apply_mask(pnp.ones((size, size - 1)))
        mp.mask_for_axis(Axis.WIRES)[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == size
        assert pnp.sum(result) == size
        mp.mask_for_axis(Axis.LAYERS)[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == 1
        assert pnp.sum(result) == 1
        mp.mask_for_axis(Axis.PARAMETERS)[(size - 1, size - 1)] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.mask) == 0
        assert pnp.sum(result) == 0

    def test_copy(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.LAYERS)[0] = True
        new_mp = mp.copy()

        mp.mask_for_axis(Axis.WIRES)[0] = True
        mp.mask_for_axis(Axis.PARAMETERS)[0, 0] = True

        assert pnp.sum(mp.mask) > pnp.sum(new_mp.mask)
        assert pnp.sum(new_mp.mask_for_axis(Axis.WIRES)) == 0
        assert pnp.sum(new_mp.mask_for_axis(Axis.LAYERS)) == pnp.sum(
            mp.mask_for_axis(Axis.LAYERS)
        )
        assert pnp.sum(new_mp.mask_for_axis(Axis.PARAMETERS)) == 0
        assert Axis.ENTANGLING not in new_mp.masks

        # also test copying of existing entanglement mask
        mp = self._create_circuit_with_entangling_gates(size)
        assert mp.mask_for_axis(Axis.ENTANGLING) is not None
        new_mp = mp.copy()
        mp.mask_for_axis(Axis.ENTANGLING)[0, 0] = True
        assert pnp.sum(mp.mask_for_axis(Axis.ENTANGLING)) == 1
        assert pnp.sum(new_mp.mask_for_axis(Axis.ENTANGLING)) == 0

    def test_parameters(self):
        size = 3
        mp = self._create_circuit(size)

        new_random_values = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size)
        )
        assert (mp.parameters != new_random_values).all()
        mp.parameters = new_random_values
        assert (mp.parameters == new_random_values).all()

    def test_shrink_layer(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.LAYERS)[:] = True
        mp.shrink(amount=1, axis=Axis.LAYERS)
        assert pnp.sum(mp.mask) == mp.mask.size - size

    def test_shrink_wire(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.WIRES)[:] = True
        mp.shrink(amount=1, axis=Axis.WIRES)
        assert pnp.sum(mp.mask) == mp.mask.size - size

    def test_shrink_parameter(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.PARAMETERS)[:] = True
        mp.shrink(amount=1, axis=Axis.PARAMETERS)
        assert pnp.sum(mp.mask) == mp.mask.size - 1

    def test_shrink_entangling(self):
        size = 3
        mp = self._create_circuit_with_entangling_gates(size)
        mp.mask_for_axis(Axis.ENTANGLING)[:] = True
        mp.shrink(amount=1, axis=Axis.ENTANGLING)
        assert (
            pnp.sum(mp.mask_for_axis(Axis.ENTANGLING))
            == mp.mask_for_axis(Axis.ENTANGLING).size - 1
        )

        # also test in case no mask is set
        mp = self._create_circuit(size)
        with pytest.raises(NotImplementedError):
            mp.shrink(amount=1, axis=Axis.ENTANGLING)
        assert Axis.ENTANGLING not in mp.masks
        assert pnp.sum(mp.mask) == 0  # also ensure that nothing else was shrunk

    def test_shrink_wrong_axis(self):
        mp = self._create_circuit(3)
        with pytest.raises(NotImplementedError):
            mp.shrink(amount=1, axis=10)

    def test_execute(self):
        mp = self._create_circuit(3)
        perturb_operation = {
            "perturb": {
                "amount": 1,
                "axis": Axis.PARAMETERS,
                "mode": PerturbationMode.ADD,
            }
        }
        # test empty operations
        assert MaskedCircuit.execute(mp, []) == mp
        # test existing method
        MaskedCircuit.execute(mp, [perturb_operation])
        assert pnp.sum(mp.mask) == 1
        # test existing method with copy
        new_mp = MaskedCircuit.execute(
            mp, [{"clear": {}}, {"copy": {}}, perturb_operation]
        )
        assert mp != new_mp
        assert pnp.sum(new_mp.mask) == 1
        # test non-existing method
        with pytest.raises(AttributeError):
            MaskedCircuit.execute(mp, [{"non_existent": {"test": 1}}])

    def test_active(self):
        mp = self._create_circuit(3)
        assert mp.active() == 9
        mp.mask_for_axis(Axis.WIRES)[0] = True
        assert mp.active() == 6
        mp.mask_for_axis(Axis.LAYERS)[0] = True
        assert mp.active() == 4
        mp.mask_for_axis(Axis.PARAMETERS)[1][1] = True
        assert mp.active() == 3

    def test_default_value(self):
        size = 3
        mp = self._create_circuit(size)
        mp.default_value = 0
        mp.mask_for_axis(Axis.WIRES)[0] = True
        mp.mask_for_axis(Axis.PARAMETERS)[2, 2] = True
        mp.mask_for_axis(Axis.LAYERS)[1] = True
        assert pnp.sum(mp.parameters[:, 0] != 0) == size
        mp.mask_for_axis(Axis.WIRES)[0] = False
        assert pnp.sum(mp.parameters[:, 0] == 0) == size
        mp.mask_for_axis(Axis.WIRES)[1] = False
        assert pnp.sum(mp.parameters[:, 1] != 0) == size
        mp.mask_for_axis(Axis.LAYERS)[1] = False
        assert pnp.sum(mp.parameters == 0) == size * 2 - 1
        mp.mask_for_axis(Axis.PARAMETERS)[2, 2] = False
        assert pnp.sum(mp.parameters == 0) == size * 2

    def test_default_value_perturb(self):
        mp = MaskedCircuit.full_circuit(
            parameters=pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(4, 3, 2)),
            layers=4,
            wires=3,
            default_value=0,
        )
        mp.mask_for_axis(Axis.PARAMETERS)[:] = True
        mp.perturb(axis=Axis.PARAMETERS, amount=0.5, mode=PerturbationMode.INVERT)
        assert pnp.sum(mp.parameters == 0) == round(0.5 * 4 * 3 * 2)

    def test_default_value_shrink(self):
        mp = MaskedCircuit.full_circuit(
            parameters=pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(4, 3, 2)),
            layers=4,
            wires=3,
            default_value=0,
        )
        mp.mask_for_axis(Axis.LAYERS)[:] = True
        mp.shrink(axis=Axis.LAYERS)
        assert pnp.sum(mp.parameters == 0) == 6

    def test_dynamic_parameters(self):
        size = 3
        circuit = self._create_circuit(size)
        circuit.mask_for_axis(Axis.LAYERS)[0] = True
        assert circuit.differentiable_parameters.size == (size - 1) * size
        assert (
            circuit.expanded_parameters(
                changed_parameters=circuit.differentiable_parameters
            ).size
            == size * size
        )
        # in this case, no wrong values can be written

        circuit._dynamic_parameters = False  # disable dynamic parameters
        assert circuit.differentiable_parameters.size == size * size
        assert (
            circuit.expanded_parameters(changed_parameters=circuit.parameters).size
            == size * size
        )
        circuit.differentiable_parameters = pnp.ones((size, size))
        # ensure that first layer has not been changed
        for i in range(size):
            assert circuit.parameters[0, i] != 1

    def test_entangling_mask_application(self):
        size = 4
        mp = self._create_circuit_with_entangling_gates(size)
        rotations = [pnp.random.choice([0, 1, 2]) for _ in range(size * size)]
        circuit = qml.QNode(
            original_variational_circuit,
            device(mp.mask_for_axis(Axis.WIRES).size),
        )

        circuit(mp.differentiable_parameters, rotations, mp)
        assert circuit.specs["gate_types"]["CZ"] == 12

        mp.perturb(axis=Axis.ENTANGLING, mode=PerturbationMode.ADD, amount=6)
        circuit(mp.differentiable_parameters, rotations, mp)
        assert circuit.specs["gate_types"]["CZ"] == 6

    def _create_circuit(self, size):
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
        return MaskedCircuit.full_circuit(
            parameters=parameters, layers=size, wires=size
        )

    def _create_circuit_with_entangling_gates(self, size):
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
        return MaskedCircuit.full_circuit(
            parameters=parameters,
            layers=size,
            wires=size,
            entangling_mask=Mask(shape=(size, size - 1)),
        )


class TestFreezableMaskedCircuit:
    def test_init(self):
        mp = create_freezable_circuit(3)
        assert mp

    def test_freeze(self):
        size = 3
        mp = create_freezable_circuit(size)
        assert (
            mp.differentiable_parameters.size == mp.mask_for_axis(Axis.PARAMETERS).size
        )
        # Test 0 amount
        mask = mp.mask
        mp.freeze(axis=Axis.LAYERS, amount=0, mode=PerturbationMode.ADD)
        assert pnp.array_equal(mp.mask, mask)
        # Test freezing of layers
        mp.freeze(axis=Axis.LAYERS, amount=1, mode=PerturbationMode.ADD)
        assert (
            mp.differentiable_parameters.size
            == mp.mask_for_axis(Axis.PARAMETERS).size - size
        )
        assert pnp.sum(mp.freeze_mask_for_axis(Axis.LAYERS)) == 1
        assert pnp.sum(mp.mask) == size
        # Test freezing of wires
        mp.freeze(axis=Axis.WIRES, amount=1, mode=PerturbationMode.ADD)
        assert pnp.sum(mp.freeze_mask_for_axis(Axis.WIRES)) == 1
        assert pnp.sum(mp.mask) == 2 * size - 1
        # Test freezing of parameters
        mp.freeze(axis=Axis.PARAMETERS, amount=1, mode=PerturbationMode.ADD)
        assert pnp.sum(mp.freeze_mask_for_axis(Axis.PARAMETERS)) == 1
        assert pnp.sum(mp.mask) == 2 * size - 1 or pnp.sum(mp.mask) == 2 * size
        # Test wrong axis
        with pytest.raises(NotImplementedError):
            mp.freeze(axis=10, amount=1, mode=PerturbationMode.ADD)

    def test_copy(self):
        mp = create_freezable_circuit(3)
        mp.perturb(amount=5, mode=PerturbationMode.ADD)
        mp.freeze(amount=2, axis=Axis.LAYERS, mode=PerturbationMode.ADD)
        mp_copy = mp.copy()
        assert isinstance(mp_copy, FreezableMaskedCircuit)
        assert pnp.array_equal(mp.mask, mp_copy.mask)
        mp.perturb(amount=5, mode=PerturbationMode.REMOVE)
        mp.freeze(amount=2, axis=Axis.LAYERS, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0
        assert not pnp.array_equal(mp.mask, mp_copy.mask)

    def test_complex(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_freezable_circuit(3, layer_size=2)
        circuit = qml.QNode(
            variational_circuit, device(mp.mask_for_axis(Axis.WIRES).size)
        )
        optimizer = qml.GradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        mp.freeze(axis=Axis.LAYERS, amount=2, mode=PerturbationMode.ADD)
        mp.freeze(axis=Axis.WIRES, amount=2, mode=PerturbationMode.ADD)

        last_changeable = pnp.sum(mp.parameters[~mp.mask])
        frozen = pnp.sum(mp.parameters[mp.mask])
        for _ in range(10):
            params = optimizer.step(
                cost_fn, mp.differentiable_parameters, masked_circuit=mp
            )
            mp.differentiable_parameters = params
            current_changeable = pnp.sum(mp.parameters[~mp.mask])
            assert last_changeable - current_changeable != 0
            assert frozen - pnp.sum(mp.parameters[mp.mask]) == 0
            last_changeable = current_changeable


class TestMask:
    def test_init(self):
        size = 3
        with pytest.raises(AssertionError):
            Mask((size,), mask=pnp.array([True, True, False, False]))
        with pytest.raises(AssertionError):
            Mask((size,), mask=pnp.array([0, 1, 3]))
        preset = [False, True, False]
        mp = Mask((size,), mask=pnp.array(preset))
        assert pnp.array_equal(mp.mask, preset)

    def test_setting(self):
        size = 3
        mp = Mask((size,))
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
        mp = Mask((3,))
        with pytest.raises(NotImplementedError):
            mp.perturb(mode=10)

    def test_perturbation(self):
        size = 3
        mp = Mask((size,))

        for i in range(1, size + 1):
            mp.perturb(i)
            mp.perturb(i, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    def test_percentage_perturbation(self):
        size = 3
        mp = Mask((size,))

        for i in [0.01, 0.1, 0.5, 0.9]:
            mp.perturb(amount=i)
            assert pnp.sum(mp.mask) == round(i * mp.mask.size)
            mp.clear()

    def test_wrong_percentage_perturbation(self):
        size = 3
        mp = Mask((size,))

        for i in [1.1, 1.5, 3.1]:
            mp.perturb(amount=i)
            assert pnp.sum(mp.mask) == round(i)
            mp.clear()

    def test_negative_perturbation(self):
        mp = Mask((3,))
        with pytest.raises(AssertionError):
            mp.perturb(amount=-1)

    def test_perturbation_remove_add(self):
        size = 3
        mp = Mask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0
            mp.perturb(amount=amount, mode=PerturbationMode.ADD)
            assert pnp.sum(mp.mask) == min(amount, size)
            mp.clear()

    def test_perturbation_invert_remove(self):
        size = 3
        mp = Mask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.INVERT)
            reversed_amount = pnp.sum(mp.mask).unwrap()  # unwrap tensor
            mp.perturb(amount=reversed_amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    def test_perturbation_add_remove(self):
        size = 3
        mp = Mask((size,))

        for amount in [random.randrange(size), 0, size, size + 1]:
            mp.perturb(amount=amount, mode=PerturbationMode.ADD)
            assert pnp.sum(mp.mask) == min(amount, size)
            mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
            assert pnp.sum(mp.mask) == 0

    @pytest.mark.parametrize(
        "mode",
        [
            (PerturbationMode.ADD, PerturbationMode.INVERT),
            (PerturbationMode.INVERT, PerturbationMode.INVERT),
        ],
    )
    def test_perturbation_mode(self, mode):
        size = 3
        mp = Mask((size,))

        for amount in [0, size, size + 1]:
            mp.perturb(amount=amount, mode=mode[0])
            mp.perturb(amount=amount, mode=mode[1])
            assert pnp.sum(mp.mask) == 0

    def test_shrink(self):
        size = 3
        mp = Mask((size,))

        for amount in range(size + 1):
            mp[:] = True
            mp.shrink(amount)
            assert pnp.sum(mp.mask) == size - amount

    def test_shrink_nd(self):
        size = 3
        mp = Mask((size, size - 1))
        for amount in range(mp.mask.size + 1):
            mp[:] = True
            mp.shrink(amount)
            assert pnp.sum(mp.mask) == mp.mask.size - amount

    def test_copy(self):
        size = 3
        mp = Mask((size,))
        new_mp = mp.copy()
        mp[0] = True
        assert pnp.sum(mp.mask) > pnp.sum(new_mp.mask)
        assert pnp.sum(new_mp.mask) == 0

    def test_apply_mask(self):
        size = 3
        mp = Mask((size,))
        with pytest.raises(IndexError):
            mp.apply_mask(pnp.ones((size - 1, size)))
        mp.mask[1] = True
        result = mp.apply_mask(pnp.ones((size,), dtype=bool))
        assert pnp.sum(mp.mask) == 1
        assert pnp.sum(result) == size - 1
