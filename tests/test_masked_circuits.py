import pytest
import random

import pennylane as qml
import pennylane.numpy as pnp

from maskit.circuits import variational_circuit as original_variational_circuit
from maskit._masks import (
    PerturbationAxis as Axis,
    PerturbationMode as Mode,
    DropoutMask,
    FreezeMask,
)
from maskit._masked_circuits import MaskedCircuit

from tests.utils import cost, create_freezable_circuit, device, variational_circuit


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
                entangling_mask=DropoutMask(shape=(size + 1, size)),
            )
        with pytest.raises(NotImplementedError):
            MaskedCircuit(
                parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
                layers=size,
                wires=size,
                masks=[(Axis.ENTANGLING, DropoutMask)],
            )
        mc = MaskedCircuit.full_circuit(
            parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
            layers=size,
            wires=size,
            wire_mask=DropoutMask(shape=(size,), mask=pnp.ones((size,), dtype=bool)),
        )
        assert pnp.array_equal(
            mc.mask_for_axis(Axis.WIRES), pnp.ones((size,), dtype=bool)
        )

    def test_mask(self):
        """
        The aggregated mask is built dynamically from the registered masks for the
        different axes. The test ensures that the mask covers the whole set of
        parameters.
        """
        size = 3
        # test circuit with all masks
        mc = MaskedCircuit.full_circuit(
            parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
            layers=size,
            wires=size,
        )
        assert mc.full_mask(DropoutMask).size == size * size

        # test circuit with no masks
        mc = MaskedCircuit(
            parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
            layers=size,
            wires=size,
        )
        assert mc.full_mask(DropoutMask).size == size * size

        # test circuit containing only layer mask
        mc = MaskedCircuit(
            parameters=pnp.random.uniform(low=0, high=1, size=(size, size)),
            layers=size,
            wires=size,
            masks=[(Axis.LAYERS, DropoutMask)],
        )
        assert mc.full_mask(DropoutMask).size == size * size

    def test_wrong_mode(self):
        mp = self._create_circuit_with_entangling_gates(3)
        with pytest.raises(AssertionError):
            mp.perturb(axis=Axis.LAYERS, mode=10)

    def test_wrong_axis(self):
        mp = self._create_circuit_with_entangling_gates(3)
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            mp.apply_mask(pnp.ones((size, size - 1)))
        mp.mask_for_axis(Axis.WIRES)[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.full_mask(DropoutMask)) == size
        assert pnp.sum(result) == size
        mp.mask_for_axis(Axis.LAYERS)[: size - 1] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.full_mask(DropoutMask)) == 1
        assert pnp.sum(result) == 1
        mp.mask_for_axis(Axis.PARAMETERS)[(size - 1, size - 1)] = True
        result = mp.apply_mask(pnp.ones((size, size), dtype=bool))
        assert pnp.sum(~mp.full_mask(DropoutMask)) == 0
        assert pnp.sum(result) == 0

    def test_copy(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.LAYERS)[0] = True
        new_mp = mp.copy()

        mp.mask_for_axis(Axis.WIRES)[0] = True
        mp.mask_for_axis(Axis.PARAMETERS)[0, 0] = True

        assert pnp.sum(mp.full_mask(DropoutMask)) > pnp.sum(
            new_mp.full_mask(DropoutMask)
        )
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
        assert (
            pnp.sum(mp.full_mask(DropoutMask)) == mp.full_mask(DropoutMask).size - size
        )

    def test_shrink_wire(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.WIRES)[:] = True
        mp.shrink(amount=1, axis=Axis.WIRES)
        assert (
            pnp.sum(mp.full_mask(DropoutMask)) == mp.full_mask(DropoutMask).size - size
        )

    def test_shrink_parameter(self):
        size = 3
        mp = self._create_circuit(size)
        mp.mask_for_axis(Axis.PARAMETERS)[:] = True
        mp.shrink(amount=1, axis=Axis.PARAMETERS)
        assert pnp.sum(mp.full_mask(DropoutMask)) == mp.full_mask(DropoutMask).size - 1

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
        with pytest.raises(ValueError):
            mp.shrink(amount=1, axis=Axis.ENTANGLING)
        assert Axis.ENTANGLING not in mp.masks
        assert (
            pnp.sum(mp.full_mask(DropoutMask)) == 0
        )  # also ensure that nothing else was shrunk

    def test_shrink_wrong_axis(self):
        mp = self._create_circuit(3)
        with pytest.raises(ValueError):
            mp.shrink(amount=1, axis=10)

    def test_execute(self):
        mp = self._create_circuit(3)
        perturb_operation = {
            "perturb": {
                "amount": 1,
                "axis": Axis.PARAMETERS,
                "mode": Mode.SET,
            }
        }
        # test empty operations
        assert MaskedCircuit.execute(mp, []) == mp
        # test existing method
        MaskedCircuit.execute(mp, [perturb_operation])
        assert pnp.sum(mp.full_mask(DropoutMask)) == 1
        # test existing method with copy
        new_mp = MaskedCircuit.execute(
            mp, [{"clear": {}}, {"copy": {}}, perturb_operation]
        )
        assert mp != new_mp
        assert pnp.sum(new_mp.full_mask(DropoutMask)) == 1
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
        mp.perturb(axis=Axis.PARAMETERS, amount=0.5, mode=Mode.INVERT)
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

        mp.perturb(axis=Axis.ENTANGLING, mode=Mode.SET, amount=6)
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
            entangling_mask=DropoutMask(shape=(size, size - 1)),
        )


class TestFreezableMaskedCircuit:
    def test_init(self):
        size = 3
        mp = create_freezable_circuit(size)
        assert mp

    def test_freeze(self):
        size = 3
        mp = create_freezable_circuit(size)
        assert (
            mp.differentiable_parameters.size
            == mp.mask_for_axis(Axis.PARAMETERS, mask_type=FreezeMask).size
        )
        # Test 0 amount
        # FIXME: does not test the same thing, befor it tested the whole mask
        mask = mp.full_mask(FreezeMask)
        mp.perturb(axis=Axis.LAYERS, amount=0, mode=Mode.SET, mask=FreezeMask)
        assert pnp.array_equal(mp.full_mask(FreezeMask), mask)
        # Test freezing of layers
        mp.perturb(axis=Axis.LAYERS, amount=1, mode=Mode.SET, mask=FreezeMask)
        assert (
            mp.differentiable_parameters.size
            == mp.mask_for_axis(Axis.PARAMETERS).size - size
        )
        assert pnp.sum(mp.mask_for_axis(axis=Axis.LAYERS, mask_type=FreezeMask)) == 1
        assert pnp.sum(mp._accumulated_mask()) == size  # dropout and freeze mask
        # Test freezing of wires
        mp.perturb(axis=Axis.WIRES, amount=1, mode=Mode.SET, mask=FreezeMask)
        assert pnp.sum(mp.mask_for_axis(axis=Axis.WIRES, mask_type=FreezeMask)) == 1
        assert (
            pnp.sum(mp._accumulated_mask()) == 2 * size - 1
        )  # dropout and freeze mask
        # Test freezing of parameters
        mp.perturb(axis=Axis.PARAMETERS, amount=1, mode=Mode.SET, mask=FreezeMask)
        assert (
            pnp.sum(mp.mask_for_axis(axis=Axis.PARAMETERS, mask_type=FreezeMask)) == 1
        )
        assert (
            pnp.sum(mp._accumulated_mask()) == 2 * size - 1
            or pnp.sum(mp._accumulated_mask()) == 2 * size  # dropout and freeze mask
        )
        # Test wrong axis
        with pytest.raises(ValueError):
            mp.perturb(axis=10, amount=1, mode=Mode.SET, mask=FreezeMask)

    def test_copy(self):
        mp = create_freezable_circuit(3)
        mp.perturb(amount=5, mode=Mode.SET)
        mp.perturb(amount=2, axis=Axis.LAYERS, mode=Mode.SET, mask=FreezeMask)
        mp_copy = mp.copy()
        assert pnp.array_equal(mp.full_mask(FreezeMask), mp_copy.full_mask(FreezeMask))
        mp.perturb(amount=5, mode=Mode.RESET)
        mp.perturb(amount=2, axis=Axis.LAYERS, mode=Mode.RESET, mask=FreezeMask)
        assert pnp.sum(mp.full_mask(FreezeMask)) == 0
        assert not pnp.array_equal(
            mp.full_mask(FreezeMask), mp_copy.full_mask(FreezeMask)
        )

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

        mp.perturb(axis=Axis.LAYERS, amount=2, mode=Mode.SET, mask=FreezeMask)
        mp.perturb(axis=Axis.WIRES, amount=2, mode=Mode.SET, mask=FreezeMask)

        last_changeable = pnp.sum(mp.parameters[~mp._accumulated_mask()])
        frozen = pnp.sum(mp.parameters[mp._accumulated_mask()])
        for _ in range(10):
            params = optimizer.step(
                cost_fn, mp.differentiable_parameters, masked_circuit=mp
            )
            mp.differentiable_parameters = params
            current_changeable = pnp.sum(mp.parameters[~mp._accumulated_mask()])
            assert last_changeable - current_changeable != 0
            assert frozen - pnp.sum(mp.parameters[mp._accumulated_mask()]) == 0
            last_changeable = current_changeable
