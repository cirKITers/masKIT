from maskit._masks import PerturbationAxis as Axis
from tests.utils import cost, create_circuit, device, variational_circuit
from tests.configurations import QHACK, GROWING, RANDOM, CLASSICAL
import pytest
import random
import pennylane as qml
import pennylane.numpy as pnp

from maskit.ensembles import AdaptiveEnsemble, Ensemble, IntervalEnsemble
from maskit.optimizers import ExtendedGradientDescentOptimizer


class TestEnsemble:
    def test_init(self):
        ensemble = Ensemble(dropout={})
        assert ensemble
        assert ensemble.perturb is True

    @pytest.mark.parametrize("definition", [QHACK, GROWING, RANDOM, CLASSICAL])
    def test_ensemble_branches(self, definition):
        size = 3
        mp = create_circuit(size)
        ensemble = Ensemble(dropout=definition)
        branches = ensemble._branch(masked_circuit=mp)
        assert branches
        assert len(definition) == len(branches)

    def test_ensemble_unintialised_branches(self):
        mp = create_circuit(3)
        ensemble = Ensemble(dropout=None)
        branches = ensemble._branch(masked_circuit=mp)
        assert branches is None

    @pytest.mark.parametrize("dropout", [QHACK, RANDOM])
    def test_ensemble_step(self, dropout):
        mp = create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        ensemble = Ensemble(dropout=dropout)

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        # compare for different steps
        current_cost = 1.0
        for steps in range(3):
            random.seed(1234)
            pnp.random.seed(1234)
            result = ensemble.step(mp.copy(), optimizer, cost_fn, ensemble_steps=steps)
            if steps > 0:
                assert result.brutto_steps > result.netto_steps
            else:
                assert result.brutto_steps == 1
                assert result.netto_steps == 1
            assert result.netto_steps == steps + 1
            assert result.cost < current_cost
            current_cost = result.cost


class TestIntervalEnsemble:
    def test_init(self):
        with pytest.raises(ValueError):
            IntervalEnsemble(dropout={}, interval=0)
        assert IntervalEnsemble(dropout={}, interval=1)

    @pytest.mark.parametrize("dropout", [QHACK, RANDOM])
    def test_step(self, dropout):
        interval = 3
        mp = create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        simple_ensemble = Ensemble(dropout=None)
        interval_ensemble = IntervalEnsemble(dropout=dropout, interval=interval)

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        # train for three steps and assert that interval ensemble is better
        simple_costs = []
        simple_mp = mp.copy()
        for _ in range(interval):
            result = simple_ensemble.step(
                simple_mp, optimizer, cost_fn, ensemble_steps=1
            )
            simple_mp = result.branch
            simple_costs.append(result.cost)
        interval_mp = mp.copy()
        for i in range(interval - 1):
            result = interval_ensemble.step(
                interval_mp, optimizer, cost_fn, ensemble_steps=1
            )
            interval_mp = result.branch
            assert simple_costs[i] == result.cost
        # last step should be better
        result = interval_ensemble.step(
            interval_mp, optimizer, cost_fn, ensemble_steps=1
        )
        assert simple_costs[-1] > result.cost


class TestAdaptiveEnsemble:
    def test_init(self):
        with pytest.raises(ValueError):
            AdaptiveEnsemble(size=0, dropout={}, epsilon=0)
        ensemble = AdaptiveEnsemble(size=5, dropout={}, epsilon=0)
        assert ensemble
        assert ensemble.perturb is False

    @pytest.mark.parametrize("dropout", [QHACK, RANDOM])
    def test_step(self, dropout):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        simple_ensemble = Ensemble(dropout=None)
        adaptive_ensemble = AdaptiveEnsemble(dropout=dropout, size=3, epsilon=0.01)

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        # train for four steps and assert that interval ensemble is better
        simple_costs = []
        simple_mp = mp.copy()
        for _ in range(4):
            result = simple_ensemble.step(
                simple_mp, optimizer, cost_fn, ensemble_steps=1
            )
            simple_mp = result.branch
            simple_costs.append(result.cost)
        adaptive_mp = mp.copy()
        for i in range(3):
            result = adaptive_ensemble.step(
                adaptive_mp, optimizer, cost_fn, ensemble_steps=1
            )
            adaptive_mp = result.branch
            assert simple_costs[i] == result.cost
        # last step should be better
        result = adaptive_ensemble.step(
            adaptive_mp, optimizer, cost_fn, ensemble_steps=1
        )
        assert simple_costs[-1] > result.cost


class TestEnsembleUseCases:
    def test_classical(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=CLASSICAL)
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            result = ensemble.step(mp, optimizer, cost_fn)
            mp = result.branch
            current_cost = result.cost
        assert current_cost == pytest.approx(0.84973999)

    def test_growing(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_circuit(3, layer_size=2)
        mp.mask(Axis.LAYERS)[1:] = True
        ensemble = Ensemble(dropout=GROWING)
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        assert pnp.sum(mp.mask(Axis.LAYERS)) == 2
        for _ in range(len(mp.mask(Axis.LAYERS)) - 1):
            result = ensemble.step(mp, optimizer, cost_fn)
            mp = result.branch
            current_cost = result.cost
        assert current_cost == pytest.approx(0.86318044)
        assert pnp.sum(mp.mask(Axis.LAYERS)) == 0

    def test_random(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=RANDOM)
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            result = ensemble.step(mp, optimizer, cost_fn)
            mp = result.branch
            current_cost = result.cost
        assert current_cost == pytest.approx(0.61161677)

    def test_qhack(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=QHACK)
        circuit = qml.QNode(variational_circuit, device(mp.mask(Axis.WIRES).size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            result = ensemble.step(mp, optimizer, cost_fn)
            mp = result.branch
            current_cost = result.cost
        assert current_cost == pytest.approx(0.63792393)
