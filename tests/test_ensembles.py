import pytest
import random
import pennylane as qml
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit, PerturbationMode, PerturbationAxis
from maskit.ensembles import AdaptiveEnsemble, Ensemble, IntervalEnsemble
from maskit.optimizers import ExtendedGradientDescentOptimizer

CLASSICAL = {
    "center": [
        {"clear": {}},
        {
            "perturb": {
                "amount": 0.1,
                "mode": PerturbationMode.ADD,
                "axis": PerturbationAxis.RANDOM,
            }
        },
    ],
}

GROWING = {"center": [{"shrink": {"amount": 1, "axis": PerturbationAxis.LAYERS}}]}

RANDOM = {
    "center": None,
    "left": [
        {"copy": {}},
        {
            "perturb": {
                "amount": 1,
                "mode": PerturbationMode.REMOVE,
                "axis": PerturbationAxis.RANDOM,
            }
        },
    ],
    "right": [
        {"copy": {}},
        {
            "perturb": {
                "amount": None,
                "mode": PerturbationMode.INVERT,
                "axis": PerturbationAxis.RANDOM,
            }
        },
    ],
}

QHACK = {
    "center": None,
    "left": [
        {"copy": {}},
        {
            "perturb": {
                "amount": 1,
                "mode": PerturbationMode.ADD,
                "axis": PerturbationAxis.RANDOM,
            },
        },
    ],
    "right": [
        {"copy": {}},
        {
            "perturb": {
                "amount": 0.05,
                "mode": PerturbationMode.REMOVE,
                "axis": PerturbationAxis.RANDOM,
            }
        },
    ],
}


class TestEnsemble:
    def test_init(self):
        ensemble = Ensemble(dropout={})
        assert ensemble
        assert ensemble.perturb is True

    @pytest.mark.parametrize("definition", [QHACK, GROWING, RANDOM, CLASSICAL])
    def test_ensemble_branches(self, definition):
        size = 3
        mp = _create_circuit(size)
        ensemble = Ensemble(dropout=definition)
        branches = ensemble._branch(masked_circuit=mp)
        assert branches
        assert len(definition) == len(branches)

    def test_ensemble_unintialised_branches(self):
        mp = _create_circuit(3)
        ensemble = Ensemble(dropout=None)
        branches = ensemble._branch(masked_circuit=mp)
        assert branches is None

    @pytest.mark.parametrize("dropout", [QHACK, RANDOM])
    def test_ensemble_step(self, dropout):
        mp = _create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        ensemble = Ensemble(dropout=dropout)

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        # compare for different steps
        cost = 1.0
        for steps in range(3):
            random.seed(1234)
            pnp.random.seed(1234)
            _params, _name, current_cost, _gradients = ensemble.step(
                mp.copy(), optimizer, cost_fn, ensemble_steps=steps
            )
            assert current_cost < cost
            cost = current_cost


class TestIntervalEnsemble:
    def test_init(self):
        with pytest.raises(ValueError):
            IntervalEnsemble(dropout={}, interval=0)
        assert IntervalEnsemble(dropout={}, interval=1)

    @pytest.mark.parametrize("dropout", [QHACK, RANDOM])
    def test_step(self, dropout):
        interval = 3
        mp = _create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        simple_ensemble = Ensemble(dropout=None)
        interval_ensemble = IntervalEnsemble(dropout=dropout, interval=interval)

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        # train for three steps and assert that interval ensemble is better
        simple_costs = []
        simple_mp = mp.copy()
        for _ in range(interval):
            simple_mp, _name, cost, _gradient = simple_ensemble.step(
                simple_mp, optimizer, cost_fn, ensemble_steps=1
            )
            simple_costs.append(cost)
        interval_mp = mp.copy()
        for i in range(interval - 1):
            interval_mp, _name, cost, _gradient = interval_ensemble.step(
                interval_mp, optimizer, cost_fn, ensemble_steps=1
            )
            assert simple_costs[i] == cost
        # last step should be better
        interval_mp, _name, cost, _gradient = interval_ensemble.step(
            interval_mp, optimizer, cost_fn, ensemble_steps=1
        )
        assert simple_costs[-1] > cost


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
        mp = _create_circuit(3, layer_size=2)
        optimizer = ExtendedGradientDescentOptimizer()
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        simple_ensemble = Ensemble(dropout=None)
        adaptive_ensemble = AdaptiveEnsemble(dropout=dropout, size=3, epsilon=0.01)

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        # train for four steps and assert that interval ensemble is better
        simple_costs = []
        simple_mp = mp.copy()
        for _ in range(4):
            simple_mp, _name, cost, _gradient = simple_ensemble.step(
                simple_mp, optimizer, cost_fn, ensemble_steps=1
            )
            simple_costs.append(cost)
        adaptive_mp = mp.copy()
        for i in range(3):
            adaptive_mp, _name, cost, _gradient = adaptive_ensemble.step(
                adaptive_mp, optimizer, cost_fn, ensemble_steps=1
            )
            assert simple_costs[i] == cost
        # last step should be better
        adaptive_mp, _name, cost, _gradient = adaptive_ensemble.step(
            adaptive_mp, optimizer, cost_fn, ensemble_steps=1
        )
        assert simple_costs[-1] > cost


class TestEnsembleUseCases:
    def test_classical(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = _create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=CLASSICAL)
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            mp, _branch_name, current_cost, _ = ensemble.step(mp, optimizer, cost_fn)
        assert current_cost == pytest.approx(0.84973999)

    def test_growing(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = _create_circuit(3, layer_size=2)
        mp.layer_mask[1:] = True
        ensemble = Ensemble(dropout=GROWING)
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        assert pnp.sum(mp.layer_mask) == 2
        for _ in range(len(mp.layer_mask) - 1):
            mp, _branch_name, current_cost, _ = ensemble.step(mp, optimizer, cost_fn)
        assert current_cost == pytest.approx(0.86318044)
        assert pnp.sum(mp.layer_mask) == 0

    def test_random(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = _create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=RANDOM)
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            mp, _branch_name, current_cost, _ = ensemble.step(mp, optimizer, cost_fn)
        assert current_cost == pytest.approx(0.61161677)

    def test_qhack(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = _create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=QHACK)
        circuit = qml.QNode(_variational_circuit, _device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return _cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            mp, _branch_name, current_cost, _ = ensemble.step(mp, optimizer, cost_fn)
        assert current_cost == pytest.approx(0.63792393)


def _cost(params, circuit, masked_circuit: MaskedCircuit) -> float:
    return 1.0 - circuit(params, masked_circuit=masked_circuit)[0]


def _device(wires: int):
    return qml.device("default.qubit", wires=wires)


def _variational_circuit(params, masked_circuit: MaskedCircuit = None):
    for layer, layer_hidden in enumerate(masked_circuit.layer_mask):
        if not layer_hidden:
            for wire, wire_hidden in enumerate(masked_circuit.wire_mask):
                if not wire_hidden:
                    if not masked_circuit.parameter_mask[layer][wire][0]:
                        qml.RX(params[layer][wire][0], wires=wire)
                    if not masked_circuit.parameter_mask[layer][wire][1]:
                        qml.RY(params[layer][wire][1], wires=wire)
            for wire in range(0, masked_circuit.layer_mask.size - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(1, masked_circuit.layer_mask.size - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
    return qml.probs(wires=range(len(masked_circuit.wire_mask)))


def _create_circuit(size: int, layer_size: int = 1):
    if layer_size == 1:
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    else:
        parameters = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size, layer_size)
        )
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
