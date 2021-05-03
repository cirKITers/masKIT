import pytest
import random
import pennylane as qml
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit, PerturbationMode, PerturbationAxis
from maskit.ensembles import AdaptiveEnsemble, Ensemble
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


# TODO: remove explicit tests for private methods
# TODO: add tests for step


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
        assert len(definition) == len(branches)

    def test_ensemble_unintialised_branches(self):
        mp = _create_circuit(3)
        ensemble = Ensemble(dropout=None)
        branches = ensemble._branch(masked_circuit=mp)
        assert len(branches) == 1
        assert list(branches.values())[0] == mp


class TestAdaptiveEnsemble:
    def test_init(self):
        with pytest.raises(ValueError):
            AdaptiveEnsemble(size=0, dropout={}, epsilon=0, enforcement_dropout=[{}])
        ensemble = AdaptiveEnsemble(
            size=5, dropout={}, epsilon=0, enforcement_dropout=[{}]
        )
        assert ensemble
        assert ensemble.perturb is False

    def test_branch(self):
        mp = _create_circuit(3)
        ensemble = AdaptiveEnsemble(
            size=2, dropout={"center": {}}, epsilon=0.1, enforcement_dropout=[{}]
        )
        ensemble.perturb = True
        branch = ensemble._branch(masked_circuit=mp)
        assert ensemble.perturb is False
        assert len(branch) == 1


class TestEnsembleUseCases:
    def test_classical(self):
        random.seed(1234)
        pnp.random.seed(1234)
        mp = _create_circuit(3, layer_size=2)
        ensemble = Ensemble(dropout=CLASSICAL)
        circuit = qml.QNode(self._circuit, self._device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return self._cost(
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
        circuit = qml.QNode(self._circuit, self._device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return self._cost(
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
        circuit = qml.QNode(self._circuit, self._device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return self._cost(
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
        circuit = qml.QNode(self._circuit, self._device(mp.wire_mask.size))
        optimizer = ExtendedGradientDescentOptimizer()

        def cost_fn(params, masked_circuit=None):
            return self._cost(
                params,
                circuit,
                masked_circuit,
            )

        current_cost = 1.0
        for _ in range(10):
            mp, _branch_name, current_cost, _ = ensemble.step(mp, optimizer, cost_fn)
        assert current_cost == pytest.approx(0.63792393)

    def _device(self, wires: int):
        return qml.device("default.qubit", wires=wires)

    def _circuit(self, params, masked_circuit: MaskedCircuit = None):
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

    def _cost(self, params, circuit, masked_circuit: MaskedCircuit) -> float:
        return 1.0 - circuit(params, masked_circuit=masked_circuit)[0]


def _create_circuit(size: int, layer_size: int = 1):
    if layer_size == 1:
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    else:
        parameters = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size, layer_size)
        )
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
