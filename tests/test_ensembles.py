import pytest
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit, PerturbationMode, PerturbationAxis
from maskit.ensembles import (
    AdaptiveEnsemble,
    Ensemble,
    EILEEN,
)


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


class TestEnsemble:
    def test_init(self):
        ensemble = Ensemble(dropout={})
        assert ensemble
        assert ensemble.perturb is True

    @pytest.mark.parametrize("definition", [EILEEN, GROWING, RANDOM, CLASSICAL])
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


def _create_circuit(size):
    parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
