import pytest
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit
from maskit.ensembles import (
    AdaptiveEnsemble,
    Ensemble,
    EILEEN,
    GROWING,
    RANDOM,
    CLASSICAL,
)


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


def _create_circuit(size):
    parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
