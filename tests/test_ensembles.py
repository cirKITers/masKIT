import pytest
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit
from maskit.ensembles import Ensemble, EILEEN, GROWING, RANDOM, CLASSICAL


@pytest.mark.parametrize("definition", [EILEEN, GROWING, RANDOM, CLASSICAL])
def test_ensemble_branches(definition):
    size = 3
    mp = _create_circuit(size)
    ensemble = Ensemble(dropout=definition)
    branches = ensemble.branch(masked_circuit=mp)
    assert len(definition) == len(branches)


def _create_circuit(size):
    parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
