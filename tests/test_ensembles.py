import pytest
import pennylane.numpy as pnp

from maskit.masks import MaskedCircuit
from maskit.ensembles import ensemble_branches, EnsembleMaskDefinitions


@pytest.mark.parametrize("definition", list(EnsembleMaskDefinitions))
def test_ensemble_branches(definition):
    size = 3
    mp = _create_circuit(size)
    branches = ensemble_branches(dropout=definition, masked_params=mp)
    assert len(definition.value) == len(branches)


def _create_circuit(size):
    parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    return MaskedCircuit(parameters=parameters, layers=size, wires=size)
