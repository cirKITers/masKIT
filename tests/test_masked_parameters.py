import pytest

import pennylane.numpy as pnp

from masked_parameters import MaskedParameters, PerturbationAxis

# TODO: unit test für None
# TODO: unit test für Entfernung der Maske, wenn keine angesetzt ist


def test_init():
    mp = MaskedParameters(
        pnp.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])),
        perturbation_axis=PerturbationAxis.RANDOM)
    assert mp


@pytest.mark.parametrize('perturbation', list(PerturbationAxis))
def test_perturbation(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    factor = 1
    if perturbation != PerturbationAxis.RANDOM:
        factor = 3
    # TODO: check if it the row that is True
    for i in range(3):
        mp.perturb(i + 1)
        print(f"current i {i}")
        print(mp.mask)
        assert pnp.sum(mp.mask) == (i + 1) * factor
        mp.perturb(-(i + 1))
        print(mp.mask)
        assert pnp.sum(mp.mask) == 0


def _create_masked_parameter(perturbation_axis: PerturbationAxis):
    return MaskedParameters(
        pnp.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])),
        perturbation_axis=perturbation_axis)
