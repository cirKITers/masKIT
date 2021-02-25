import random
import pytest

import pennylane.numpy as pnp

from masked_parameters import MaskedParameters, PerturbationAxis, PerturbationMode

# TODO: unit test für None
# TODO: unit test für Entfernung der Maske, wenn keine angesetzt ist
# TODO: test len indices == 0
# TODO: check if it is the row/column as selected
# TODO: test auf None


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
    for i in range(3):
        mp.perturb(i + 1)
        print(f"current i {i}")
        print(mp.mask)
        assert pnp.sum(mp.mask) == (i + 1) * factor
        mp.perturb(i + 1, mode=PerturbationMode.REMOVE)
        print(mp.mask)
        assert pnp.sum(mp.mask) == 0


@pytest.mark.parametrize('perturbation', list(PerturbationAxis))
def test_negative_perturbation(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    with pytest.raises(AssertionError):
        mp.perturb(-1)


# (PerturbationMode.INVERT, PerturbationMode.REMOVE),
# (PerturbationMode.INVERT, PerturbationMode.INVERT),
# (PerturbationMode.ADD, PerturbationMode.INVERT),
# (PerturbationMode.REMOVE, PerturbationMode.ADD),

@pytest.mark.parametrize('perturbation', list(PerturbationAxis))
def test_perturbation_add_remove(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    factor = 1
    if perturbation != PerturbationAxis.RANDOM:
        factor = 3

    maximum_size = mp.mask.size if perturbation == PerturbationAxis.RANDOM else len(mp.mask)
    for amount in [random.randrange(maximum_size), 0, maximum_size, maximum_size + 1]:
        print(f"amount {amount}, factor {factor}, perturbation {perturbation}")
        mp.perturb(amount=amount, mode=PerturbationMode.ADD)
        assert pnp.sum(mp.mask) == min(amount, maximum_size) * factor
        mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0


def _create_masked_parameter(perturbation_axis: PerturbationAxis):
    return MaskedParameters(
        pnp.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])),
        perturbation_axis=perturbation_axis)
