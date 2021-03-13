import random
import pytest

import pennylane.numpy as pnp

from maskit.masked_parameters import (
    MaskedParameters,
    PerturbationAxis,
    PerturbationMode,
)

# TODO: unit test für None
# TODO: unit test für Entfernung der Maske, wenn keine angesetzt ist
# TODO: test len indices == 0
# TODO: check if it is the row/column as selected
# TODO: test auf None


def test_init():
    mp = MaskedParameters(
        pnp.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])),
        perturbation_axis=PerturbationAxis.RANDOM,
    )
    assert mp


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
def test_perturbation(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    factor = 1
    if perturbation != PerturbationAxis.RANDOM:
        factor = 3
    for i in range(3):
        mp.perturb(i + 1)
        assert pnp.sum(mp.mask) == (i + 1) * factor
        mp.perturb(i + 1, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
def test_negative_perturbation(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    with pytest.raises(AssertionError):
        mp.perturb(-1)


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
def test_perturbation_remove_add(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    factor = 1
    if perturbation != PerturbationAxis.RANDOM:
        factor = 3

    maximum_size = (
        mp.mask.size if perturbation == PerturbationAxis.RANDOM else len(mp.mask)
    )
    for amount in [random.randrange(maximum_size), 0, maximum_size, maximum_size + 1]:
        mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0
        mp.perturb(amount=amount, mode=PerturbationMode.ADD)
        assert pnp.sum(mp.mask) == min(amount, maximum_size) * factor
        mp.reset()


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
@pytest.mark.parametrize(
    "mode",
    [
        (PerturbationMode.ADD, PerturbationMode.INVERT),
        (PerturbationMode.INVERT, PerturbationMode.INVERT),
    ],
)
def test_perturbation_mode(perturbation, mode):
    mp = _create_masked_parameter(perturbation_axis=perturbation)

    maximum_size = (
        mp.mask.size if perturbation == PerturbationAxis.RANDOM else len(mp.mask)
    )
    for amount in [0, maximum_size, maximum_size + 1]:
        mp.perturb(amount=amount, mode=mode[0])
        mp.perturb(amount=amount, mode=mode[1])
        assert pnp.sum(mp.mask) == 0


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
def test_perturbation_invert_remove(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)

    maximum_size = (
        mp.mask.size if perturbation == PerturbationAxis.RANDOM else len(mp.mask)
    )
    for amount in [random.randrange(maximum_size), 0, maximum_size, maximum_size + 1]:
        mp.perturb(amount=amount, mode=PerturbationMode.INVERT)
        reversed_amount = pnp.sum(mp.mask)
        mp.perturb(amount=reversed_amount, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0


@pytest.mark.parametrize("perturbation", list(PerturbationAxis))
def test_perturbation_add_remove(perturbation):
    mp = _create_masked_parameter(perturbation_axis=perturbation)
    factor = 1
    if perturbation != PerturbationAxis.RANDOM:
        factor = 3

    maximum_size = (
        mp.mask.size if perturbation == PerturbationAxis.RANDOM else len(mp.mask)
    )
    for amount in [random.randrange(maximum_size), 0, maximum_size, maximum_size + 1]:
        mp.perturb(amount=amount, mode=PerturbationMode.ADD)
        assert pnp.sum(mp.mask) == min(amount, maximum_size) * factor
        mp.perturb(amount=amount, mode=PerturbationMode.REMOVE)
        assert pnp.sum(mp.mask) == 0


def _create_masked_parameter(perturbation_axis: PerturbationAxis):
    return MaskedParameters(
        pnp.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])),
        perturbation_axis=perturbation_axis,
    )
