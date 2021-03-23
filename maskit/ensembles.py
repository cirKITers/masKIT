from typing import Dict

from maskit.masks import PerturbationAxis, PerturbationMode, MaskedCircuit


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

EILEEN = {
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

GROWING = {"center": [{"shrink": {"amount": 1, "axis": PerturbationAxis.LAYERS}}]}


def ensemble_branches(
    dropout: Dict,
    masked_params: MaskedCircuit,
) -> Dict[str, MaskedCircuit]:
    branches = {}
    for key in dropout:
        operations = dropout[key]
        new_branch = masked_params
        if operations is not None:
            for operation_dict in operations:
                for operation, parameters in operation_dict.items():
                    value = new_branch.__getattribute__(operation)(**parameters)
                    if value is not None:
                        new_branch = value
        branches[key] = new_branch
    return branches
