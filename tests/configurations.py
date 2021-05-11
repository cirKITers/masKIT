from maskit.masks import PerturbationAxis, PerturbationMode


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
