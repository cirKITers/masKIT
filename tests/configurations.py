from maskit._masks import PerturbationAxis, PerturbationMode


CLASSICAL = {
    "center": [
        {"clear": {}},
        {
            "perturb": {
                "amount": 0.1,
                "mode": PerturbationMode.SET,
                "axis": PerturbationAxis.PARAMETERS,
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
                "mode": PerturbationMode.RESET,
                "axis": PerturbationAxis.PARAMETERS,
            }
        },
    ],
    "right": [
        {"copy": {}},
        {
            "perturb": {
                "amount": None,
                "mode": PerturbationMode.INVERT,
                "axis": PerturbationAxis.PARAMETERS,
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
                "mode": PerturbationMode.SET,
                "axis": PerturbationAxis.PARAMETERS,
            },
        },
    ],
    "right": [
        {"copy": {}},
        {
            "perturb": {
                "amount": 0.05,
                "mode": PerturbationMode.RESET,
                "axis": PerturbationAxis.PARAMETERS,
            }
        },
    ],
}
