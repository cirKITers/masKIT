import pennylane as qml
import pennylane.numpy as pnp

from maskit._masks import PerturbationAxis as Axis, DropoutMask, FreezeMask
from maskit._masked_circuits import MaskedCircuit


def device(wires: int):
    return qml.device("default.qubit", wires=wires)


def cost(params, circuit, masked_circuit: MaskedCircuit) -> float:
    return 1.0 - circuit(params, masked_circuit=masked_circuit)[0]


def plain_cost(params, circuit) -> float:
    return 1.0 - circuit(params)[0]


def create_circuit(size: int, layer_size: int = 1):
    if layer_size == 1:
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    else:
        parameters = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size, layer_size)
        )
    return MaskedCircuit.full_circuit(parameters=parameters, layers=size, wires=size)


def create_freezable_circuit(size: int, layer_size: int = 1):
    if layer_size == 1:
        parameters = pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(size, size))
    else:
        parameters = pnp.random.uniform(
            low=-pnp.pi, high=pnp.pi, size=(size, size, layer_size)
        )
    return MaskedCircuit(
        parameters=parameters,
        layers=size,
        wires=size,
        masks=(
            (axis, mask_type)
            for axis in Axis
            if axis is not Axis.ENTANGLING
            for mask_type in [DropoutMask, FreezeMask]
        ),
    )
    # TODO: remove me
    # return FreezableMaskedCircuit.full_circuit(
    #     parameters=parameters, layers=size, wires=size
    # )


def variational_circuit(params, masked_circuit: MaskedCircuit = None):
    full_parameters = masked_circuit.expanded_parameters(params)
    for layer, layer_hidden in enumerate(masked_circuit.mask(Axis.LAYERS)):
        if not layer_hidden:
            for wire, wire_hidden in enumerate(masked_circuit.mask(Axis.WIRES)):
                if not wire_hidden:
                    if not masked_circuit.mask(Axis.PARAMETERS)[layer][wire][0]:
                        qml.RX(full_parameters[layer][wire][0], wires=wire)
                    if not masked_circuit.mask(Axis.PARAMETERS)[layer][wire][1]:
                        qml.RY(full_parameters[layer][wire][1], wires=wire)
            for wire in range(0, masked_circuit.mask(Axis.LAYERS).size - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(1, masked_circuit.mask(Axis.LAYERS).size - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
    return qml.probs(wires=range(len(masked_circuit.mask(Axis.WIRES))))


def plain_variational_circuit(params):
    layers = params.shape[0]
    wires = params.shape[1]
    for layer in range(layers):
        for wire in range(wires):
            qml.RX(params[layer][wire][0], wires=wire)
            qml.RY(params[layer][wire][1], wires=wire)
        for wire in range(0, layers - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, layers - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
    return qml.probs(wires=range(wires))
