import random
from tests.configurations import RANDOM
from maskit.ensembles import Ensemble
from tests.utils import (
    plain_cost,
    cost,
    create_circuit,
    device,
    plain_variational_circuit,
    variational_circuit,
)
from maskit.optimizers import ExtendedGradientDescentOptimizer, L_BFGS_B

import pennylane as qml
import pennylane.numpy as pnp


class TestLBFGSBOptimizer:
    def test_init(self):
        optimizer = L_BFGS_B()
        assert optimizer

    def test_plain(self):
        optimizer = L_BFGS_B()
        mp = create_circuit(3, layer_size=2)
        mp_step = mp.copy()
        circuit = qml.QNode(plain_variational_circuit, device(mp.wire_mask.size))

        def cost_fn(params):
            return plain_cost(
                params,
                circuit,
            )

        base_cost = cost_fn(mp.parameters)
        _params, final_cost, _gradient = optimizer.optimize(cost_fn, mp.parameters)
        params, step_cost, _gradient = optimizer.step_cost_and_grad(
            cost_fn, mp_step.parameters
        )
        assert final_cost < base_cost
        assert step_cost < base_cost
        assert final_cost < step_cost

        new_params = optimizer.step(cost_fn, mp_step.parameters)
        assert pnp.array_equal(params, new_params)
        new_params, new_cost = optimizer.step_and_cost(cost_fn, new_params)
        assert new_cost < step_cost

    def test_shape(self):
        optimizer = L_BFGS_B()
        original_optimizer = ExtendedGradientDescentOptimizer()
        mp = create_circuit(3, layer_size=2)
        mp_original = mp.copy()
        circuit = qml.QNode(plain_variational_circuit, device(mp.wire_mask.size))

        def cost_fn(params):
            return plain_cost(
                params,
                circuit,
            )

        params, _cost, _gradient = optimizer.step_cost_and_grad(cost_fn, mp.parameters)
        original_params, _cost, _gradient = original_optimizer.step_cost_and_grad(
            cost_fn, mp_original.parameters
        )
        assert mp.parameters.shape == params.shape
        assert original_params.shape == params.shape

    def test_mask(self):
        optimizer = L_BFGS_B()
        mp = create_circuit(3, layer_size=2)
        mp_step = mp.copy()
        circuit = qml.QNode(variational_circuit, device(mp.wire_mask.size))

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit=masked_circuit,
            )

        base_cost = cost_fn(mp.parameters, masked_circuit=mp)
        _params, final_cost, _gradient = optimizer.optimize(
            cost_fn, mp.differentiable_parameters, masked_circuit=mp
        )
        params, step_cost, _gradient = optimizer.step_cost_and_grad(
            cost_fn, mp_step.differentiable_parameters, masked_circuit=mp_step
        )
        assert final_cost < base_cost
        assert step_cost < base_cost
        assert final_cost < step_cost

    def test_ensemble(self):
        random.seed(1234)
        pnp.random.seed(1234)
        optimizer = L_BFGS_B()
        ensemble = Ensemble(dropout=RANDOM)
        mp = create_circuit(3, layer_size=2)
        circuit = qml.QNode(variational_circuit, device(mp.wire_mask.size))

        def cost_fn(params, masked_circuit=None):
            return cost(
                params,
                circuit,
                masked_circuit=masked_circuit,
            )

        base_cost = cost_fn(mp.parameters, masked_circuit=mp)
        for _ in range(5):
            result = ensemble.step(
                masked_circuit=mp,
                optimizer=optimizer,
                objective_fn=cost_fn,
                ensemble_steps=1,
            )
            assert result.cost < base_cost
