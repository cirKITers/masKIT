from typing import Optional, Tuple
from pennylane import GradientDescentOptimizer, AdamOptimizer
from pennylane._grad import grad as get_gradient
from enum import Enum
from pennylane.numpy import ndarray
import scipy.optimize as sciopt


class L_BFGS_B:
    """
    The L-BFGS-B optimiser provides a wrapper for the implementation provided
    in scipy. Please see the :py:func:`~sciopt.fmin_l_bfgs_b` documentation for
    further details.

    In case the method :py:meth:`~.step` is used, the value of parameter `maxiter`
    is ignored and interpreted as `1` instead.

    :param bounds: tuple of `min` and `max` for each value of provided parameters
    :param m: maximum number of variable metric corrections used to define the
        limited memory matrix
    :param factr: information on when to stop iterating, e.g. 1e12 for low accuracy;
        1e7 for moderate accuracy; 10.0 for extremely high accuracy
    :param pgtol: when to stop iterating with regards to gradients
    :param epsilon: Step size used when `approx_grad` is `True`
    :param iprint: Frequency of output
    :param maxfun: Maximum number of function evaluations
    :param maxiter: Maximum number of iterations
    :param disp: If zero, then no output. If positive this over-rides `iprint`
    :param callback: Called after each iteration with current parameters
    :param maxls: Maximum number of line search steps (per iteration)
    """

    __slots__ = (
        "bounds",
        "m",
        "factr",
        "pgtol",
        "epsilon",
        "iprint",
        "maxfun",
        "maxiter",
        "disp",
        "callback",
        "maxls",
    )

    def __init__(
        self,
        bounds: Optional[ndarray] = None,
        m: int = 10,
        factr: float = 1e7,  # qiskit: factr: float = 10,
        pgtol: float = 1e-5,
        epsilon: float = 1e-8,
        iprint: int = -1,
        maxfun: int = 15000,  # qiskit: maxfun: int = 1000,
        maxiter: int = 15000,
        disp=None,
        callback=None,
        maxls: int = 20,
    ):
        self.bounds = bounds
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.iprint = iprint
        self.maxfun = maxfun
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.maxls = maxls

    def optimize(
        self,
        objective_fn,
        parameters: ndarray,
        *args,
        grad_fn=None,
        **kwargs,
    ) -> Tuple[ndarray, float, ndarray]:
        """
        :param objective_fn: Function to minimize
        :param parameters: Initial guess of parameters
        :param grad_fn: The gradient of `func`. In case of `None`, the gradient is
            approximated numerically
        """
        return self._optimize(
            objective_fn, parameters, self.maxiter, *args, grad_fn=grad_fn, **kwargs
        )

    def _optimize(
        self, objective_fn, parameters: ndarray, maxiter, *args, grad_fn, **kwargs
    ) -> Tuple[ndarray, float, ndarray]:
        shape = parameters.shape
        shaped_fn = self._reshaping_objective_fn(objective_fn, shape, **kwargs)
        approx_grad = False if grad_fn is not None else True
        updated_parameters, cost, info = sciopt.fmin_l_bfgs_b(
            shaped_fn,
            parameters.flatten(),
            fprime=get_gradient(shaped_fn) if grad_fn is None else grad_fn,
            *args,
            approx_grad=approx_grad,
            bounds=self.bounds,
            m=self.m,
            factr=self.factr,
            pgtol=self.pgtol,
            epsilon=self.epsilon,
            iprint=self.iprint,
            maxfun=self.maxfun,
            maxiter=maxiter,
            disp=self.disp,
            callback=self.callback,
            maxls=self.maxls,
        )
        return updated_parameters.reshape(shape), cost, info["grad"]

    def _reshaping_objective_fn(self, objective_fn, shape, **kwargs):
        return lambda params, *args: objective_fn(
            params.reshape(shape), *args, **kwargs
        )

    def step(self, objective_fn, parameters, *args, grad_fn=None, **kwargs) -> ndarray:
        """
        :param objective_fn: Function to minimize
        :param parameters: Initial guess of parameters
        :param grad_fn: The gradient of `func`. In case of `None`, the gradient is
            approximated numerically
        """
        return self.step_cost_and_grad(
            objective_fn, parameters, *args, grad_fn=grad_fn, **kwargs
        )[0]

    def step_and_cost(
        self, objective_fn, parameters, *args, grad_fn=None, **kwargs
    ) -> Tuple[ndarray, float]:
        """
        :param objective_fn: Function to minimize
        :param parameters: Initial guess of parameters
        :param grad_fn: The gradient of `func`. In case of `None`, the gradient is
            approximated numerically
        """
        return self.step_cost_and_grad(
            objective_fn, parameters, *args, grad_fn=grad_fn, **kwargs
        )[:2]

    def step_cost_and_grad(
        self, objective_fn, parameters, *args, grad_fn=None, **kwargs
    ) -> Tuple[ndarray, float, ndarray]:
        """
        :param objective_fn: Function to minimize
        :param parameters: Initial guess of parameters
        :param grad_fn: The gradient of `func`. In case of `None`, the gradient is
            approximated numerically
        """
        updated_parameters, cost, gradients = self._optimize(
            objective_fn, parameters, 1, *args, grad_fn=grad_fn, **kwargs
        )
        return updated_parameters, cost, gradients


class ExtendedGradientDescentOptimizer(GradientDescentOptimizer):
    def step_cost_and_grad(self, objective_fn, *args, grad_fn=None, **kwargs):
        """
        This function copies the functionality of the GradientDescentOptimizer
        one-to-one but changes the return statement to also return the gradient.
        """
        gradient, forward = self.compute_grad(
            objective_fn, args, kwargs, grad_fn=grad_fn
        )
        new_args = self.apply_grad(gradient, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward, gradient
        return new_args, forward, gradient

    def __repr__(self):
        return f"{self.__class__.__name__}({self._stepsize})"


class ExtendedAdamOptimizer(ExtendedGradientDescentOptimizer, AdamOptimizer):
    pass


class ExtendedOptimizers(Enum):
    GD = ExtendedGradientDescentOptimizer
    ADAM = ExtendedAdamOptimizer
    L_BFGS_B = L_BFGS_B
