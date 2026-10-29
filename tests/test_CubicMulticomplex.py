"""
Compare multicomplex, analytic and automatic PR/SRK derivatives.
Check energy, critical Jacobians, fourth derivatives and local corrections.
The NWE state covers nonzero interactions and the high-acentric-factor PR branch.
Test states use K and mol/L; full discovery and TPD searches are excluded.
"""

from importlib import import_module

import numpy as np
import pytest
import torch

from HelmholtzCPCubic import CriticalPointSolver


pytest.importorskip("multicomplex")


@pytest.mark.parametrize("name,temperature,density", (
    ("C1C2_PR", 250.0, 5.0),
    ("C1C2_SRK", 250.0, 5.0),
    ("NWE", 400.0, 3.0),
))
def test_multicomplex_preserves_full_critical_derivatives(name, temperature, density):
    """
    Check energy, Hessian, CP residual/Jacobian and fourth contraction.
    """

    model, fractions, config = import_module(f"examples.{name}").makeCase()
    solvers = {mode: CriticalPointSolver(model, config, backend=mode)
               for mode in ("analytic", "pymcx", "ad")}
    state = np.log([temperature, density])
    evaluations = {mode: solver._directEvaluator.criticalEvaluation(state, fractions)
                   for mode, solver in solvers.items()}
    reference = evaluations["ad"]
    for evaluation in evaluations.values():
        np.testing.assert_allclose(evaluation.residual, reference.residual,
                                   rtol=2.0e-9, atol=2.0e-10)
        np.testing.assert_allclose(evaluation.jacobian, reference.jacobian,
                                   rtol=2.0e-9, atol=2.0e-10)
        np.testing.assert_allclose(evaluation.critical.stabilityMatrix,
                                   reference.critical.stabilityMatrix,
                                   rtol=2.0e-9, atol=2.0e-10)
    mcxKernel = solvers["pymcx"]._directEvaluator
    densities = density * fractions
    np.testing.assert_allclose(float(mcxKernel._freeEnergy(temperature, densities)),
                               float(model.freeEnergy(temperature, model.asTensor(densities))),
                               rtol=2.0e-12, atol=1.0e-10)
    direction = reference.critical.minEigenvector
    step = torch.zeros((), dtype=model.dtype, requires_grad=True)
    derivative = model.freeEnergy(temperature, model.asTensor(densities) + step * direction)
    for order in range(4):
        derivative = torch.autograd.grad(derivative, step, create_graph=order < 3)[0]
    fourth = mcxKernel._derivative(temperature, densities, (direction.numpy(),), (0, 0, 4))
    np.testing.assert_allclose(fourth, float(derivative), rtol=2.0e-9, atol=2.0e-10)
    assert solvers["pymcx"].directBackendName == "cubic_pymcx"
    assert CriticalPointSolver(model, config).directBackendName == "helmholtz_ad"


@pytest.mark.parametrize("name", ("C1C2_PR", "C1C2_SRK"))
def test_multicomplex_local_correction_preserves_original_acceptance(name):
    """
    Use the same documented direct start, solver and unchanged tolerances.
    """

    model, fractions, config = import_module(f"examples.{name}").makeCase()
    points = {mode: CriticalPointSolver(model, config, backend=mode).solveDirect(fractions)
              for mode in ("analytic", "pymcx")}
    for point in points.values():
        assert point.converged and point.rNorm <= config.resTol
        independent = CriticalPointSolver(model, config).kernel.evaluate(point.T, point.rho, fractions)
        assert float(torch.linalg.vector_norm(torch.stack((independent.qMin, independent.cStar)))) <= config.resTol
    np.testing.assert_allclose(
        [points["pymcx"].T, points["pymcx"].P, points["pymcx"].rho],
        [points["analytic"].T, points["analytic"].P, points["analytic"].rho],
        rtol=1.0e-9, atol=1.0e-8,
    )
