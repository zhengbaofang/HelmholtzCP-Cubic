"""
Check trust-region steps, convergence, curvature and bound handling.
Use dimensionless polynomial and Rosenbrock-type objectives.
"""

from __future__ import annotations

import numpy as np
import pytest

from HelmholtzCPCubic.Numerics import (
    ScalarTrustRegionConfig,
    minimizeScalar,
)
from HelmholtzCPCubic.Numerics.ScalarTrustRegion import _nearExactStep


@pytest.mark.parametrize("radius", [1.0, 1.0e-4, 1.0e-8, 2.3283064365386963e-10, 1.0e-12])
def test_boundary_step_has_keasy_accuracy_at_small_radii(radius: float) -> None:
    """
    Approximate boundaries retain relative accuracy and the analytic direction.
    """

    gradient = np.array([.7, 3., -2.])
    hessian = 2.0 * np.eye(3)
    expectedDirection = -gradient / np.linalg.norm(gradient)
    step = _nearExactStep(gradient, hessian, radius)
    assert np.linalg.norm(step) <= radius * (1 + 1e-14)
    assert np.linalg.norm(step) / radius >= 0.9 - 1.0e-14
    np.testing.assert_allclose(step / np.linalg.norm(step), expectedDirection,
                               rtol=1.0e-10, atol=1.0e-12)
    assert float(gradient @ step + 0.5 * step @ hessian @ step) < 0.0


def test_keasy_controls_boundary_solution() -> None:
    """
    The tolerance must affect an anisotropic boundary solve, not be ignored.
    """

    gradient = np.array([4., 2., 1.])
    hessian = np.diag([2., 3., 4.])
    coarse = _nearExactStep(gradient, hessian, 1., keasy=0.1)
    fine = _nearExactStep(gradient, hessian, 1., keasy=0.01)
    assert not np.array_equal(coarse, fine)
    for step, tolerance in ((coarse, 0.1), (fine, 0.01)):
        assert 1. - tolerance - 1.0e-14 <= np.linalg.norm(step) <= 1. + 1.0e-14


@pytest.mark.parametrize("keasy", [0., 1., -1., float("nan")])
def test_invalid_keasy_is_rejected(keasy: float) -> None:
    """
    A bad inner boundary tolerance must fail explicitly.
    """

    with pytest.raises(ValueError, match="keasy"):
        _nearExactStep(np.ones(2), np.eye(2), 1., keasy=keasy)


def test_repeated_negative_eigenspace_satisfies_quadratic_kkt() -> None:
    """
    A rotated repeated-eigenvalue hard case must reach the correct boundary.
    """

    basis, _ = np.linalg.qr(np.array([[1., 2., 3.], [-2., 1., .5], [.3, -.7, 1.]]))
    hessian = basis @ np.diag([-2., -2., 3.]) @ basis.T
    gradient = basis @ np.array([0., 0., .3])
    step = _nearExactStep(gradient, hessian, 1.)
    assert np.linalg.norm(step) == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose((hessian + 2*np.eye(3)) @ step + gradient, 0., atol=1e-12)


@pytest.mark.parametrize("initialValue", [1.0, 7.5e-9])
def test_quadratic_one_dimensional_minimizes_to_zero(initialValue: float) -> None:
    """
    Also reject premature success for an initial gradient of 1.5e-8.
    """

    def objective(state):
        return (float(state[0] ** 2), 2.0 * state, np.asarray([[2.0]]))

    result = minimizeScalar(
        objective,
        np.asarray([initialValue]),
        np.asarray([-10.0]),
        np.asarray([10.0]),
    )
    assert result.converged
    assert result.stationarityNorm <= 1.0e-8
    assert result.value == pytest.approx(0.0, abs=1.0e-10)
    assert result.state[0] == pytest.approx(0.0, abs=1.0e-8)


@pytest.mark.parametrize("diagonalScaling", [False, True])
def test_caller_residual_check_prevents_premature_coordinate_convergence(diagonalScaling) -> None:
    """
    A small scaled gradient cannot replace an independent residual check.
    """

    checkedStates = []

    def objective(state):
        return float(state[0] ** 2), 2.0 * state, np.asarray([[2.0]])

    def qualified(state, value, gradient, hessian):
        checkedStates.append(state.copy())
        return abs(state[0]) <= 1.0e-12

    result = minimizeScalar(
        objective, np.asarray([4.0e-9]), -np.ones(1), np.ones(1),
        config=ScalarTrustRegionConfig(diagonalScaling=diagonalScaling),
        convergenceCheck=qualified,
    )
    assert result.converged
    assert result.evaluations == 2
    assert result.termination == "trial_caller_convergence"
    assert checkedStates[0][0] == 4.0e-9
    assert abs(result.state[0]) <= 1.0e-12


@pytest.mark.parametrize("initialValue", [0.0, 1.0])
def test_failed_caller_qualification_is_not_overridden(initialValue: float) -> None:
    """
    Default convergence cannot override failed current or trial qualification.
    """

    def objective(state):
        return float(state[0] ** 2), 2.0 * state, np.asarray([[2.0]])

    result = minimizeScalar(
        objective, np.asarray([initialValue]), -2.0 * np.ones(1), 2.0 * np.ones(1),
        convergenceCheck=lambda state, value, gradient, hessian: False,
    )
    assert not result.converged
    assert result.termination == "step_or_radius_tolerance"
    assert result.stationarityNorm == 0.0


@pytest.mark.parametrize("diagonalScaling", [False, True])
def test_two_dimensional_rosenbrock_like_minimum(diagonalScaling) -> None:
    """
    Recover the minimum of a coupled two-dimensional objective.
    """

    def objective(state):
        x, y = state
        value = (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2
        gradient = np.asarray(
            [
                -2.0 * (1.0 - x) - 400.0 * x * (y - x**2),
                200.0 * (y - x**2),
            ]
        )
        hessian = np.asarray(
            [
                [2.0 - 400.0 * (y - 3.0 * x**2), -400.0 * x],
                [-400.0 * x, 200.0],
            ]
        )
        return float(value), gradient, hessian

    result = minimizeScalar(
        objective,
        np.asarray([-1.2, 1.0]),
        np.asarray([-5.0, -5.0]),
        np.asarray([5.0, 5.0]),
        config=ScalarTrustRegionConfig(diagonalScaling=diagonalScaling),
    )
    assert result.converged
    assert result.state[0] == pytest.approx(1.0, abs=1.0e-4)
    assert result.state[1] == pytest.approx(1.0, abs=1.0e-4)


def test_indefinite_objective_escapes_stationary_point() -> None:
    """
    A small-gradient indefinite point must not stop the iteration.
    """

    # f(x) = -0.5 x^2: origin is a stationary point with negative curvature.
    def objective(state):
        value = -0.5 * state[0] ** 2
        gradient = np.asarray([-state[0]])
        hessian = np.asarray([[-1.0]])
        return float(value), gradient, hessian

    result = minimizeScalar(
        objective,
        np.asarray([0.5]),
        np.asarray([-2.0]),
        np.asarray([2.0]),
        config=ScalarTrustRegionConfig(maxEval=200),
    )

    # The minimum over the open box is at -2 (boundary excluded); the solver
    # must move away from the origin and decrease the value substantially.
    assert result.value < -1.0


@pytest.mark.parametrize("diagonalScaling", [False, True])
def test_open_box_bounds_are_respected(diagonalScaling) -> None:

    # Interior minimum (0.5, 0); the box is open so the exact minimum stays
    # strictly feasible and the stationarity criterion can be met.
    """
    Keep accepted iterates strictly inside the supplied open bounds.
    """

    def objective(state):
        value = (state[0] - 0.5) ** 2 + state[1] ** 2
        gradient = 2.0 * (state - np.asarray([0.5, 0.0]))
        hessian = 2.0 * np.eye(2)
        return float(value), gradient, hessian

    result = minimizeScalar(
        objective,
        np.asarray([0.2, 0.5]),
        np.asarray([0.1, -1.0]),
        np.asarray([1.0, 1.0]),
        config=ScalarTrustRegionConfig(diagonalScaling=diagonalScaling),
    )
    assert result.converged
    assert result.state[0] == pytest.approx(0.5, abs=1.0e-6)
    assert result.state[1] == pytest.approx(0.0, abs=1.0e-6)


def test_initial_state_outside_bounds_is_rejected() -> None:
    """
    Reject an initial state outside the open coordinate bounds.
    """

    def objective(state):
        return float(state[0] ** 2), 2.0 * state, np.asarray([[2.0]])

    with pytest.raises(ValueError):
        minimizeScalar(
            objective,
            np.asarray([0.0]),
            np.asarray([0.1]),
            np.asarray([1.0]),
        )


def test_config_rejects_inconsistent_controls() -> None:
    """
    Reject invalid or inconsistent trust-region controls.
    """

    with pytest.raises(ValueError):
        ScalarTrustRegionConfig(stationarityTol=0.0)
    with pytest.raises(ValueError):
        ScalarTrustRegionConfig(maxEval=0)
    with pytest.raises(ValueError):
        ScalarTrustRegionConfig(initialRadius=5.0, maxRadius=1.0)
    with pytest.raises(ValueError):
        ScalarTrustRegionConfig(acceptanceRatio=0.5)


def test_rejected_trials_reuse_eigh_and_acceptance_refreshes_curvature(monkeypatch) -> None:
    """
    Use one factorization for retries and refresh it for accepted new models.
    """

    original = np.linalg.eigh
    matrices = []

    def observed(hessian):
        matrices.append(hessian.copy())
        return original(hessian)

    monkeypatch.setattr(np.linalg, "eigh", observed)

    def infeasible(state):
        if state[0] != 0.0:
            return None
        return 1.0, np.ones(1), np.ones((1, 1))

    result = minimizeScalar(infeasible, np.zeros(1), -np.ones(1), np.ones(1))
    assert not result.converged and result.evaluations > 2
    assert len(matrices) == 1
    matrices.clear()

    def nonlinear(state):
        exponential = np.exp(state[0])
        return float(exponential - 2.0 * state[0]), np.array([exponential - 2.0]), np.array([[exponential]])

    result = minimizeScalar(nonlinear, np.zeros(1), -10.0 * np.ones(1), 10.0 * np.ones(1))
    assert result.converged
    np.testing.assert_allclose(result.state, [np.log(2.0)], atol=1.0e-8)
    assert len(matrices) > 1
    assert not np.array_equal(matrices[0], matrices[-1])
