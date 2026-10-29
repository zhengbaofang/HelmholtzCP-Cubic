"""
Check Jacobian-SVD reuse in the bounded least-squares solver.
Rejected trials reuse a factorization; accepted states refresh it.
A failed factorization must remain retryable. Test coordinates are dimensionless.
"""

from __future__ import annotations

import numpy as np

from HelmholtzCPCubic.Numerics import solveLeastSquares


def _solve(evaluate, initial):
    """
    Solve the one-variable test problem with fixed bounds and tolerances.
    """

    return solveLeastSquares(
        evaluate, np.asarray(initial), np.array([-10.0]), np.array([10.0]),
        resTol=1.0e-8, stepTol=1.0e-10, maxEval=160,
        initialRadius=1.0, maxRadius=20.0, acceptanceRatio=0.05,
    )


def test_rejected_trials_reuse_svd_and_new_models_refresh_it(monkeypatch) -> None:
    """
    Preserve the nonlinear root while distinguishing retry and acceptance.
    """

    original = np.linalg.svd
    matrices = []

    def observed(jacobian, **kwargs):
        matrices.append(jacobian.copy())
        return original(jacobian, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", observed)

    def infeasible(state):
        if state[0] != 0.0:
            raise FloatingPointError("controlled domain rejection")
        return np.ones(1), np.ones((1, 1))

    result = _solve(infeasible, [0.0])
    assert not result.converged and result.evaluations > 2
    assert len(matrices) == 1
    matrices.clear()
    result = _solve(lambda x: (np.array([x[0] ** 2 - 2.0]), np.array([[2.0 * x[0]]])), [1.0])
    assert result.converged
    np.testing.assert_allclose(result.state, [np.sqrt(2.0)], atol=1.0e-8)
    assert len(matrices) > 1


def test_failed_svd_is_not_retained_as_a_valid_factorization(monkeypatch) -> None:
    """
    A transient linear-algebra failure must not poison later retries.
    """

    original = np.linalg.svd
    attempts = []

    def failOnce(jacobian, **kwargs):
        attempts.append(jacobian.copy())
        if len(attempts) == 1:
            raise np.linalg.LinAlgError("controlled first decomposition failure")
        return original(jacobian, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", failOnce)
    result = _solve(lambda x: (x - 2.0, np.ones((1, 1))), [0.0])
    assert result.converged
    assert result.subproblemFailures == 1
    assert len(attempts) >= 2
    np.testing.assert_allclose(result.state, [2.0], atol=1.0e-8)
