"""
Check batched scalar roots, bracket ordering and reported failures.
Polynomial inputs and tolerances use dimensionless coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from HelmholtzCPCubic.Numerics.BatchedBracket import solveBatchedBrackets


def test_batched_brackets_converge_together_without_losing_order() -> None:
    """
    Recover three polynomial roots in input-bracket order using batched calls.
    """

    batchSizes: list[int] = []

    def polynomial(values: np.ndarray) -> np.ndarray:
        batchSizes.append(values.size)
        return (values - 0.7) * (values - 2.2) * (values - 4.8)

    roots = solveBatchedBrackets(
        polynomial,
        np.array([0.0, 1.5, 4.0]),
        np.array([1.3, 3.0, 6.0]),
        relativeTolerance=1.0e-12,
        absoluteTolerance=1.0e-14,
        maxIterations=100,
    )

    assert roots == pytest.approx((0.7, 2.2, 4.8), abs=1.0e-11)
    assert batchSizes[0] == 6  # all bracket endpoints share one evaluation
    assert any(size > 1 for size in batchSizes[1:])


def test_batched_brackets_expose_invalid_values_and_iteration_failure() -> None:
    """
    Report missing sign changes, non-finite values and exhausted iterations.
    """

    with pytest.raises(ValueError, match="sign change"):
        solveBatchedBrackets(
            lambda values: values**2 + 1.0,
            np.array([-1.0]),
            np.array([1.0]),
            relativeTolerance=1.0e-8,
            absoluteTolerance=1.0e-12,
            maxIterations=10,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        solveBatchedBrackets(
            lambda values: np.full_like(values, np.nan),
            np.array([-1.0]),
            np.array([1.0]),
            relativeTolerance=1.0e-8,
            absoluteTolerance=1.0e-12,
            maxIterations=10,
        )
    with pytest.raises(RuntimeError, match="iteration limit"):
        solveBatchedBrackets(
            lambda values: values**3 - 2.0,
            np.array([0.0]),
            np.array([2.0]),
            relativeTolerance=1.0e-14,
            absoluteTolerance=1.0e-15,
            maxIterations=1,
        )
