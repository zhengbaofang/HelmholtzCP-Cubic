r"""
Solve independent sign-change brackets together using Chandrupatla (1997).
Combine interpolation and bisection while retaining each root's bracket.
Report invalid inputs, non-finite values and iteration-limit failures.
Absolute coordinate tolerances use the same units as the supplied bounds.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np


def solveBatchedBrackets(
    function: Callable[[np.ndarray], np.ndarray],
    leftBounds: np.ndarray,
    rightBounds: np.ndarray,
    *,
    relativeTolerance: float,
    absoluteTolerance: float,
    maxIterations: int,
) -> np.ndarray:
    """
    Return one root from every independent sign-change bracket.

    Args:
        function: Elementwise vector function whose output preserves the shape
            of its one-dimensional finite input.
        leftBounds: Finite lower bracket endpoints.
        rightBounds: Finite upper bracket endpoints.
        relativeTolerance: Nonnegative relative bracket-width tolerance.
        absoluteTolerance: Positive absolute bracket-width tolerance.
        maxIterations: Positive maximum number of batched iterations.

    Returns:
        Root estimates in the same order as the input brackets.

    Raises:
        ValueError: If inputs are inconsistent or a bracket has no sign change.
        FloatingPointError: If the function returns a non-finite value.
        RuntimeError: If any bracket exceeds the iteration limit.
    """

    left = np.asarray(leftBounds, dtype=float).copy()
    right = np.asarray(rightBounds, dtype=float).copy()
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError("batched root bounds must be equal nonempty vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("batched root bounds must be finite")
    if np.any(left >= right):
        raise ValueError("each batched root bracket must be strictly increasing")
    if (
        not math.isfinite(relativeTolerance)
        or relativeTolerance < 0.0
        or not math.isfinite(absoluteTolerance)
        or absoluteTolerance <= 0.0
    ):
        raise ValueError(
            "batched root relative tolerance must be nonnegative and the "
            "absolute tolerance must be positive"
        )
    if (
        not isinstance(maxIterations, int)
        or isinstance(maxIterations, bool)
        or maxIterations <= 0
    ):
        raise ValueError("maxIterations must be a positive integer")

    endpointValues = _evaluate(function, np.concatenate((left, right)))
    bracketCount = left.size
    leftValues = endpointValues[:bracketCount]
    rightValues = endpointValues[bracketCount:]
    roots = np.empty(bracketCount, dtype=float)
    converged = np.zeros(bracketCount, dtype=bool)

    leftIsRoot = leftValues == 0.0
    rightIsRoot = rightValues == 0.0
    roots[leftIsRoot] = left[leftIsRoot]
    roots[rightIsRoot & ~leftIsRoot] = right[rightIsRoot & ~leftIsRoot]
    converged |= leftIsRoot | rightIsRoot
    invalid = (
        np.signbit(leftValues) == np.signbit(rightValues)
    ) & ~converged
    if np.any(invalid):
        raise ValueError(
            f"{int(np.count_nonzero(invalid))} batched root bracket(s) "
            "do not contain a sign change"
        )

    x1 = left
    f1 = leftValues
    x2 = right
    f2 = rightValues
    x3 = np.empty_like(x1)
    f3 = np.empty_like(f1)
    interpolationFraction = np.full(bracketCount, 0.5, dtype=float)

    for _iteration in range(maxIterations):
        _markConverged(
            roots,
            converged,
            x1,
            f1,
            x2,
            f2,
            relativeTolerance,
            absoluteTolerance,
        )
        activeIndices = np.flatnonzero(~converged)
        if activeIndices.size == 0:
            return roots

        trial = x1[activeIndices] + interpolationFraction[activeIndices] * (
            x2[activeIndices] - x1[activeIndices]
        )
        trialValues = _evaluate(function, trial)
        exact = trialValues == 0.0
        if np.any(exact):
            exactIndices = activeIndices[exact]
            roots[exactIndices] = trial[exact]
            converged[exactIndices] = True

        updateIndices = activeIndices[~exact]
        if updateIndices.size == 0:
            continue
        updateValues = trialValues[~exact]
        updateTrials = trial[~exact]
        sameSign = (
            np.signbit(updateValues) == np.signbit(f1[updateIndices])
        )
        oldX1 = x1[updateIndices].copy()
        oldF1 = f1[updateIndices].copy()
        oldX2 = x2[updateIndices].copy()
        oldF2 = f2[updateIndices].copy()
        x3[updateIndices] = np.where(sameSign, oldX1, oldX2)
        f3[updateIndices] = np.where(sameSign, oldF1, oldF2)
        x2[updateIndices] = np.where(sameSign, oldX2, oldX1)
        f2[updateIndices] = np.where(sameSign, oldF2, oldF1)
        x1[updateIndices] = updateTrials
        f1[updateIndices] = updateValues
        interpolationFraction[updateIndices] = _nextFractions(
            x1[updateIndices],
            f1[updateIndices],
            x2[updateIndices],
            f2[updateIndices],
            x3[updateIndices],
            f3[updateIndices],
            relativeTolerance,
            absoluteTolerance,
        )

    _markConverged(
        roots,
        converged,
        x1,
        f1,
        x2,
        f2,
        relativeTolerance,
        absoluteTolerance,
    )
    if np.any(~converged):
        raise RuntimeError(
            f"{int(np.count_nonzero(~converged))} batched root bracket(s) "
            "exceeded the iteration limit"
        )
    return roots


def _evaluate(
    function: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate and validate one vector of elementwise function values.
    """

    result = np.asarray(function(values), dtype=float)
    if result.shape != values.shape:
        raise ValueError("batched root function must preserve input shape")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("batched root function returned non-finite values")
    return result


def _markConverged(
    roots: np.ndarray,
    converged: np.ndarray,
    x1: np.ndarray,
    f1: np.ndarray,
    x2: np.ndarray,
    f2: np.ndarray,
    relativeTolerance: float,
    absoluteTolerance: float,
) -> None:
    """
    Mark brackets whose width satisfies the requested tolerance.
    """

    active = ~converged
    chooseFirst = np.abs(f1) <= np.abs(f2)
    estimates = np.where(chooseFirst, x1, x2)
    tolerances = absoluteTolerance + relativeTolerance * np.abs(estimates)
    finished = active & (np.abs(x2 - x1) <= tolerances)
    roots[finished] = estimates[finished]
    converged[finished] = True


def _nextFractions(
    x1: np.ndarray,
    f1: np.ndarray,
    x2: np.ndarray,
    f2: np.ndarray,
    x3: np.ndarray,
    f3: np.ndarray,
    relativeTolerance: float,
    absoluteTolerance: float,
) -> np.ndarray:
    """
    Return safeguarded Chandrupatla interpolation fractions.
    """

    width = np.abs(x2 - x1)
    chooseFirst = np.abs(f1) <= np.abs(f2)
    estimates = np.where(chooseFirst, x1, x2)
    tolerances = absoluteTolerance + relativeTolerance * np.abs(estimates)
    fractions = np.full(x1.shape, 0.5, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (x1 - x2) / (x3 - x2)
        phi = (f1 - f2) / (f3 - f2)
        alpha = (x3 - x1) / (x2 - x1)
        admissible = (
            np.isfinite(xi)
            & np.isfinite(phi)
            & np.isfinite(alpha)
            & (xi > 0.0)
            & (xi < 1.0)
            & (phi > 1.0 - np.sqrt(1.0 - xi))
            & (phi < np.sqrt(xi))
        )
        interpolation = (
            f1 / (f1 - f2) * f3 / (f3 - f2)
            - alpha * f1 / (f3 - f1) * f2 / (f2 - f3)
        )
    admissible &= np.isfinite(interpolation)
    fractions[admissible] = interpolation[admissible]
    minimumFraction = np.minimum(
        0.5,
        0.5 * tolerances / np.maximum(width, np.finfo(float).tiny),
    )
    return np.clip(fractions, minimumFraction, 1.0 - minimumFraction)


__all__ = ["solveBatchedBrackets"]
