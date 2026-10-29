"""
Solve one sign-change bracket using Chandrupatla (1997).
Call the batched solver with one interval and propagate its failures.
Absolute coordinate tolerances use the same units as the supplied bounds.
"""

from collections.abc import Callable

import numpy as np

from .BatchedBracket import solveBatchedBrackets


def solveBracket(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    xtol: float = 2.0e-12,
    rtol: float = 4.0 * np.finfo(float).eps,
    maxiter: int = 100,
) -> float:
    """
    Return a bracketed scalar root; invalid or exhausted solves raise.

    The keyword names preserve the source call sites' coordinate tolerances.
    The function must be finite on a continuous sign-change interval.
    """

    def evaluate(values: np.ndarray) -> np.ndarray:
        """
        Evaluate each independent scalar coordinate without changing the bracket algorithm.
        """

        return np.array([function(float(value)) for value in values])

    return float(solveBatchedBrackets(
        evaluate,
        np.array([left], dtype=float),
        np.array([right], dtype=float),
        relativeTolerance=rtol,
        absoluteTolerance=xtol,
        maxIterations=maxiter,
    )[0])
