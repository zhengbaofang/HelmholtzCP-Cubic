"""
Locate where a traced spinodal crosses an edge of the search region.
Correct the crossing against the spinodal equation and the fixed edge value.
Coordinates are logarithmic temperature and density, as supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from ..Numerics.LeastSquaresTrustRegion import solveLeastSquares


_ManifoldEvaluation = Callable[
    [np.ndarray],
    tuple[np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class BoundaryCrossingDiagnostic:
    """
    Result and work evidence for one inset-edge crossing correction.

    Parameters
    ----------
    resolved:
        Whether the augmented manifold equations converged within tolerance.
    logState:
        Corrected ``(ln(T), ln(rho))`` state, or ``None`` after failure.
    fixedAxis:
        Fixed logarithmic coordinate: zero for temperature, one for density.
    logEdge:
        Dimensionless natural-log value of the crossed inset edge.
    nEval:
        Number of augmented residual and Jacobian evaluations.
    rNorm:
        Dimensionless least-squares residual norm; infinity if unavailable.
    termination:
        Exact numerical termination or failure description.
    """

    resolved: bool
    logState: tuple[float, float] | None
    fixedAxis: int
    logEdge: float
    nEval: int
    rNorm: float
    termination: str


def _locateBoundaryCrossing(
    evaluateManifold: _ManifoldEvaluation,
    previousState: np.ndarray,
    currentState: np.ndarray,
    fixedAxis: int,
    logEdge: float,
    lowerBound: np.ndarray,
    upperBound: np.ndarray,
    *,
    resTol: float,
    stepTol: float,
    maxCorrectorEval: int,
    acceptanceRatio: float,
    maxRadius: float,
) -> BoundaryCrossingDiagnostic:
    """
    Correct one inset-edge crossing bracketed by accepted contour states.

    ``previousState``, ``currentState``, ``lowerBound``, and ``upperBound``
    contain dimensionless ``(ln(T), ln(rho))`` coordinates. The corrector is
    restricted to a local box around the accepted segment. Invalid geometry,
    failed model evaluation, or non-convergence is returned explicitly in the
    diagnostic rather than replaced by an approximate segment intersection.
    """

    startState = np.asarray(previousState, dtype=float)
    endState = np.asarray(currentState, dtype=float)
    lower = np.asarray(lowerBound, dtype=float)
    upper = np.asarray(upperBound, dtype=float)
    if not (
        startState.shape == endState.shape == lower.shape == upper.shape == (2,)
    ):
        raise ValueError("boundary crossing requires four two-state vectors")
    if fixedAxis not in (0, 1):
        raise ValueError("fixedAxis must be zero or one")
    if not math.isfinite(logEdge):
        raise ValueError("logEdge must be finite")

    startSide = startState[fixedAxis] - logEdge
    endSide = endState[fixedAxis] - logEdge
    if startSide == endSide or startSide * endSide > 0.0:
        return BoundaryCrossingDiagnostic(
            resolved=False,
            logState=None,
            fixedAxis=fixedAxis,
            logEdge=logEdge,
            nEval=0,
            rNorm=math.inf,
            termination="accepted_segment_does_not_cross_edge",
        )

    segment = endState - startState
    segmentLength = float(np.linalg.norm(segment))
    if not math.isfinite(segmentLength) or segmentLength <= stepTol:
        return BoundaryCrossingDiagnostic(
            resolved=False,
            logState=None,
            fixedAxis=fixedAxis,
            logEdge=logEdge,
            nEval=0,
            rNorm=math.inf,
            termination="accepted_segment_unresolved",
        )
    fraction = startSide / (startSide - endSide)
    initialState = startState + fraction * segment
    initialState[fixedAxis] = logEdge
    localLower = np.maximum(
        lower,
        np.minimum(startState, endState) - segmentLength,
    )
    localUpper = np.minimum(
        upper,
        np.maximum(startState, endState) + segmentLength,
    )
    if np.any(localLower >= localUpper):
        return BoundaryCrossingDiagnostic(
            resolved=False,
            logState=None,
            fixedAxis=fixedAxis,
            logEdge=logEdge,
            nEval=0,
            rNorm=math.inf,
            termination="crossing_box_degenerate",
        )

    nEval = 0

    def evaluateBoundary(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment the contour manifold with one fixed-edge equation.
        """

        nonlocal nEval
        nEval += 1
        residual, jacobian = evaluateManifold(state)
        boundaryRow = np.zeros((1, state.size), dtype=float)
        boundaryRow[0, fixedAxis] = 1.0
        return (
            np.concatenate(
                (
                    np.asarray(residual, dtype=float),
                    [state[fixedAxis] - logEdge],
                )
            ),
            np.vstack((np.asarray(jacobian, dtype=float), boundaryRow)),
        )

    radius = min(segmentLength, maxRadius)
    try:
        result = solveLeastSquares(
            evaluateBoundary,
            initialState,
            localLower,
            localUpper,
            resTol=resTol,
            stepTol=stepTol,
            maxEval=maxCorrectorEval,
            initialRadius=radius,
            maxRadius=radius,
            acceptanceRatio=acceptanceRatio,
        )
    except (
        FloatingPointError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return BoundaryCrossingDiagnostic(
            resolved=False,
            logState=None,
            fixedAxis=fixedAxis,
            logEdge=logEdge,
            nEval=nEval,
            rNorm=math.inf,
            termination=f"crossing_failed: {type(error).__name__}: {error}",
        )

    logState = (
        tuple(float(value) for value in result.state)
        if result.converged
        else None
    )
    return BoundaryCrossingDiagnostic(
        resolved=result.converged,
        logState=logState,
        fixedAxis=fixedAxis,
        logEdge=logEdge,
        nEval=nEval,
        rNorm=result.rNorm,
        termination="crossing_resolved" if result.converged else result.termination,
    )


__all__ = ["BoundaryCrossingDiagnostic"]
