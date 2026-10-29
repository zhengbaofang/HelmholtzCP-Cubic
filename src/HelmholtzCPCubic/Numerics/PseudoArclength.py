"""
Model-independent pseudo-arclength predictor-corrector utilities.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .LeastSquaresTrustRegion import (
    LeastSquaresTrustRegionResult,
    solveLeastSquares,
)


Array = np.ndarray
ManifoldEvaluation = Callable[[Array], tuple[Array, Array]]


def nullSpaceTangent(
    jacobian: Array,
    previousTangent: Array | None = None,
) -> Array:
    """
    Return a normalized, consistently oriented null vector of a codimension-one Jacobian.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != matrix.shape[0] + 1:
        raise ValueError("pseudo-arclength Jacobian must have shape (n, n + 1)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pseudo-arclength Jacobian must be finite")
    _left, singular, rightT = np.linalg.svd(matrix, full_matrices=True)
    if singular.size == 0 or singular[0] == 0.0:
        raise np.linalg.LinAlgError("manifold Jacobian has no resolved row space")
    threshold = np.finfo(float).eps * max(matrix.shape) * float(singular[0])
    if int(np.count_nonzero(singular > threshold)) != matrix.shape[0]:
        raise np.linalg.LinAlgError("manifold Jacobian is rank deficient")
    tangent = rightT[-1].copy()
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0 or not np.isfinite(norm):
        raise np.linalg.LinAlgError("could not determine a finite manifold tangent")
    tangent /= norm
    if previousTangent is not None:
        previous = np.asarray(previousTangent, dtype=float)
        if previous.shape != tangent.shape:
            raise ValueError("previousTangent has an inconsistent shape")
        if float(tangent @ previous) < 0.0:
            tangent *= -1.0
    return tangent


def correctPseudoArclength(
    evaluateManifold: ManifoldEvaluation,
    predictor: Array,
    tangent: Array,
    lower: Array,
    upper: Array,
    *,
    resTol: float,
    stepTol: float,
    maxEval: int,
    initialRadius: float,
    maxRadius: float,
    acceptanceRatio: float,
) -> LeastSquaresTrustRegionResult:
    """
    Correct a predictor on the hyperplane normal to its tangent.

    The returned Jacobian is the final augmented matrix: the manifold rows
    followed by the normalized predictor-tangent row.
    """

    predicted = np.asarray(predictor, dtype=float)
    direction = np.asarray(tangent, dtype=float)
    if predicted.ndim != 1 or direction.shape != predicted.shape:
        raise ValueError("predictor and tangent must be one-dimensional arrays of equal shape")
    tangentNorm = float(np.linalg.norm(direction))
    if tangentNorm == 0.0 or not np.isfinite(tangentNorm):
        raise ValueError("tangent must have a finite positive norm")
    direction = direction / tangentNorm

    def evaluateAugmented(state: Array) -> tuple[Array, Array]:
        """
        Append the dimensionless tangent-plane constraint to the original residual.
        """

        residual, jacobian = evaluateManifold(state)
        residualArray = np.asarray(residual, dtype=float)
        jacobianArray = np.asarray(jacobian, dtype=float)
        if residualArray.shape != (state.size - 1,) or jacobianArray.shape != (
            state.size - 1,
            state.size,
        ):
            raise ValueError("manifold callback must return n-1 residuals for n state variables")
        arclength = float(direction @ (state - predicted))
        return (
            np.concatenate((residualArray, [arclength])),
            np.vstack((jacobianArray, direction)),
        )

    return solveLeastSquares(
        evaluateAugmented,
        predicted,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        resTol=resTol,
        stepTol=stepTol,
        maxEval=maxEval,
        initialRadius=initialRadius,
        maxRadius=maxRadius,
        acceptanceRatio=acceptanceRatio,
    )
