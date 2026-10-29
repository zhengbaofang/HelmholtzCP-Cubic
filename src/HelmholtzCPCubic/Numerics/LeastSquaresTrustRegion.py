"""
Solve bounded nonlinear least-squares problems with a trust-region method.
Compute steps from the Jacobian SVD without forming normal equations.
Reuse the factorization for rejected trials and refresh it after an accepted step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray
ResidualJacobian = Callable[[Array], tuple[Array, Array]]


@dataclass(frozen=True)
class LeastSquaresTrustRegionResult:
    """
    Final state and Jacobian, termination reason and rejected-step counts.
    """

    state: Array
    residual: Array
    jacobian: Array
    rNorm: float
    iterations: int
    evaluations: int
    converged: bool
    termination: str
    rejectedSteps: int = 0
    infeasibleEvaluations: int = 0
    subproblemFailures: int = 0
    cauchySteps: int = 0
    boundaryReflections: int = 0


def thinSvdStep(
    jacobian: Array,
    residual: Array,
    radius: float,
    *,
    maxIter: int = 60,
    relTol: float = 1.0e-10,
) -> tuple[Array, float, bool]:
    r"""
    Solve ``min ||J s + r||`` subject to ``||s|| <= radius``.

    The secular equation is bracketed before safeguarded Newton iterations.
    No explicit ``J.T @ J`` matrix is used to calculate the step.
    """

    return _thinSvdStep(jacobian, residual, radius, maxIter=maxIter, relTol=relTol)


def _thinSvdStep(
    jacobian: Array,
    residual: Array,
    radius: float,
    *,
    maxIter: int = 60,
    relTol: float = 1.0e-10,
    factorCache: dict[str, tuple[Array, ...]] | None = None,
) -> tuple[Array, float, bool]:
    r"""
    Solve ``min ||J s + r||`` subject to ``||s|| <= radius``.

    The secular equation is bracketed before safeguarded Newton iterations.
    No explicit ``J.T @ J`` matrix is used to calculate the step.
    """

    matrix = np.asarray(jacobian, dtype=float)
    values = np.asarray(residual, dtype=float)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("Jacobian and residual dimensions are inconsistent")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("Jacobian and residual must be finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust-region radius must be finite and positive")

    try:

        # The outer solver owns this cache until its current model changes.
        if factorCache is None:
            left, singular, rightT = np.linalg.svd(matrix, full_matrices=False)
        else:
            if "factors" not in factorCache:
                factorCache["factors"] = np.linalg.svd(matrix, full_matrices=False)
            left, singular, rightT = factorCache["factors"]
    except np.linalg.LinAlgError:
        return _cauchyStep(matrix, values, radius), 0.0, False

    projected = left.T @ values
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros(matrix.shape[1]), 0.0, False
    threshold = np.finfo(float).eps * max(matrix.shape) * float(singular[0])
    active = singular > threshold
    unconstrained = np.zeros_like(singular)
    unconstrained[active] = -projected[active] / singular[active]
    unconstrainedStep = rightT.T @ unconstrained
    if np.linalg.norm(unconstrainedStep) <= radius:
        return unconstrainedStep, 0.0, True

    numerator = singular * projected
    singularSquared = singular**2

    def diagonalStep(multiplier: float) -> tuple[Array, float, float]:
        """
        Evaluate the regularized step in the Jacobian singular-vector basis.
        """

        denominator = singularSquared + multiplier
        coordinates = -numerator / denominator
        norm = float(np.linalg.norm(coordinates))
        derivative = (
            0.0
            if norm == 0.0
            else -float(np.sum(numerator**2 / denominator**3)) / norm
        )
        return coordinates, norm, derivative

    lower = 0.0
    upper = max(float(singular[0] ** 2), 1.0e-16)
    for _ in range(100):
        _coordinates, norm, _derivative = diagonalStep(upper)
        if norm <= radius:
            break
        upper *= 2.0
    else:
        return _cauchyStep(matrix, values, radius), upper, False

    multiplier = 0.5 * (lower + upper)
    converged = False
    coordinates = unconstrained
    for _ in range(maxIter):
        coordinates, norm, derivative = diagonalStep(multiplier)
        difference = norm - radius
        if abs(difference) <= relTol * radius:
            converged = True
            break
        if difference > 0.0:
            lower = multiplier
        else:
            upper = multiplier
        newton = multiplier - difference / derivative if derivative < 0.0 else np.nan
        multiplier = (
            newton
            if np.isfinite(newton) and lower < newton < upper
            else 0.5 * (lower + upper)
        )

    if not converged:
        coordinates, norm, _derivative = diagonalStep(upper)
        converged = abs(norm - radius) <= 1.0e-7 * radius
        multiplier = upper
        if not converged:
            return _cauchyStep(matrix, values, radius), multiplier, False
    step = rightT.T @ coordinates
    stepNorm = float(np.linalg.norm(step))
    if stepNorm > radius:
        step *= radius / stepNorm
    return step, multiplier, converged


def reflectStep(
    step: Array,
    state: Array,
    lower: Sequence[float] | Array,
    upper: Sequence[float] | Array,
) -> tuple[Array, dict[str, object]]:
    """
    Reflect a step at box boundaries while preserving its path length.
    """

    proposed = np.asarray(step, dtype=float)
    origin = np.asarray(state, dtype=float)
    lowerArray = np.asarray(lower, dtype=float)
    upperArray = np.asarray(upper, dtype=float)
    if not (proposed.shape == origin.shape == lowerArray.shape == upperArray.shape):
        raise ValueError("step, state and bounds must have identical shapes")
    if np.any(lowerArray > origin) or np.any(origin > upperArray):
        raise ValueError("state must lie inside the bounds")
    if np.any(lowerArray >= upperArray):
        raise ValueError("each lower bound must be smaller than its upper bound")

    tolerance = 64.0 * np.finfo(float).eps
    if np.all(origin + proposed >= lowerArray - tolerance) and np.all(
        origin + proposed <= upperArray + tolerance
    ):
        return proposed.copy(), {"reflections": 0, "activeBounds": ()}

    travelled = np.zeros_like(proposed)
    remaining = proposed.copy()
    activeBounds: list[tuple[int, str]] = []
    reflections = 0
    for _ in range(4 * proposed.size + 4):
        if np.linalg.norm(remaining) <= tolerance:
            break
        current = origin + travelled
        fraction = 1.0
        hits: list[tuple[int, str]] = []
        for index, component in enumerate(remaining):
            if component > tolerance:
                candidate = (upperArray[index] - current[index]) / component
                side = "upper"
            elif component < -tolerance:
                candidate = (lowerArray[index] - current[index]) / component
                side = "lower"
            else:
                continue
            if candidate < fraction - tolerance:
                fraction = max(0.0, candidate)
                hits = [(index, side)]
            elif abs(candidate - fraction) <= tolerance and candidate <= 1.0:
                hits.append((index, side))

        fraction = min(max(fraction, 0.0), 1.0)
        travelled += fraction * remaining
        if fraction >= 1.0 - tolerance:
            remaining.fill(0.0)
            break

        # Reflect only the untravelled segment so the consumed distance is
        # not counted again after a boundary hit.
        remaining *= 1.0 - fraction
        if not hits:
            break
        for index, side in hits:
            remaining[index] *= -1.0
            activeBounds.append((index, side))
            reflections += 1

    endpoint = np.minimum(np.maximum(origin + travelled, lowerArray), upperArray)
    reflected = endpoint - origin
    return reflected, {
        "reflections": reflections,
        "activeBounds": tuple(activeBounds),
    }


def solveLeastSquares(
    evaluate: ResidualJacobian,
    initial: Array,
    lower: Array,
    upper: Array,
    *,
    resTol: float,
    stepTol: float,
    maxEval: int,
    initialRadius: float,
    maxRadius: float,
    acceptanceRatio: float,
    stagnationRelativeTol: float = 1.0e-12,
    maxStagnantEvaluations: int = 12,
) -> LeastSquaresTrustRegionResult:
    """
    Solve bounded nonlinear least squares with a thin-SVD TR step.

    ``stagnationRelativeTol`` defines meaningful decrease relative to the
    current least-squares cost. Consecutive valid trials below that threshold,
    including ratio-rejected trials, terminate after
    ``maxStagnantEvaluations``. Failed callback evaluations consume the
    function-evaluation budget and are reported in the result.

    Raises:
        ValueError: If the state, bounds, tolerances, evaluation budget, trust
            radii, or acceptance ratio do not define a finite valid problem.
    """

    state = np.asarray(initial, dtype=float).copy()
    lowerArray = np.asarray(lower, dtype=float)
    upperArray = np.asarray(upper, dtype=float)
    if not (state.shape == lowerArray.shape == upperArray.shape):
        raise ValueError("initial state and bounds must have identical shapes")
    if state.ndim != 1 or state.size == 0:
        raise ValueError("initial state and bounds must be nonempty vectors")
    if not (
        np.all(np.isfinite(state))
        and np.all(np.isfinite(lowerArray))
        and np.all(np.isfinite(upperArray))
    ):
        raise ValueError("initial state and bounds must be finite")
    if np.any(lowerArray >= upperArray):
        raise ValueError("each lower bound must be smaller than its upper bound")
    if np.any(state < lowerArray) or np.any(state > upperArray):
        raise ValueError("initial state must lie inside the bounds")
    for name, value in (("resTol", resTol), ("stepTol", stepTol)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not isinstance(maxEval, int)
        or isinstance(maxEval, bool)
        or maxEval < 1
    ):
        raise ValueError("maxEval must be a positive integer")
    if not (
        math.isfinite(initialRadius)
        and math.isfinite(maxRadius)
        and 0.0 < initialRadius <= maxRadius
    ):
        raise ValueError("trust-region radii must satisfy 0 < initial <= maximum")
    if not math.isfinite(acceptanceRatio) or not (0.0 <= acceptanceRatio < 1.0):
        raise ValueError("acceptanceRatio must lie in [0, 1)")
    if (
        isinstance(stagnationRelativeTol, bool)
        or not math.isfinite(stagnationRelativeTol)
        or stagnationRelativeTol <= 0.0
    ):
        raise ValueError("stagnationRelativeTol must be finite and positive")
    if (
        not isinstance(maxStagnantEvaluations, int)
        or isinstance(maxStagnantEvaluations, bool)
        or maxStagnantEvaluations < 1
    ):
        raise ValueError("maxStagnantEvaluations must be a positive integer")
    radius = initialRadius
    residual, jacobian = _evaluateResidual(evaluate, state)
    evaluations = 1
    iterations = 0
    rejectedSteps = 0
    infeasibleEvaluations = 0
    subproblemFailures = 0
    cauchySteps = 0
    boundaryReflections = 0
    stagnantEvaluations = 0
    maximumColumnNorms = np.zeros(state.size)

    def finish(
        norm: float,
        converged: bool,
        termination: str,
    ) -> LeastSquaresTrustRegionResult:
        """
        Return the retained state and convergence and rejection details.
        """

        return LeastSquaresTrustRegionResult(
            state=state,
            residual=residual,
            jacobian=jacobian,
            rNorm=norm,
            iterations=iterations,
            evaluations=evaluations,
            converged=converged,
            termination=termination,
            rejectedSteps=rejectedSteps,
            infeasibleEvaluations=infeasibleEvaluations,
            subproblemFailures=subproblemFailures,
            cauchySteps=cauchySteps,
            boundaryReflections=boundaryReflections,
        )

    initialNorm = float(np.linalg.norm(residual))
    if initialNorm <= resTol:
        return finish(initialNorm, True, "residual")

    # Retain successful step factors only while the current state is unchanged.
    factorCache = {}
    while evaluations < maxEval:
        norm = float(np.linalg.norm(residual))
        if norm <= resTol:
            return finish(norm, True, "residual")

        # MINPACK/SciPy-style Jacobian column scaling makes the Euclidean TR
        # radius meaningful when log(T) and log(rho) sensitivities differ by
        # orders of magnitude.  The maxima are retained to avoid oscillatory
        # rescaling between rejected steps.
        columnNorms = np.linalg.norm(jacobian, axis=0)
        maximumColumnNorms = np.maximum(maximumColumnNorms, columnNorms)
        variableScale = 1.0 / np.maximum(maximumColumnNorms, 1.0e-14)
        scaledJacobian = jacobian * variableScale[np.newaxis, :]
        scaledStep, _multiplier, subproblemOk = _thinSvdStep(
            scaledJacobian, residual, radius, factorCache=factorCache,
        )
        rawStep = variableScale * scaledStep
        step, reflection = reflectStep(rawStep, state, lowerArray, upperArray)
        boundaryReflections += int(reflection["reflections"])
        stepNorm = float(np.linalg.norm(step))
        stepLimited = (
            stepNorm
            <= stepTol * max(1.0, float(np.linalg.norm(state)))
        )

        linearized = residual + jacobian @ step
        predicted = 0.5 * (float(residual @ residual) - float(linearized @ linearized))
        if not subproblemOk:
            subproblemFailures += 1
            if stepNorm > 0.0:
                cauchySteps += 1
        if predicted <= 0.0 or not np.isfinite(predicted):
            if not subproblemOk:
                return finish(norm, False, "linear_algebra_failure")
            if stepLimited:
                return finish(norm, False, "step_tolerance")
            radius *= 0.25
            iterations += 1
            if radius <= stepTol:
                break
            continue

        trialState = state + step

        # An attempted callback consumes the evaluation budget even when the
        # trial state is outside the model domain and evaluation fails.
        evaluations += 1
        try:
            trialResidual, trialJacobian = _evaluateResidual(evaluate, trialState)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            infeasibleEvaluations += 1
            rejectedSteps += 1
            radius *= 0.25
            iterations += 1
            if radius <= stepTol:
                break
            continue

        cost = 0.5 * float(residual @ residual)
        trialCost = 0.5 * float(trialResidual @ trialResidual)
        ratio = (cost - trialCost) / predicted
        if ratio < 0.25:
            radius = max(0.25 * radius, stepTol)
        elif (
            ratio > 0.75
            and np.linalg.norm(scaledStep) >= 0.8 * radius
            and not reflection["reflections"]
        ):
            radius = min(2.0 * radius, maxRadius)

        accepted = ratio > acceptanceRatio
        if accepted:
            reduction = cost - trialCost
            state = trialState
            residual = trialResidual
            jacobian = trialJacobian

            # New derivatives invalidate the previous model factorization.
            factorCache.clear()
            if reduction > stagnationRelativeTol * cost:
                stagnantEvaluations = 0
            else:
                stagnantEvaluations += 1
        else:
            rejectedSteps += 1
            stagnantEvaluations += 1
        iterations += 1
        norm = float(np.linalg.norm(residual))
        if norm <= resTol:
            return finish(norm, True, "residual")
        if stagnantEvaluations >= maxStagnantEvaluations:
            return finish(norm, False, "insufficient_progress")
        if stepLimited:
            return finish(norm, False, "step_tolerance")

    norm = float(np.linalg.norm(residual))
    if norm <= resTol:
        termination = "residual"
    elif evaluations >= maxEval:
        termination = "maximum_evaluations"
    else:
        termination = "radius_tolerance"
    return finish(norm, norm <= resTol, termination)


def _cauchyStep(jacobian: Array, residual: Array, radius: float) -> Array:
    """
    Return the exact minimizer of the linear model along ``-J.T @ r``.
    """

    gradient = jacobian.T @ residual
    gradientNorm = float(np.linalg.norm(gradient))
    if gradientNorm == 0.0:
        return np.zeros(jacobian.shape[1])
    direction = gradient / gradientNorm
    projected = jacobian @ direction
    curvature = float(projected @ projected)
    if curvature <= 0.0 or not math.isfinite(curvature):
        stepLength = radius
    else:
        stepLength = min(gradientNorm / curvature, radius)
    return -stepLength * direction


def _evaluateResidual(evaluate: ResidualJacobian, state: Array) -> tuple[Array, Array]:
    """
    Evaluate the residual and Jacobian; reject inconsistent shapes or non-finite values.
    """

    residual, jacobian = evaluate(state)
    residualArray = np.asarray(residual, dtype=float)
    jacobianArray = np.asarray(jacobian, dtype=float)
    if residualArray.ndim != 1 or jacobianArray.shape != (
        residualArray.size,
        state.size,
    ):
        raise ValueError("residual/Jacobian callback returned inconsistent shapes")
    if not np.all(np.isfinite(residualArray)) or not np.all(np.isfinite(jacobianArray)):
        raise FloatingPointError("residual/Jacobian callback returned non-finite values")
    return residualArray, jacobianArray
