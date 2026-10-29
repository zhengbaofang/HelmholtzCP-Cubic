r"""
Minimize a scalar objective from its gradient and Hessian within box bounds.
Solve the trust-region subproblem using More and Sorensen (1983).
Support optional diagonal scaling and caller-supplied convergence checks.
The objective is dimensionless; step lengths use the supplied coordinate metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray
ScalarObjective = Callable[[Array], tuple[float, Array, Array] | None]
ScalarConvergenceCheck = Callable[[Array, float, Array, Array], bool]


@dataclass(frozen=True)
class ScalarTrustRegionConfig:
    """
    Numerical controls for one dimensionless scalar minimization.

    ``stationarityTol`` applies to the default Euclidean gradient norm (1e-8).
    A caller convergence check supplies its own residual and curvature rules.
    ``diagonalScaling`` selects the metric D=sqrt(abs(diag(H))); it is off
    by default. With scaling enabled, radii use ||D p||. Zero diagonal entries
    retain unit scale. ``stepTol`` still checks the supplied coordinate step.
    ``fractionToBoundary`` keeps open-bound variables strictly feasible.
    """

    stationarityTol: float = 1.0e-8
    stepTol: float = 1.0e-10
    maxEval: int = 160
    initialRadius: float = 1.0
    maxRadius: float = 20.0
    acceptanceRatio: float = 0.05
    fractionToBoundary: float = 0.995
    diagonalScaling: bool = False

    def __post_init__(self) -> None:
        """
        Reject inconsistent tolerances, limits, radii, and ratios.
        """

        if not math.isfinite(self.stationarityTol) or self.stationarityTol <= 0.0:
            raise ValueError("stationarityTol must be finite and positive")
        if not math.isfinite(self.stepTol) or self.stepTol <= 0.0:
            raise ValueError("stepTol must be finite and positive")
        if not isinstance(self.maxEval, int) or self.maxEval < 1:
            raise ValueError("maxEval must be a positive integer")
        if not (
            math.isfinite(self.initialRadius)
            and math.isfinite(self.maxRadius)
            and 0.0 < self.initialRadius <= self.maxRadius
        ):
            raise ValueError("trust-region radii must satisfy 0 < initial <= maximum")
        if not math.isfinite(self.acceptanceRatio) or not (
            0.0 <= self.acceptanceRatio < 0.25
        ):
            raise ValueError("acceptanceRatio must lie in [0, 0.25)")
        if not math.isfinite(self.fractionToBoundary) or not (
            0.0 < self.fractionToBoundary < 1.0
        ):
            raise ValueError("fractionToBoundary must lie in (0, 1)")
        if not isinstance(self.diagonalScaling, bool):
            raise ValueError("diagonalScaling must be a boolean")


@dataclass(frozen=True)
class ScalarTrustRegionResult:
    """
    Final minimization state, residual measures and termination reason.
    """

    state: Array
    value: float
    gradient: Array
    hessian: Array
    stationarityNorm: float
    lambdaMin: float
    iterations: int
    evaluations: int
    converged: bool
    termination: str

    def __post_init__(self) -> None:
        """
        Freeze finite result arrays and reject inconsistent counters.
        """

        state = np.asarray(self.state, dtype=float).copy()
        gradient = np.asarray(self.gradient, dtype=float).copy()
        hessian = np.asarray(self.hessian, dtype=float).copy()
        if state.ndim != 1 or gradient.shape != state.shape:
            raise ValueError("scalar result state and gradient must be equal vectors")
        if hessian.shape != (state.size, state.size):
            raise ValueError("scalar result Hessian has inconsistent dimensions")
        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(gradient))
            and np.all(np.isfinite(hessian))
            and math.isfinite(self.value)
            and math.isfinite(self.stationarityNorm)
            and math.isfinite(self.lambdaMin)
        ):
            raise ValueError("scalar result contains NaN or Inf")

        # A rejected subproblem can shrink the radius without evaluating the
        # objective, so iteration count is intentionally independent of the
        # objective-evaluation count.
        if self.iterations < 0 or self.evaluations < 1:
            raise ValueError("scalar result counters are inconsistent")
        if not self.termination:
            raise ValueError("scalar result termination must be nonempty")
        state.setflags(write=False)
        gradient.setflags(write=False)
        hessian.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)


def minimizeScalar(
    objective: ScalarObjective,
    initial: Array,
    lower: Array,
    upper: Array,
    config: ScalarTrustRegionConfig | None = None,
    *,
    convergenceCheck: ScalarConvergenceCheck | None = None,
) -> ScalarTrustRegionResult:
    r"""
    Minimize an exact scalar model inside one open coordinate box.

    The callback returns ``(value, gradient, Hessian)`` or ``None`` for an
    infeasible model state. A small gradient norm does not stop at an
    indefinite stationary point: significant negative curvature is sent back
    to the same Moré--Sorensen subproblem.

    ``convergenceCheck(state, value, gradient, Hessian)`` replaces the default
    gradient-norm and curvature success test when supplied. The caller must
    qualify both stationarity and minimum curvature, including any physical
    residuals, without changing the supplied arrays. The same check controls
    current-point and trial-point success. A qualified trial returns directly;
    objective-decrease checks govern trials that have not yet converged.
    A roundoff-level progress step can be retained while the convergence check
    is still false; it does not relax convergence.
    Reported gradient norm and minimum eigenvalue remain in the supplied coordinates.

    The default metric is Euclidean. Optional diagonal scaling changes the step
    metric only. A fraction-to-boundary step keeps open bounds feasible.
    Reuse the Hessian factorization for retries at one state and refresh it after
    an accepted state. Near stationarity, a gradient-reducing trial may be accepted
    when predicted and actual changes are both within the objective roundoff scale.
    The caller convergence check still determines success.
    """

    controls = config or ScalarTrustRegionConfig()
    state = np.asarray(initial, dtype=float).copy()
    lowerArray = np.asarray(lower, dtype=float)
    upperArray = np.asarray(upper, dtype=float)
    if not (state.shape == lowerArray.shape == upperArray.shape) or state.ndim != 1:
        raise ValueError("initial state and bounds must be equal-length vectors")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial scalar state must be finite")
    if np.any(np.isnan(lowerArray)) or np.any(np.isnan(upperArray)):
        raise ValueError("scalar bounds cannot contain NaN")
    if np.any(lowerArray >= upperArray):
        raise ValueError("each lower scalar bound must be smaller than its upper bound")
    if np.any(state <= lowerArray) or np.any(state >= upperArray):
        raise ValueError("initial scalar state must lie strictly inside its bounds")

    current = _evaluateObjective(objective, state)
    if current is None:
        raise ValueError("initial scalar state is infeasible")
    value, gradient, hessian = current
    radius = controls.initialRadius
    evaluations = 1
    iterations = 0

    # Retain successful step factors only while the current state is unchanged.
    factorCache = {}
    while True:
        stationarityNorm = _stableNorm(gradient)
        lambdaMin = float(np.linalg.eigvalsh(hessian)[0])
        curvatureTol = _curvatureTolerance(hessian)
        converged = (
            bool(convergenceCheck(state, value, gradient, hessian))
            if convergenceCheck is not None
            else stationarityNorm <= controls.stationarityTol
            and lambdaMin >= -curvatureTol
        )
        if converged:
            return _result(
                state,
                value,
                gradient,
                hessian,
                stationarityNorm,
                lambdaMin,
                iterations,
                evaluations,
                True,
                (
                    "caller_convergence" if convergenceCheck is not None
                    else "stationarity_and_curvature"
                ),
            )
        if evaluations >= controls.maxEval:
            break

        try:
            if controls.diagonalScaling:

                # Solve in scaled coordinates s=D*p, then map the step back.
                # A zero diagonal supplies no scale; retain that coordinate's
                # unit metric without clipping nonzero curvature or adding damping.
                diagonalScale = np.sqrt(np.abs(np.diag(hessian)))
                diagonalScale = np.where(diagonalScale > 0.0, diagonalScale, 1.0)
                scaledHessian = hessian / (diagonalScale[:, None] * diagonalScale[None, :])
                step = _nearExactStep(
                    gradient / diagonalScale, scaledHessian, radius,
                    factorCache=factorCache,
                ) / diagonalScale
            else:
                step = _nearExactStep(gradient, hessian, radius, factorCache=factorCache)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            radius *= 0.25
            iterations += 1
            if radius <= controls.stepTol:
                break
            continue
        step = _fractionToBoundary(
            state,
            step,
            lowerArray,
            upperArray,
            controls.fractionToBoundary,
        )
        stepNorm = _stableNorm(step)

        # Bounds can shorten the physical step; test the actual step in the
        # same metric used for the trust-region subproblem.
        trustStepNorm = _stableNorm(diagonalScale * step) if controls.diagonalScaling else stepNorm
        stepLimited = stepNorm <= controls.stepTol * max(1.0, _stableNorm(state))
        predicted = -float(gradient @ step + 0.5 * step @ hessian @ step)
        if predicted <= 0.0 or not math.isfinite(predicted):
            if stepLimited:
                break
            radius *= 0.25
            iterations += 1
            if radius <= controls.stepTol:
                break
            continue

        trialState = state + step
        evaluations += 1
        trial = _evaluateObjective(objective, trialState)
        if trial is None:
            radius *= 0.25
            iterations += 1
            if radius <= controls.stepTol:
                break
            continue
        trialValue, trialGradient, trialHessian = trial
        trialNorm = _stableNorm(trialGradient)
        trialLambdaMin = float(np.linalg.eigvalsh(trialHessian)[0])
        trialConverged = (
            bool(convergenceCheck(trialState, trialValue, trialGradient, trialHessian))
            if convergenceCheck is not None
            else trialNorm <= controls.stationarityTol
            and trialLambdaMin >= -_curvatureTolerance(trialHessian)
        )

        # Complete local qualification ends this search. Reuse the evaluated
        # trial; objective cancellation must not veto the caller's residuals.
        if trialConverged:
            return _result(
                trialState,
                trialValue,
                trialGradient,
                trialHessian,
                trialNorm,
                trialLambdaMin,
                iterations + 1,
                evaluations,
                True,
                (
                    "trial_caller_convergence" if convergenceCheck is not None
                    else "trial_stationarity_and_curvature"
                ),
            )
        actual = value - trialValue
        ratio = actual / predicted

        # Near a zero TPD minimum the scalar value is a cancellation of two
        # free-energy values and an Nc-term dot product.  Its forward-error
        # allowance therefore scales with vector length. Keep physical success
        # separate from accepting useful progress within that value uncertainty.
        roundoffTol = 64.0 * np.finfo(float).eps * max(1, state.size) * max(
            1.0,
            abs(value),
            abs(trialValue),
        )
        accepted = bool(ratio > controls.acceptanceRatio and trialValue <= value)
        roundoffProgress = bool(
            not accepted
            and predicted <= roundoffTol  # Positive prediction was checked above.
            and abs(actual) <= roundoffTol
            and trialNorm < stationarityNorm
            and trialLambdaMin >= -_curvatureTolerance(trialHessian)
        )
        if roundoffProgress:
            accepted = True

        # A ratio formed from roundoff-sized value changes cannot justify a
        # radius change for this progress step. Ordinary steps keep their rule.
        # Unqualified rejected trials still shrink the radius.
        if not roundoffProgress:
            if ratio < 0.25:
                radius = max(0.25 * radius, controls.stepTol)
            elif ratio > 0.75 and trustStepNorm >= 0.8 * radius:
                radius = min(2.0 * radius, controls.maxRadius)
        if accepted:
            state = trialState
            value = trialValue
            gradient = trialGradient
            hessian = trialHessian

            # New derivatives invalidate the previous model factorization.
            factorCache.clear()
        iterations += 1
        if stepLimited or radius <= controls.stepTol:
            break

    stationarityNorm = _stableNorm(gradient)
    lambdaMin = float(np.linalg.eigvalsh(hessian)[0])
    return _result(
        state,
        value,
        gradient,
        hessian,
        stationarityNorm,
        lambdaMin,
        iterations,
        evaluations,
        False,
        (
            "maximum_evaluations"
            if evaluations >= controls.maxEval
            else "step_or_radius_tolerance"
        ),
    )


def _nearExactStep(
    gradient: Array,
    hessian: Array,
    radius: float,
    *,
    factorCache: dict[str, tuple[Array, ...]] | None = None,
    keasy: float = 0.1,
) -> Array:
    r"""
    Solve the dense Euclidean trust-region subproblem spectrally.

    The returned step approximately minimizes ``g.T p + 0.5 p.T H p`` subject to
    ``||p||_2 <= radius``. Boundary candidates must satisfy
    ``abs(||p||/radius-1) <= keasy`` before projection;
    ``keasy`` must be finite and in (0, 1). When the gradient is orthogonal to the leftmost
    eigenspace of an indefinite Hessian, the missing boundary component is
    supplied explicitly along that eigenspace (the Moré--Sorensen hard case).

    The relative boundary tolerance defaults to Keasy=0.1 (More and Sorensen,
    1983, Eq. 3.10). A safeguarded multiplier bracket produces a feasible,
    model-decreasing step; interior and hard-case steps retain the KKT checks.
    """

    gradient = np.asarray(gradient, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    if gradient.ndim != 1 or hessian.shape != (gradient.size, gradient.size):
        raise ValueError("gradient and Hessian dimensions are inconsistent")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise ValueError("gradient and Hessian must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust-region radius must be finite and positive")
    if not math.isfinite(keasy) or not 0.0 < keasy < 1.0:
        raise ValueError("keasy must be finite and in (0, 1)")
    symmetric = 0.5 * (hessian + hessian.T)

    # The outer solver owns this cache until its current model changes.
    if factorCache is None:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    else:
        if "factors" not in factorCache:
            factorCache["factors"] = np.linalg.eigh(symmetric)
        eigenvalues, eigenvectors = factorCache["factors"]
    coefficients = eigenvectors.T @ gradient
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    eigenTol = 32.0 * np.finfo(float).eps * scale
    lambdaMin = float(eigenvalues[0])
    gradientNorm = _stableNorm(gradient)

    def checkedStep(
        coordinates: Array, shift: float, *, approximateBoundary: bool = False,
    ) -> Array:
        """
        Qualify the requested approximate boundary or exact interior/hard case.

        Check Keasy before projecting an accepted boundary overshoot. A feasible
        endpoint alone cannot qualify an exhausted multiplier solve, and exact
        complementarity must not silently override the approximate criterion.
        """

        step = eigenvectors @ coordinates
        norm = _stableNorm(step)
        if not np.all(np.isfinite(step)) or not math.isfinite(norm):
            raise FloatingPointError("trust-region subproblem returned NaN or Inf")
        if approximateBoundary and abs(norm / radius - 1.0) > keasy:
            raise FloatingPointError("trust-region boundary failed Keasy qualification")
        if not approximateBoundary and norm > radius * (1.0 + 1.0e-10):
            raise FloatingPointError("trust-region subproblem exceeds its radius")
        if norm > radius:
            step *= radius / norm
            norm = _stableNorm(step)
        kktScale = max(gradientNorm + (scale + abs(shift)) * norm, np.finfo(float).tiny)
        complementarityScale = max(
            gradientNorm + (scale + abs(shift)) * radius, np.finfo(float).tiny,
        )
        if (
            shift < 0.0
            or lambdaMin + shift < -64.0 * np.finfo(float).eps * max(scale, abs(shift))
            or (not approximateBoundary and (
                _stableNorm(symmetric @ step + shift * step + gradient) > 1.0e-10 * kktScale
                or abs(shift * (radius - norm)) > 1.0e-9 * complementarityScale
            ))
        ):
            raise FloatingPointError("trust-region subproblem failed KKT qualification")
        decrease = -float(gradient @ step + 0.5 * step @ symmetric @ step)
        if not math.isfinite(decrease) or decrease < 0.0:
            raise FloatingPointError("trust-region subproblem has no finite model decrease")
        return step

    if lambdaMin > eigenTol:
        newton = -coefficients / eigenvalues
        if _stableNorm(newton) <= radius:
            return checkedStep(newton, 0.0)

    shiftFloor = max(0.0, -lambdaMin)
    shifted = eigenvalues + shiftFloor
    nullSpace = shifted <= eigenTol
    regular = ~nullSpace
    base = np.zeros_like(coefficients)
    base[regular] = -coefficients[regular] / shifted[regular]
    nullGradientNorm = _stableNorm(coefficients[nullSpace])
    gradientTol = 32.0 * np.finfo(float).eps * max(
        1.0,
        gradientNorm,
    )
    baseNorm = _stableNorm(base)
    if nullGradientNorm <= gradientTol and baseNorm <= radius:
        if shiftFloor == 0.0:
            return checkedStep(base, 0.0)
        hard = np.zeros_like(coefficients)
        boundaryLength = radius * math.sqrt(max(1.0 - (baseNorm / radius)**2, 0.0))
        if nullGradientNorm > 0.0:
            hard[nullSpace] = -boundaryLength * coefficients[nullSpace] / nullGradientNorm
        else:
            hard[int(np.flatnonzero(nullSpace)[0])] = boundaryLength
        return checkedStep(base + hard, shiftFloor)

    lowerShift = shiftFloor

    def shiftedStep(shift: float) -> tuple[Array, float]:
        """
        Evaluate a trial spectral step for the scalar-objective trust region.
        """

        denominators = eigenvalues + shift
        if np.any(denominators <= 0.0):
            return np.full_like(coefficients, np.nan), float("inf")
        values = -coefficients / denominators
        return values, _stableNorm(values)

    upperShift = max(
        lowerShift + scale,
        shiftFloor + gradientNorm / max(radius, np.finfo(float).tiny),
    )
    upperValues, upperNorm = shiftedStep(upperShift)
    for _ in range(80):
        if upperNorm <= radius:
            break
        upperShift = shiftFloor + 2.0 * (upperShift - shiftFloor)
        upperValues, upperNorm = shiftedStep(upperShift)
    else:
        raise FloatingPointError("could not bracket the trust-region multiplier")

    # Safeguarded Newton solves ||p(lambda)||/radius - 1 = 0. Its relative
    # residual measures boundary accuracy independently of radius. The upper
    # endpoint stores a feasible step; do not use an unchecked midpoint.
    best = upperValues
    shift = 0.5 * (lowerShift + upperShift)
    for _ in range(100):
        values, norm = shiftedStep(shift)
        ratio = norm / radius
        if math.isfinite(ratio) and abs(ratio - 1.0) <= keasy:
            return checkedStep(values, shift, approximateBoundary=True)
        if norm > radius:
            lowerShift = shift
        else:
            upperShift = shift
            best = values
        derivative = (
            -ratio * float(np.sum((values / norm)**2 / (eigenvalues + shift)))
            if norm > 0.0 and math.isfinite(ratio) else float("nan")
        )
        newtonShift = shift - (ratio - 1.0) / derivative if derivative < 0.0 else float("nan")
        shift = (
            newtonShift if lowerShift < newtonShift < upperShift
            else 0.5 * (lowerShift + upperShift)
        )
        if shift == lowerShift or shift == upperShift:
            break
    return checkedStep(best, upperShift, approximateBoundary=True)


def _fractionToBoundary(
    state: Array,
    step: Array,
    lower: Array,
    upper: Array,
    fraction: float,
) -> Array:
    """
    Shorten one step so every finite open bound remains strictly feasible.
    """

    factor = 1.0
    for index, direction in enumerate(step):
        if direction < 0.0 and math.isfinite(lower[index]):
            factor = min(
                factor,
                fraction * (state[index] - lower[index]) / (-direction),
            )
        elif direction > 0.0 and math.isfinite(upper[index]):
            factor = min(
                factor,
                fraction * (upper[index] - state[index]) / direction,
            )
    return np.asarray(step, dtype=float) * max(0.0, min(1.0, factor))


def _evaluateObjective(
    objective: ScalarObjective,
    state: Array,
) -> tuple[float, Array, Array] | None:
    """
    Validate one scalar-objective callback result and symmetrize its Hessian.
    """

    evaluated = objective(np.asarray(state, dtype=float))
    if evaluated is None:
        return None
    value, gradient, hessian = evaluated
    gradientArray = np.asarray(gradient, dtype=float)
    hessianArray = np.asarray(hessian, dtype=float)
    if gradientArray.shape != state.shape or hessianArray.shape != (
        state.size,
        state.size,
    ):
        raise ValueError("scalar objective returned inconsistent dimensions")
    if not (
        math.isfinite(float(value))
        and np.all(np.isfinite(gradientArray))
        and np.all(np.isfinite(hessianArray))
    ):
        raise FloatingPointError("scalar objective returned NaN or Inf")
    return (
        float(value),
        gradientArray.copy(),
        0.5 * (hessianArray + hessianArray.T),
    )


def _curvatureTolerance(hessian: Array) -> float:
    """
    Return a roundoff-scaled threshold for numerical negative curvature.
    """

    return 32.0 * np.finfo(float).eps * max(
        1.0,
        float(np.linalg.norm(hessian, ord=2)),
    )


def _stableNorm(values: Array) -> float:
    """
    Return a Euclidean norm without overflow in wide intermediate vectors.
    """

    array = np.asarray(values, dtype=float)
    maximum = float(np.max(np.abs(array), initial=0.0))
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _result(
    state: Array,
    value: float,
    gradient: Array,
    hessian: Array,
    stationarityNorm: float,
    lambdaMin: float,
    iterations: int,
    evaluations: int,
    converged: bool,
    termination: str,
) -> ScalarTrustRegionResult:
    """
    Build one immutable scalar result without deriving extra success rules.
    """

    return ScalarTrustRegionResult(
        state=state,
        value=value,
        gradient=gradient,
        hessian=hessian,
        stationarityNorm=stationarityNorm,
        lambdaMin=lambdaMin,
        iterations=iterations,
        evaluations=evaluations,
        converged=converged,
        termination=termination,
    )


__all__ = [
    "ScalarTrustRegionConfig",
    "ScalarTrustRegionResult",
    "minimizeScalar",
]
