"""
Keller (1977) pseudo-arclength continuation with shared thin-SVD correction.
The augmented equations use the same bounded least-squares trust-region solver
as the fixed-composition critical-point and spinodal correctors. Both physical
residual and arclength tolerances must pass before accepting a point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .LeastSquaresTrustRegion import solveLeastSquares


ResidualJacobian = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class ImplicitCurveTermination(str, Enum):
    """
    Explicit reasons for ending an implicit-curve trace.
    """

    MAX_STEPS = "maximum_steps_reached"
    BOUNDS_REACHED = "bounds_reached"
    MIN_STEP = "minimum_step_reached"
    RETRIES_EXHAUSTED = "retries_exhausted"
    RANK_LOSS = "jacobian_rank_loss"


class ImplicitCorrectorTermination(str, Enum):
    """
    Explicit outcomes of one implicit-curve correction attempt.
    """

    ACCEPTED = "accepted"
    PREDICTOR_OUTSIDE_BOUNDS = "predictor_outside_bounds"
    EVALUATION_FAILURE = "evaluation_failure"
    CORRECTION_TOO_LARGE = "correction_too_large"
    TRUST_REGION_FAILURE = "trust_region_failure"


@dataclass(frozen=True)
class ImplicitCurveConfig:
    """
    Controls for Keller pseudo-arclength prediction and correction.
    """

    initialStep: float = 0.02
    minStep: float = 2.0e-4
    maxStep: float = 0.05
    growthFactor: float = 1.3
    shrinkFactor: float = 0.5
    maxSteps: int = 100
    maxRetry: int = 7
    maxCorrectorEvaluations: int = 80
    stepTol: float = 1.0e-12
    resTol: float = 1.0e-8
    arcTol: float = 1.0e-10
    rankTol: float = 1.0e-10
    maxCorrectionFactor: float = 2.5
    fastIte: int = 3
    slowIte: int = 7
    growthTurn: float = 0.10
    shrinkTurn: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "initialStep",
            "minStep",
            "maxStep",
            "growthFactor",
            "shrinkFactor",
            "resTol",
            "stepTol",
            "arcTol",
            "rankTol",
            "maxCorrectionFactor",
            "growthTurn",
            "shrinkTurn",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.minStep <= self.initialStep <= self.maxStep:
            raise ValueError("require 0 < minStep <= initialStep <= maxStep")
        if self.growthFactor <= 1.0 or not 0.0 < self.shrinkFactor < 1.0:
            raise ValueError("invalid step growth or shrink factor")
        for name in ("maxSteps", "maxCorrectorEvaluations", "fastIte", "slowIte"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("maxRetry",):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.slowIte < self.fastIte:
            raise ValueError("slowIte must not be below fastIte")
        if min(
            self.resTol,
            self.stepTol,
            self.arcTol,
            self.rankTol,
            self.maxCorrectionFactor,
        ) <= 0.0:
            raise ValueError("implicit-curve tolerances must be positive")
        if not 0.0 < self.growthTurn < self.shrinkTurn < math.pi:
            raise ValueError("require 0 < growthTurn < shrinkTurn < pi")


@dataclass(frozen=True)
class ImplicitCurvePoint:
    """
    One accepted point and its local null-space tangent.
    """

    state: np.ndarray
    res: np.ndarray
    tangent: np.ndarray
    rNorm: float
    nullNorm: float


@dataclass(frozen=True)
class ImplicitCurveAttempt:
    """
    Diagnostics for one predictor-corrector attempt.
    """

    stepIndex: int
    retryIndex: int
    stepSize: float
    nIte: int
    rNorm: float
    rArc: float
    correctionNorm: float
    termination: ImplicitCorrectorTermination
    nEval: int = 0
    solverTermination: str = "not_started"

    @property
    def accepted(self) -> bool:
        """
        Return whether this correction attempt was accepted.
        """

        return self.termination is ImplicitCorrectorTermination.ACCEPTED


@dataclass(frozen=True)
class ImplicitCurveResult:
    """
    Accepted curve segment, attempts and explicit termination.
    """

    points: tuple[ImplicitCurvePoint, ...]
    attempts: tuple[ImplicitCurveAttempt, ...]
    termination: ImplicitCurveTermination

    @property
    def completed(self) -> bool:
        """
        Return whether tracing reached a normal finite endpoint.
        """

        return self.termination in {
            ImplicitCurveTermination.MAX_STEPS,
            ImplicitCurveTermination.BOUNDS_REACHED,
        }


@dataclass(frozen=True)
class _CorrectorResult:
    state: np.ndarray
    res: np.ndarray
    jacobian: np.ndarray
    nIte: int
    rNorm: float
    rArc: float
    correctionNorm: float
    termination: ImplicitCorrectorTermination
    nEval: int = 0
    solverTermination: str = "not_started"

    @property
    def accepted(self) -> bool:
        """
        Return whether both equation and continuation constraints were accepted.
        """

        return self.termination is ImplicitCorrectorTermination.ACCEPTED


class ImplicitCurveTracer:
    """
    Trace a regular codimension-one system by pseudo arclength.

    For an ``n``-coordinate state, ``evaluate`` must return ``n - 1``
    residuals and their ``(n - 1) x n`` Jacobian.  The original binary
    critical-locus use therefore remains the special case ``n = 3``.
    """

    def __init__(
        self,
        evaluate: ResidualJacobian,
        lowerBounds: Sequence[float],
        upperBounds: Sequence[float],
        config: ImplicitCurveConfig | None = None,
    ) -> None:
        self.evaluate = evaluate
        self.lower = np.asarray(lowerBounds, dtype=float)
        self.upper = np.asarray(upperBounds, dtype=float)
        self.config = config or ImplicitCurveConfig()
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or self.lower.size < 2
        ):
            raise ValueError(
                "implicit-curve bounds must be equal-length vectors "
                "with at least two coordinates"
            )
        self._nState = int(self.lower.size)
        self._nRes = self._nState - 1
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("implicit-curve bounds must be finite")
        if not np.all(self.lower < self.upper):
            raise ValueError("implicit-curve lower bounds must be below upper bounds")

    def trace(
        self,
        seed: Sequence[float],
        *,
        initialTangent: Sequence[float] | None = None,
        direction: int = 1,
    ) -> ImplicitCurveResult:
        """
        Trace one orientation from an already corrected seed.
        """

        if (
            not isinstance(direction, int)
            or isinstance(direction, bool)
            or direction not in (-1, 1)
        ):
            raise ValueError("direction must be -1 or 1")
        u = np.asarray(seed, dtype=float)
        if u.shape != self.lower.shape or not self._inside(u):
            raise ValueError(
                "seed must be a finite in-bounds state matching the bounds"
            )
        res, jacobian = self._checkedEvaluation(u)
        if np.linalg.norm(res) > self.config.resTol:
            raise ValueError("seed does not satisfy the implicit residual tolerance")
        align = None
        if initialTangent is not None:
            align = np.asarray(initialTangent, dtype=float)
            if align.shape != self.lower.shape or not np.all(np.isfinite(align)):
                raise ValueError(
                    "initialTangent must be finite and match the bounds"
                )
        tangent = self._tangent(jacobian, align)
        tangent *= direction
        points = [self._point(u, res, jacobian, tangent)]
        attempts: list[ImplicitCurveAttempt] = []
        ds = self.config.initialStep

        for stepIndex in range(self.config.maxSteps):
            for retryIndex in range(self.config.maxRetry + 1):
                predictor = u + ds * tangent
                if not self._inside(predictor):
                    outcome = None
                    termination = (
                        ImplicitCorrectorTermination.PREDICTOR_OUTSIDE_BOUNDS
                    )
                else:
                    outcome = self._correct(predictor, tangent, ds)
                    termination = outcome.termination
                if outcome is not None:
                    attempts.append(
                        ImplicitCurveAttempt(
                            stepIndex=stepIndex,
                            retryIndex=retryIndex,
                            stepSize=ds,
                            nIte=outcome.nIte,
                            rNorm=outcome.rNorm,
                            rArc=outcome.rArc,
                            correctionNorm=outcome.correctionNorm,
                            termination=outcome.termination,
                            nEval=outcome.nEval,
                            solverTermination=outcome.solverTermination,
                        )
                    )
                else:
                    attempts.append(
                        ImplicitCurveAttempt(
                            stepIndex=stepIndex,
                            retryIndex=retryIndex,
                            stepSize=ds,
                            nIte=0,
                            rNorm=math.inf,
                            rArc=math.inf,
                            correctionNorm=math.inf,
                            termination=termination,
                        )
                    )
                if outcome is not None and outcome.accepted:
                    try:
                        nextTangent = self._tangent(outcome.jacobian, tangent)
                    except np.linalg.LinAlgError:
                        return ImplicitCurveResult(
                            points=tuple(points),
                            attempts=tuple(attempts),
                            termination=ImplicitCurveTermination.RANK_LOSS,
                        )
                    turn = math.acos(
                        float(np.clip(np.dot(tangent, nextTangent), -1.0, 1.0))
                    )
                    u = outcome.state
                    tangent = nextTangent
                    points.append(
                        self._point(
                            u,
                            outcome.res,
                            outcome.jacobian,
                            tangent,
                        )
                    )
                    if (
                        outcome.nIte <= self.config.fastIte
                        and turn <= self.config.growthTurn
                    ):
                        ds = min(self.config.maxStep, ds * self.config.growthFactor)
                    elif (
                        outcome.nIte >= self.config.slowIte
                        or turn >= self.config.shrinkTurn
                    ):
                        ds = max(self.config.minStep, ds * self.config.shrinkFactor)
                    break
                nextDs = ds * self.config.shrinkFactor
                if nextDs < self.config.minStep:
                    atBounds = (
                        termination
                        is ImplicitCorrectorTermination.PREDICTOR_OUTSIDE_BOUNDS
                    )
                    termination = (
                        ImplicitCurveTermination.BOUNDS_REACHED
                        if atBounds
                        else ImplicitCurveTermination.MIN_STEP
                    )
                    return ImplicitCurveResult(
                        points=tuple(points),
                        attempts=tuple(attempts),
                        termination=termination,
                    )
                ds = nextDs
            else:
                return ImplicitCurveResult(
                    points=tuple(points),
                    attempts=tuple(attempts),
                    termination=ImplicitCurveTermination.RETRIES_EXHAUSTED,
                )
        return ImplicitCurveResult(
            points=tuple(points),
            attempts=tuple(attempts),
            termination=ImplicitCurveTermination.MAX_STEPS,
        )

    def _correct(
        self,
        predictor: np.ndarray,
        tangent: np.ndarray,
        ds: float,
    ) -> _CorrectorResult:
        """
        Solve the augmented equations with the shared bounded corrector.

        Scaling the arc row by resTol/arcTol makes the solver's norm criterion
        sufficient for both requested tolerances. The correction distance is
        restricted relative to the predictor to reduce branch switching.
        """

        arcScale = self.config.resTol / self.config.arcTol
        maxCorrection = self.config.maxCorrectionFactor * ds
        predictorEvaluation: tuple[np.ndarray, np.ndarray] | None = None

        def evaluate(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """
            Return the local residual and Jacobian for the shared numerical solver.
            """

            if not self._inside(u):
                raise FloatingPointError("implicit state is outside open bounds")
            if np.linalg.norm(u - predictor) > maxCorrection:
                raise FloatingPointError("implicit correction exceeds local radius")
            if predictorEvaluation is not None and np.array_equal(u, predictor):
                return predictorEvaluation
            res, jacobian = self._checkedEvaluation(u)
            rArc = float(np.dot(tangent, u - predictor))
            return (
                np.concatenate((res, [arcScale * rArc])),
                np.vstack((jacobian, arcScale * tangent)),
            )

        try:
            predictorEvaluation = evaluate(predictor)

            # The solver measures steps after Jacobian column scaling. The
            # largest column norm maps the original-coordinate ball into an
            # enclosing scaled ball; evaluate still enforces its true radius.
            radiusScale = max(1.0e-14, float(np.max(np.linalg.norm(
                predictorEvaluation[1], axis=0,
            ))))
            result = solveLeastSquares(
                evaluate, predictor, self.lower, self.upper,
                resTol=self.config.resTol,
                stepTol=self.config.stepTol,
                maxEval=self.config.maxCorrectorEvaluations,
                initialRadius=min(ds, maxCorrection) * radiusScale,
                maxRadius=maxCorrection * radiusScale,
                acceptanceRatio=0.1,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return self._failed(
                predictor.copy(), np.full(self._nRes, math.inf),
                np.full((self._nRes, self._nState), math.nan),
                0, math.inf, 0.0,
                ImplicitCorrectorTermination.EVALUATION_FAILURE,
            )
        res = result.residual[:-1]
        jacobian = result.jacobian[:-1]
        rArc = float(np.dot(tangent, result.state - predictor))
        correctionNorm = float(np.linalg.norm(result.state - predictor))
        accepted = (
            result.converged
            and self._inside(result.state)
            and np.linalg.norm(res) <= self.config.resTol
            and abs(rArc) <= self.config.arcTol
            and correctionNorm <= maxCorrection
        )
        termination = (
            ImplicitCorrectorTermination.ACCEPTED if accepted
            else ImplicitCorrectorTermination.TRUST_REGION_FAILURE
        )
        return _CorrectorResult(
            state=result.state, res=res, jacobian=jacobian,
            nIte=result.iterations, rNorm=float(np.linalg.norm(res)),
            rArc=rArc, correctionNorm=correctionNorm, termination=termination,
            nEval=result.evaluations, solverTermination=result.termination,
        )

    def _checkedEvaluation(
        self,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        res, jacobian = self.evaluate(u)
        res = np.asarray(res, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)
        resShape = (self._nRes,)
        jacobianShape = (self._nRes, self._nState)
        if res.shape != resShape or jacobian.shape != jacobianShape:
            raise ValueError(
                "evaluate must return residual and Jacobian shapes "
                f"{resShape} and {jacobianShape}"
            )
        if not np.all(np.isfinite(res)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("implicit residual or Jacobian is non-finite")
        sigma = np.linalg.svd(jacobian, compute_uv=False)
        if sigma[-1] <= self.config.rankTol * max(1.0, sigma[0]):
            raise np.linalg.LinAlgError("implicit Jacobian lost row rank")
        return res, jacobian

    def _tangent(
        self,
        jacobian: np.ndarray,
        align: np.ndarray | None,
    ) -> np.ndarray:
        _left, sigma, right = np.linalg.svd(jacobian, full_matrices=True)
        if sigma[-1] <= self.config.rankTol * max(1.0, sigma[0]):
            raise np.linalg.LinAlgError("implicit Jacobian lost row rank")
        tangent = right[-1]
        tangent /= np.linalg.norm(tangent)
        if align is not None and np.dot(tangent, align) < 0.0:
            tangent = -tangent
        return tangent

    def _point(
        self,
        u: np.ndarray,
        res: np.ndarray,
        jacobian: np.ndarray,
        tangent: np.ndarray,
    ) -> ImplicitCurvePoint:
        return ImplicitCurvePoint(
            state=u.copy(),
            res=res.copy(),
            tangent=tangent.copy(),
            rNorm=float(np.linalg.norm(res)),
            nullNorm=float(np.linalg.norm(jacobian @ tangent)),
        )

    def _failed(
        self,
        u: np.ndarray,
        res: np.ndarray,
        jacobian: np.ndarray,
        nIte: int,
        rArc: float,
        correctionNorm: float,
        termination: ImplicitCorrectorTermination,
    ) -> _CorrectorResult:
        return _CorrectorResult(
            state=u,
            res=res,
            jacobian=jacobian,
            nIte=nIte,
            rNorm=float(np.linalg.norm(res)),
            rArc=rArc,
            correctionNorm=correctionNorm,
            termination=termination,
        )

    def _inside(self, u: np.ndarray) -> bool:
        return bool(
            u.shape == self.lower.shape
            and np.all(np.isfinite(u))
            and np.all(u > self.lower)
            and np.all(u < self.upper)
        )


__all__ = [
    "ImplicitCorrectorTermination",
    "ImplicitCurveAttempt",
    "ImplicitCurveConfig",
    "ImplicitCurvePoint",
    "ImplicitCurveResult",
    "ImplicitCurveTermination",
    "ImplicitCurveTracer",
]
