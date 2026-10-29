"""
Trace fixed-composition spinodals and refine critical-point candidates.
Use a single automatic start or starts on the edges of the search region.
Returned temperatures and densities use K and mol/L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ..Numerics.BracketRoot import solveBracket

from ..Common.Composition import (
    compositionDensityScale,
    compositionTemperatureScale,
    normalizeComposition,
)
from ..Kernel.Critical import (
    DensityStabilityEvaluation,
    DensityStabilityState,
)
from ..Numerics.PseudoArclength import (
    correctPseudoArclength,
    nullSpaceTangent,
)
from .PointSelection import deduplicatePoints
from .CriticalPoint import (
    CriticalPointResult,
    CriticalSolveAttempt,
    CriticalPointSolver,
)


@dataclass(frozen=True)
class CriticalDiscoveryConfig:
    """
    Controls for fixed-composition spinodal tracing and critical-point searches.
    Step lengths, radii and tolerances use dimensionless log(T/K), log(rho/(mol/L))
    coordinates and residuals. Scale factors, ratios and counts are dimensionless.
    """

    seedDensityScaleFactor: float = 0.50
    seedTemperatureScaleFactor: float = 1.50
    seedTemperatureDecreaseFactor: float = 0.90
    maximumSeedEvaluations: int = 60
    initialStep: float = 0.03
    minimumStep: float = 0.0025
    maximumStep: float = 0.04
    growthFactor: float = 1.25
    shrinkFactor: float = 0.5
    fastEval: int = 8
    maximumSteps: int = 500
    maximumAttempts: int = 1000
    maximumCorrectorEvaluations: int = 80
    eventCStarTolerance: float = 5.0e-3
    resTol: float = 1.0e-8
    stepTol: float = 1.0e-10
    initialRadius: float = 0.20
    maxRadius: float = 1.0
    acceptanceRatio: float = 0.1
    boundarySamplesPerEdge: int = 17
    boundaryInsetFraction: float = 1.0e-3

    def __post_init__(self) -> None:
        finiteControls = (
            self.seedDensityScaleFactor,
            self.seedTemperatureScaleFactor,
            self.seedTemperatureDecreaseFactor,
            self.initialStep,
            self.minimumStep,
            self.maximumStep,
            self.growthFactor,
            self.shrinkFactor,
            self.eventCStarTolerance,
            self.resTol,
            self.stepTol,
            self.initialRadius,
            self.maxRadius,
            self.acceptanceRatio,
            self.boundaryInsetFraction,
        )
        if not all(math.isfinite(value) for value in finiteControls):
            raise ValueError("critical discovery controls must be finite")
        if not 0.0 < self.seedDensityScaleFactor < 1.0:
            raise ValueError("seedDensityScaleFactor must lie between zero and one")
        if self.seedTemperatureScaleFactor <= 1.0:
            raise ValueError("seedTemperatureScaleFactor must exceed one")
        if not 0.0 < self.seedTemperatureDecreaseFactor < 1.0:
            raise ValueError(
                "seedTemperatureDecreaseFactor must lie between zero and one"
            )
        if self.maximumSeedEvaluations < 2:
            raise ValueError("maximumSeedEvaluations must be at least two")
        if not 0.0 < self.minimumStep <= self.initialStep <= self.maximumStep:
            raise ValueError("contour steps must satisfy 0 < minimum <= initial <= maximum")
        if self.growthFactor <= 1.0 or not 0.0 < self.shrinkFactor < 1.0:
            raise ValueError("contour growth and shrink factors are invalid")
        if min(
            self.fastEval,
            self.maximumSteps,
            self.maximumAttempts,
            self.maximumCorrectorEvaluations,
        ) < 1:
            raise ValueError("contour iteration controls must be positive")
        if min(self.eventCStarTolerance, self.resTol, self.stepTol) <= 0.0:
            raise ValueError("contour and event tolerances must be positive")
        if not 0.0 < self.initialRadius <= self.maxRadius:
            raise ValueError("contour radii must satisfy 0 < initial <= maximum")
        if not 0.0 <= self.acceptanceRatio < 1.0:
            raise ValueError("acceptanceRatio must lie in [0, 1)")
        if (
            not isinstance(self.boundarySamplesPerEdge, int)
            or isinstance(self.boundarySamplesPerEdge, bool)
            or self.boundarySamplesPerEdge < 3
        ):
            raise ValueError(
                "boundarySamplesPerEdge must be an integer of at least three"
            )
        if not 0.0 < self.boundaryInsetFraction < 0.5:
            raise ValueError("boundaryInsetFraction must lie between zero and one half")


@dataclass(frozen=True)
class CriticalContourPoint:
    """
    One accepted spinodal point: T [K], rho [mol/L], qMin and cStar [-].
    Arc length is dimensionless in logarithmic temperature/density coordinates.
    """

    T: float
    rho: float
    qMin: float
    cStar: float
    arcLength: float
    correctorEvaluations: int


@dataclass(frozen=True)
class CriticalDiscoveryResult:
    """
    Critical roots and bounded-coverage diagnostics from one spinodal trace.
    """

    roots: tuple[CriticalPointResult, ...]
    attempts: tuple[CriticalSolveAttempt, ...]
    contour: tuple[CriticalContourPoint, ...]
    completed: bool
    terminationReason: str
    seedEvaluations: int
    attemptedSteps: int
    rejectedSteps: int
    coverageKind: str
    rootSetCertified: bool


@dataclass(frozen=True)
class BoundaryCriticalDiscoveryResult:
    """
    Roots and diagnostics from spinodal traces started at search-region edges.
    """

    roots: tuple[CriticalPointResult, ...]
    contours: tuple[CriticalDiscoveryResult, ...]
    completed: bool
    boundaryEvaluations: int
    boundaryFailures: tuple[str, ...]
    coverageKind: str
    rootSetCertified: bool


class _SeedEvaluationLimit(RuntimeError):
    """
    Internal signal that the one-dimensional seed budget is exhausted.
    """


class _AcceptedStabilityStateMismatch(RuntimeError):
    """
    Signal an internal mismatch between corrector and retained evidence.
    """


class BellCriticalFinder:
    """
    Discover ordinary CP roots on fixed-composition stability contours.
    """

    def __init__(
        self,
        solver: CriticalPointSolver,
        config: CriticalDiscoveryConfig | None = None,
    ) -> None:
        self.solver = solver
        self.config = config or CriticalDiscoveryConfig()

    def find(
        self,
        composition: np.ndarray | list[float] | tuple[float, ...],
    ) -> CriticalDiscoveryResult:
        """
        Trace $q_{min}=0$ and refine crossing or near-tangent $c^*=0$ events.
        """

        x = normalizeComposition(composition, self.solver.model.componentCount)
        tempBounds, densityBounds = self.solver.stateBounds(x)
        lower = np.log([tempBounds[0], densityBounds[0]])
        upper = np.log([tempBounds[1], densityBounds[1]])
        try:
            seed, seedEvaluations, seedReason = self._seed(
                x,
                tempBounds,
                densityBounds,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return CriticalDiscoveryResult(
                roots=(),
                attempts=(),
                contour=(),
                completed=False,
                terminationReason=f"seed_evaluation_failed: {type(error).__name__}: {error}",
                seedEvaluations=0,
                attemptedSteps=0,
                rejectedSteps=0,
                coverageKind="none",
                rootSetCertified=False,
            )
        if seed is None:
            return CriticalDiscoveryResult(
                roots=(),
                attempts=(),
                contour=(),
                completed=False,
                terminationReason=seedReason,
                seedEvaluations=seedEvaluations,
                attemptedSteps=0,
                rejectedSteps=0,
                coverageKind="none",
                rootSetCertified=False,
            )

        return self._traceSeed(
            x,
            seed,
            seedEvaluations,
            orientationAxis=1,
            orientationSign=1,
            coverageKind="outer_spinodal",
        )

    def findBoundaryContours(
        self,
        composition: np.ndarray | list[float] | tuple[float, ...],
    ) -> BoundaryCriticalDiscoveryResult:
        """
        Trace sign-changing spinodal components found on the state-box edges.

        The finite boundary scan covers only transverse ``qMin=0`` intersections
        resolved by the configured edge samples. Closed interior contours and
        tangential boundary contacts remain outside the guarantee.
        """

        x = normalizeComposition(composition, self.solver.model.componentCount)
        tempBounds, densityBounds = self.solver.stateBounds(x)
        lower = np.log([tempBounds[0], densityBounds[0]])
        upper = np.log([tempBounds[1], densityBounds[1]])
        seeds, boundaryEvaluations, boundaryFailures = self._boundarySeeds(
            x,
            lower,
            upper,
        )
        contours: list[CriticalDiscoveryResult] = []
        for seed, orientationAxis, orientationSign in seeds:
            if self._contourSeedCovered(seed, contours):
                continue
            contours.append(
                self._traceSeed(
                    x,
                    seed,
                    0,
                    orientationAxis=orientationAxis,
                    orientationSign=orientationSign,
                    coverageKind="boundary_sign_change_spinodal",
                )
            )
        roots = tuple(
            deduplicatePoints(
                (
                    point
                    for contour in contours
                    for point in contour.roots
                ),
                self.solver.config,
            )
        )
        completed = not boundaryFailures and all(
            contour.completed for contour in contours
        )
        return BoundaryCriticalDiscoveryResult(
            roots=roots,
            contours=tuple(contours),
            completed=completed,
            boundaryEvaluations=boundaryEvaluations,
            boundaryFailures=boundaryFailures,
            coverageKind="boundary_sign_change_spinodals",
            rootSetCertified=False,
        )

    def _traceSeed(
        self,
        x: np.ndarray,
        seed: np.ndarray,
        seedEvaluations: int,
        *,
        orientationAxis: int,
        orientationSign: int,
        coverageKind: str,
    ) -> CriticalDiscoveryResult:
        """
        Trace one spinodal component from a corrected interior seed.
        """

        tempBounds, densityBounds = self.solver.stateBounds(x)
        lower = np.log([tempBounds[0], densityBounds[0]])
        upper = np.log([tempBounds[1], densityBounds[1]])

        try:
            seedEvaluation = self.solver.kernel.stabilityEvaluation(seed, x)
            _residual, jacobian = self._stabilityArrays(seedEvaluation)
            tangent = nullSpaceTangent(jacobian)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return CriticalDiscoveryResult(
                roots=(),
                attempts=(),
                contour=(),
                completed=False,
                terminationReason=f"seed_tangent_failed: {type(error).__name__}: {error}",
                seedEvaluations=seedEvaluations,
                attemptedSteps=0,
                rejectedSteps=0,
                coverageKind="none",
                rootSetCertified=False,
            )
        if tangent[orientationAxis] * orientationSign < 0.0:
            tangent *= -1.0

        try:
            first, criticalDirection = self._contourPoint(
                seed,
                x,
                0.0,
                0,
                stabilityState=seedEvaluation.stability,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return CriticalDiscoveryResult(
                roots=(),
                attempts=(),
                contour=(),
                completed=False,
                terminationReason=f"seed_event_failed: {type(error).__name__}: {error}",
                seedEvaluations=seedEvaluations,
                attemptedSteps=0,
                rejectedSteps=0,
                coverageKind="none",
                rootSetCertified=False,
            )
        contour = [first]
        roots: list[CriticalPointResult] = []
        attempts: list[CriticalSolveAttempt] = []
        state = seed
        previousState: np.ndarray | None = None
        stepSize = self.config.initialStep
        arcLength = 0.0
        attempted = 0
        rejected = 0
        terminationReason = "maximum_steps_reached"
        completed = False

        while len(contour) - 1 < self.config.maximumSteps:
            if attempted >= self.config.maximumAttempts:
                terminationReason = "maximum_attempts_reached"
                break
            predictor = state + stepSize * tangent
            if np.any(predictor <= lower) or np.any(predictor >= upper):
                terminationReason = "state_bounds_reached"
                completed = True
                break
            attempted += 1
            latestEvaluation: DensityStabilityEvaluation | None = None

            def evaluateAndRetain(
                value: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                """
                Evaluate a spinodal state and retain its latest derivatives.
                """

                nonlocal latestEvaluation
                latestEvaluation = self.solver.kernel.stabilityEvaluation(
                    value,
                    x,
                )
                return self._stabilityArrays(latestEvaluation)

            try:
                correction = correctPseudoArclength(
                    evaluateAndRetain,
                    predictor,
                    tangent,
                    lower,
                    upper,
                    resTol=self.config.resTol,
                    stepTol=self.config.stepTol,
                    maxEval=self.config.maximumCorrectorEvaluations,
                    initialRadius=self.config.initialRadius,
                    maxRadius=self.config.maxRadius,
                    acceptanceRatio=self.config.acceptanceRatio,
                )
                if not correction.converged:
                    raise RuntimeError(correction.termination)
                candidateState = correction.state
                if latestEvaluation is None or not np.array_equal(
                    latestEvaluation.logState.cpu().numpy(),
                    candidateState,
                ):
                    raise _AcceptedStabilityStateMismatch(
                        "corrector result does not match its latest stability evaluation"
                    )

                # The pseudo-arclength result retains the accepted augmented
                # Jacobian. Its first row is the stability-contour Jacobian;
                # reuse it instead of rebuilding the same AD graph.
                candidateJacobian = correction.jacobian[:-1]
                candidateTangent = nullSpaceTangent(candidateJacobian, tangent)
                increment = float(np.linalg.norm(candidateState - state))
                if not math.isfinite(increment) or increment <= 0.0:
                    raise ValueError("contour increment must be finite and positive")
                candidate, candidateCriticalDirection = self._contourPoint(
                    candidateState,
                    x,
                    arcLength + increment,
                    correction.evaluations,
                    referenceDirection=criticalDirection,
                    stabilityState=latestEvaluation.stability,
                )
            except _AcceptedStabilityStateMismatch:
                raise
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
                rejected += 1
                stepSize *= self.config.shrinkFactor
                if stepSize < self.config.minimumStep:
                    terminationReason = "minimum_step_reached"
                    break
                continue

            previous = contour[-1]
            if previous.cStar * candidate.cStar <= 0.0 and (
                previous.cStar != candidate.cStar
            ):
                attempt = self._refineEvent(
                    x,
                    state,
                    candidateState,
                    previous.cStar,
                    candidate.cStar,
                    len(attempts),
                )
                attempts.append(attempt)
                if attempt.accepted and attempt.point is not None:
                    roots.append(attempt.point)

            contour.append(candidate)
            if len(contour) >= 3:
                if previousState is None:
                    raise RuntimeError("Bell contour state history is incomplete")
                eventStates = self._quadraticEventStates(
                    (contour[-3], contour[-2], contour[-1]),
                    (previousState, state, candidateState),
                )
                for eventState in eventStates:
                    attempt = self._refineEvent(
                        x,
                        eventState,
                        eventState,
                        0.0,
                        0.0,
                        len(attempts),
                        eventKind="tangent",
                    )
                    attempts.append(attempt)
                    if attempt.accepted and attempt.point is not None:
                        roots.append(attempt.point)

            previousState = state.copy()
            state = candidateState.copy()
            tangent = candidateTangent
            criticalDirection = candidateCriticalDirection
            arcLength = candidate.arcLength
            stepSize = self._acceptedStepSize(
                stepSize,
                correction.evaluations,
            )

        unique = tuple(deduplicatePoints(roots, self.solver.config))
        return CriticalDiscoveryResult(
            roots=unique,
            attempts=tuple(attempts),
            contour=tuple(contour),
            completed=completed,
            terminationReason=terminationReason,
            seedEvaluations=seedEvaluations,
            attemptedSteps=attempted,
            rejectedSteps=rejected,
            coverageKind=coverageKind,
            rootSetCertified=False,
        )

    def _acceptedStepSize(
        self,
        currentStep: float,
        correctorEvaluations: int,
    ) -> float:
        """
        Return the next step after one accepted contour correction.

        The base policy preserves the original behavior: grow only after a
        fast correction and otherwise retain the current step.  Specialized
        finders may override this internal hook without copying the contour
        continuation and event-refinement implementation.
        """

        if correctorEvaluations <= self.config.fastEval:
            return min(
                currentStep * self.config.growthFactor,
                self.config.maximumStep,
            )
        return currentStep

    def _boundarySeeds(
        self,
        composition: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[tuple[tuple[np.ndarray, int, int], ...], int, tuple[str, ...]]:
        """
        Bracket transverse stability-contour intersections on four inset edges.
        """

        span = upper - lower
        insetLower = lower + self.config.boundaryInsetFraction * span
        insetUpper = upper - self.config.boundaryInsetFraction * span
        edges = (
            ("lower_temperature", 0, insetLower[0], 1, 1),
            ("upper_temperature", 0, insetUpper[0], 1, -1),
            ("lower_density", 1, insetLower[1], 0, 1),
            ("upper_density", 1, insetUpper[1], 0, -1),
        )
        evaluations = 0
        failures: list[str] = []
        cache: dict[tuple[float, float], float] = {}
        seeds: list[tuple[np.ndarray, int, int]] = []

        def qAt(state: np.ndarray) -> float:
            """
            Evaluate the dimensionless stability residual, counting callback evaluations.
            """

            nonlocal evaluations
            key = (float(state[0]), float(state[1]))
            if key not in cache:
                evaluations += 1
                value = self.solver.kernel.stabilityResidual(
                    state,
                    composition,
                )
                cache[key] = float(value.detach().cpu().item())
                if not math.isfinite(cache[key]):
                    raise FloatingPointError(
                        "boundary stability residual is nonfinite"
                    )
            return cache[key]

        def addSeed(state: np.ndarray, axis: int, sign: int) -> None:
            """
            Retain a deduplicated boundary seed and its inward orientation.
            """

            if any(
                np.linalg.norm(state - existing[0])
                <= max(self.config.stepTol, 1.0e-9)
                for existing in seeds
            ):
                return
            seeds.append((np.asarray(state, dtype=float), axis, sign))

        for edgeName, fixedAxis, fixedValue, variableAxis, inwardSign in edges:
            coordinates = np.linspace(
                insetLower[variableAxis],
                insetUpper[variableAxis],
                self.config.boundarySamplesPerEdge,
            )

            def stateAt(coordinate: float) -> np.ndarray:
                """
                Map an edge coordinate to the two logarithmic state coordinates.
                """

                state = np.empty(2, dtype=float)
                state[fixedAxis] = fixedValue
                state[variableAxis] = coordinate
                return state

            values: list[float] = []
            for coordinate in coordinates:
                try:
                    value = qAt(stateAt(float(coordinate)))
                except (
                    FloatingPointError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    value = math.nan
                    failures.append(
                        f"{edgeName}_evaluation_failed: "
                        f"{type(error).__name__}: {error}"
                    )
                values.append(value)

            for index, (left, right) in enumerate(
                zip(coordinates[:-1], coordinates[1:])
            ):
                leftValue = values[index]
                rightValue = values[index + 1]
                if not (math.isfinite(leftValue) and math.isfinite(rightValue)):
                    continue
                if abs(leftValue) <= self.config.resTol:
                    addSeed(stateAt(float(left)), fixedAxis, inwardSign)
                if abs(rightValue) <= self.config.resTol:
                    addSeed(stateAt(float(right)), fixedAxis, inwardSign)
                if leftValue * rightValue >= 0.0:
                    continue
                try:
                    root = solveBracket(
                        lambda coordinate: qAt(stateAt(float(coordinate))),
                        float(left),
                        float(right),
                        xtol=max(self.config.stepTol, 1.0e-12),
                        maxiter=self.config.maximumSeedEvaluations,
                    )
                except (
                    FloatingPointError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    failures.append(
                        f"{edgeName}_bracket_failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                addSeed(stateAt(root), fixedAxis, inwardSign)

        seeds.sort(key=lambda item: (item[1], item[2], item[0][0], item[0][1]))
        return tuple(seeds), evaluations, tuple(failures)

    def _contourSeedCovered(
        self,
        seed: np.ndarray,
        contours: list[CriticalDiscoveryResult],
    ) -> bool:
        """
        Return whether a boundary seed already lies on a traced contour.
        """

        tolerance = 1.25 * self.config.maximumStep
        return any(
            min(
                np.linalg.norm(
                    seed - np.log([point.T, point.rho])
                )
                for point in result.contour
            )
            <= tolerance
            for result in contours
            if result.contour
        )

    def _quadraticEventStates(
        self,
        points: tuple[CriticalContourPoint, CriticalContourPoint, CriticalContourPoint],
        states: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        """
        Return quadratic near-tangent event starts from three contour points.

        Sign-changing intervals are handled separately. For three same-sign
        samples, a local minimum in ``abs(cStar)`` is fitted as a quadratic in
        contour arc length. Real interpolated zeros recover an even number of
        crossings hidden between samples; a near-zero vertex supplies one
        candidate for a possible tangent event. These are only starts for the
        full two-equation critical corrector, never accepted roots by themselves.
        """

        cValues = np.asarray([point.cStar for point in points], dtype=float)
        if not np.all(np.isfinite(cValues)):
            return ()
        if cValues[0] * cValues[1] <= 0.0 or cValues[1] * cValues[2] <= 0.0:
            return ()
        if not (
            abs(cValues[1]) < abs(cValues[0])
            and abs(cValues[1]) <= abs(cValues[2])
        ):
            return ()

        arcValues = np.asarray([point.arcLength for point in points], dtype=float)
        span = arcValues[2] - arcValues[0]
        if not np.all(np.isfinite(arcValues)) or span <= 0.0:
            return ()
        scaledArc = (arcValues - arcValues[1]) / span
        curvature, slope, offset = np.polyfit(scaledArc, cValues, 2)
        coefficientScale = max(abs(curvature), abs(slope), abs(offset), 1.0)
        if abs(curvature) <= np.finfo(float).eps * coefficientScale:
            return ()
        vertex = -slope / (2.0 * curvature)
        if not scaledArc[0] < vertex < scaledArc[2]:
            return ()
        vertexC = float(curvature * vertex**2 + slope * vertex + offset)
        if (
            abs(vertexC) > self.config.eventCStarTolerance
            and vertexC * cValues[1] > 0.0
        ):
            return ()

        discriminant = slope**2 - 4.0 * curvature * offset
        eventCoordinates: list[float]
        discriminantTol = np.finfo(float).eps * coefficientScale**2 * 32.0
        if discriminant >= -discriminantTol:
            rootOffset = math.sqrt(max(discriminant, 0.0))
            eventCoordinates = [
                (-slope - rootOffset) / (2.0 * curvature),
                (-slope + rootOffset) / (2.0 * curvature),
            ]
        else:
            eventCoordinates = [vertex]

        eventStates: list[np.ndarray] = []
        for coordinate in sorted(eventCoordinates):
            if not scaledArc[0] < coordinate < scaledArc[2]:
                continue
            targetArc = arcValues[1] + coordinate * span
            if targetArc <= arcValues[1]:
                leftIndex, rightIndex = 0, 1
            else:
                leftIndex, rightIndex = 1, 2
            localSpan = arcValues[rightIndex] - arcValues[leftIndex]
            fraction = (targetArc - arcValues[leftIndex]) / localSpan
            eventState = states[leftIndex] + fraction * (
                states[rightIndex] - states[leftIndex]
            )
            if (
                not eventStates
                or np.linalg.norm(eventState - eventStates[-1]) > 1.0e-10
            ):
                eventStates.append(np.asarray(eventState, dtype=float))
        return tuple(eventStates)

    def _seed(
        self,
        composition: np.ndarray,
        tempBounds: tuple[float, float],
        densityBounds: tuple[float, float],
    ) -> tuple[np.ndarray | None, int, str]:
        """
        Bracket the highest-temperature spinodal point on a low-density cut.
        """

        lowerT, upperT = tempBounds
        lowerRho = densityBounds[0] * (1.0 + 1.0e-10)
        upperRho = densityBounds[1] * (1.0 - 1.0e-10)
        densityScale = compositionDensityScale(
            self.solver.model,
            composition,
        )
        rho = self.config.seedDensityScaleFactor * densityScale
        if not lowerRho < rho < upperRho:
            return None, 0, "seed_density_out_of_bounds"

        lowerSearchT = lowerT * (1.0 + 1.0e-10)
        upperSearchT = upperT * (1.0 - 1.0e-10)
        temperatureScale = compositionTemperatureScale(
            self.solver.model,
            composition,
        )
        highT = float(
            np.clip(
                self.config.seedTemperatureScaleFactor * temperatureScale,
                lowerSearchT,
                upperSearchT,
            )
        )
        evaluations = 0

        def qAt(T: float) -> float:
            """
            Evaluate the dimensionless stability residual, counting callback evaluations.
            """

            nonlocal evaluations
            if evaluations >= self.config.maximumSeedEvaluations:
                raise _SeedEvaluationLimit
            evaluations += 1
            value = self.solver.kernel.stabilityResidual(
                np.log([T, rho]),
                composition,
            )
            return float(value.detach().cpu().item())

        qHigh = qAt(highT)
        if not math.isfinite(qHigh):
            return None, evaluations, "seed_residual_nonfinite"
        if abs(qHigh) <= self.config.resTol:
            return np.log([highT, rho]), evaluations, "seed_found"

        while qHigh < 0.0 and evaluations < self.config.maximumSeedEvaluations:
            candidate = min(
                upperSearchT,
                highT / self.config.seedTemperatureDecreaseFactor,
            )
            if candidate <= highT:
                break
            highT = candidate
            qHigh = qAt(highT)
            if not math.isfinite(qHigh):
                return None, evaluations, "seed_residual_nonfinite"
        if qHigh < 0.0:
            if (
                evaluations >= self.config.maximumSeedEvaluations
                and highT < upperSearchT
            ):
                return None, evaluations, "seed_evaluation_limit"
            return None, evaluations, "seed_bracket_not_found"

        upperBracket, qUpper = highT, qHigh
        lowerBracket, qLower = highT, qHigh
        while evaluations < self.config.maximumSeedEvaluations:
            candidate = max(
                lowerSearchT,
                lowerBracket * self.config.seedTemperatureDecreaseFactor,
            )
            if candidate >= lowerBracket:
                break
            lowerBracket = candidate
            qLower = qAt(lowerBracket)
            if not math.isfinite(qLower):
                return None, evaluations, "seed_residual_nonfinite"
            if qLower <= 0.0:
                break

        if qLower > 0.0:
            if (
                evaluations >= self.config.maximumSeedEvaluations
                and lowerBracket > lowerSearchT
            ):
                return None, evaluations, "seed_evaluation_limit"
            return None, evaluations, "seed_bracket_not_found"
        if self.config.maximumSeedEvaluations - evaluations < 2:
            return None, evaluations, "seed_evaluation_limit"
        try:
            root = solveBracket(
                qAt,
                lowerBracket,
                upperBracket,
                xtol=max(self.config.stepTol, 1.0e-12),
                maxiter=self.config.maximumSeedEvaluations - evaluations,
            )
        except _SeedEvaluationLimit:
            return None, evaluations, "seed_evaluation_limit"
        except (RuntimeError, ValueError):
            return None, evaluations, "seed_bracket_failed"
        return np.log([root, rho]), evaluations, "seed_found"

    def _stabilityEvaluation(
        self,
        logState: np.ndarray,
        composition: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        evaluation = self.solver.kernel.stabilityEvaluation(
            logState,
            composition,
        )
        return self._stabilityArrays(evaluation)

    @staticmethod
    def _stabilityArrays(
        evaluation: DensityStabilityEvaluation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the residual and Jacobian from the retained kernel state.
        """

        return (
            evaluation.residual.cpu().numpy(),
            evaluation.jacobian.cpu().numpy(),
        )

    def _contourPoint(
        self,
        logState: np.ndarray,
        composition: np.ndarray,
        arcLength: float,
        evaluations: int,
        *,
        referenceDirection: np.ndarray | None = None,
        stabilityState: DensityStabilityState | None = None,
    ) -> tuple[CriticalContourPoint, np.ndarray]:
        """
        Return one contour point and its continuously oriented CP direction.

        The eigensolver may reverse an eigenvector when the canonical pivot
        component changes. Because the third directional derivative is odd in
        that vector, an uncorrected reversal creates a false ``cStar`` sign
        event. Aligning successive simple-eigenvalue directions removes that
        representation discontinuity without changing the physical zero set.
        """

        T, rho = np.exp(logState)
        state = (
            self.solver.kernel.evaluate(T, rho, composition)
            if stabilityState is None
            else self.solver.kernel.criticalStateFromStability(stabilityState)
        )
        criticalDirection = (
            state.minEigenvector.detach().cpu().numpy().copy()
        )
        cStar = float(state.cStar.detach().cpu().item())
        if referenceDirection is not None:
            overlap = float(criticalDirection @ referenceDirection)
            if not math.isfinite(overlap):
                raise FloatingPointError(
                    "critical eigenvector overlap must be finite"
                )
            if overlap < 0.0:
                criticalDirection *= -1.0
                cStar *= -1.0
        point = CriticalContourPoint(
            T=float(T),
            rho=float(rho),
            qMin=float(state.qMin.detach().cpu().item()),
            cStar=cStar,
            arcLength=arcLength,
            correctorEvaluations=evaluations,
        )
        return point, criticalDirection

    def _refineEvent(
        self,
        composition: np.ndarray,
        leftState: np.ndarray,
        rightState: np.ndarray,
        leftC: float,
        rightC: float,
        index: int,
        *,
        eventKind: str = "crossing",
    ) -> CriticalSolveAttempt:
        fraction = abs(leftC) / max(abs(leftC) + abs(rightC), 1.0e-300)
        initial = leftState + fraction * (rightState - leftState)
        T, rho = np.exp(initial)
        source = (
            f"bell_event_{index}"
            if eventKind == "crossing"
            else f"bell_{eventKind}_event_{index}"
        )
        try:
            point = self.solver.correct(
                composition,
                float(T),
                float(rho),
                source=source,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return CriticalSolveAttempt(
                startT=float(T),
                startRho=float(rho),
                accepted=False,
                reason=f"{source}: {type(error).__name__}: {error}",
            )
        accepted = point.converged and point.nNegativeEigenvalues == 0
        reason = "accepted" if accepted else "not_converged_or_nonordinary"
        return CriticalSolveAttempt(
            startT=float(T),
            startRho=float(rho),
            accepted=accepted,
            reason=reason,
            point=point,
        )


__all__ = [
    "BellCriticalFinder",
    "BoundaryCriticalDiscoveryResult",
    "CriticalContourPoint",
    "CriticalDiscoveryConfig",
    "CriticalDiscoveryResult",
]
