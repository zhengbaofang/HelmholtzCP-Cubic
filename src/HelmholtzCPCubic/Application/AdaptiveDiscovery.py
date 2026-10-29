"""
Find fixed-composition critical points from automatically located edge roots.
Trace each spinodal, correct critical candidates and track edge crossings.
Return critical points in K, bar and mol/L, together with search status.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..Kernel.Critical import DensityStabilityState
from .AdaptiveBoundary import (
    AdaptiveBoundaryConfig,
    AdaptiveBoundaryDiagnostic,
    _AdaptiveBoundaryScanner,
)
from .AdaptiveSeeds import _BoundarySeed, _findAdaptiveBoundarySeeds
from .BoundaryCrossing import (
    BoundaryCrossingDiagnostic,
    _locateBoundaryCrossing,
)
from .CriticalPoint import CriticalPointSolver
from .Discovery import (
    BoundaryCriticalDiscoveryResult,
    CriticalContourPoint,
    CriticalDiscoveryConfig,
    CriticalDiscoveryResult,
)
from .PredictiveDiscovery import (
    LinearForecastStepConfig,
    PredictiveBellCriticalFinder,
)


_CROSSING_SEED_LOG_TOL = 1.0e-6
_EDGE_TOL_FACTOR = 10.0
_OUTER_GAP_FRACTION = 0.5


class AdaptiveBoundaryBellFinder(PredictiveBellCriticalFinder):
    """
    Discover critical points on adaptively resolved boundary components.

    The finder combines four-edge adaptive seed isolation, predictive spinodal
    contour propagation, and accepted-step inset-edge crossing correction.
    Public physical roots use K, bar, and mol/L. Boundary and contour states
    remain dimensionless ``(ln(T), ln(rho))`` coordinates internally.

    Correct edge crossings are used to avoid retracing a completed component.
    Closed interior components are outside this initialization, so
    rootSetCertified remains False. Unresolved edges and crossings are reported.
    """

    def __init__(
        self,
        solver: CriticalPointSolver,
        config: CriticalDiscoveryConfig | None = None,
        *,
        stepConfig: LinearForecastStepConfig | None = None,
        boundaryConfig: AdaptiveBoundaryConfig | None = None,
    ) -> None:
        """
        Create a finder with independent tracing and boundary controls.

        Parameters
        ----------
        solver:
            Critical-point solver defining the EOS and local Bell kernel.
        config:
            Contour, event, and state-box controls. If omitted, the predictive
            finder selects a configuration consistent with ``stepConfig``.
        stepConfig:
            Predictive contour-step controls in logarithmic arc coordinates.
        boundaryConfig:
            Adaptive edge-isolation tolerances and recursive work budget.

        Raises
        ------
        ValueError
            If either configuration is invalid or its contour-step ceilings
            are inconsistent, as defined by the underlying configuration
            contracts.
        """

        super().__init__(solver, config, stepConfig)
        self._boundaryScanner = _AdaptiveBoundaryScanner(boundaryConfig)
        self.boundaryDiagnostics: tuple[AdaptiveBoundaryDiagnostic, ...] = ()
        self.crossingDiagnostics: tuple[BoundaryCrossingDiagnostic, ...] = ()
        self._coverageComposition: np.ndarray | None = None
        self._outerBounds: tuple[np.ndarray, np.ndarray] | None = None
        self._insetBounds: tuple[np.ndarray, np.ndarray] | None = None
        self._activePreviousState: np.ndarray | None = None
        self._activeCrossings: list[np.ndarray] = []
        self._pendingBoundaryStepCap: float | None = None
        self._crossingsByContour: list[tuple[np.ndarray, ...]] = []
        self._crossingRecords: list[BoundaryCrossingDiagnostic] = []

    @property
    def boundaryConfig(self) -> AdaptiveBoundaryConfig:
        """
        Return the immutable adaptive boundary configuration.
        """

        return self._boundaryScanner.config

    def findBoundaryContours(
        self,
        composition: np.ndarray | list[float] | tuple[float, ...],
    ) -> BoundaryCriticalDiscoveryResult:
        """
        Discover boundary-reachable critical points conservatively.

        ``composition`` is a dimensionless mole-fraction vector. Returned
        physical roots use K, bar, and mol/L. An unresolved adaptive edge or
        localized crossing sets ``completed=False`` and is retained in
        ``boundaryFailures``. ``boundaryEvaluations`` counts both adaptive
        edge evaluations and localized crossing corrector evaluations.
        """

        self.boundaryDiagnostics = ()
        self.crossingDiagnostics = ()
        self._crossingsByContour = []
        self._crossingRecords = []
        try:
            result = super().findBoundaryContours(composition)
            crossingFailures = tuple(
                "crossing_unresolved: "
                f"axis={item.fixedAxis}, logEdge={item.logEdge:.17g}: "
                f"{item.termination}"
                for item in self._crossingRecords
                if not item.resolved
            )
            nCrossingEval = sum(item.nEval for item in self._crossingRecords)
            return replace(
                result,
                completed=result.completed and not crossingFailures,
                boundaryEvaluations=result.boundaryEvaluations + nCrossingEval,
                boundaryFailures=result.boundaryFailures + crossingFailures,
                coverageKind="adaptive_boundary_sign_change_spinodals",
                rootSetCertified=False,
            )
        finally:
            self.crossingDiagnostics = tuple(self._crossingRecords)
            self._coverageComposition = None
            self._outerBounds = None
            self._insetBounds = None
            self._activePreviousState = None
            self._activeCrossings = []
            self._pendingBoundaryStepCap = None

    def _boundarySeeds(
        self,
        composition: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[tuple[_BoundarySeed, ...], int, tuple[str, ...]]:
        """
        Return adaptive seeds and retain crossing-localization geometry.
        """

        span = upper - lower
        insetLower = lower + self.config.boundaryInsetFraction * span
        insetUpper = upper - self.config.boundaryInsetFraction * span
        self._coverageComposition = composition.copy()
        self._outerBounds = (lower.copy(), upper.copy())
        self._insetBounds = (insetLower.copy(), insetUpper.copy())
        result = _findAdaptiveBoundarySeeds(
            self.solver,
            self.config,
            self._boundaryScanner,
            composition,
            insetLower,
            insetUpper,
        )
        self.boundaryDiagnostics = result.diagnostics
        return result.seeds, result.nEval, result.failures

    def _traceSeed(
        self,
        composition: np.ndarray,
        seed: np.ndarray,
        seedEvaluations: int,
        *,
        orientationAxis: int,
        orientationSign: int,
        coverageKind: str,
    ) -> CriticalDiscoveryResult:
        """
        Isolate crossing records for one traced spinodal component.
        """

        self._activePreviousState = None
        self._activeCrossings = []
        self._pendingBoundaryStepCap = None
        result = super()._traceSeed(
            composition,
            seed,
            seedEvaluations,
            orientationAxis=orientationAxis,
            orientationSign=orientationSign,
            coverageKind=coverageKind,
        )

        # Only completed traces can mark another edge start as already covered.
        self._crossingsByContour.append(
            tuple(state.copy() for state in self._activeCrossings)
            if result.completed
            else ()
        )
        return result

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
        Record crossed inset edges and select the next boundary step cap.
        """

        point, direction = super()._contourPoint(
            logState,
            composition,
            arcLength,
            evaluations,
            referenceDirection=referenceDirection,
            stabilityState=stabilityState,
        )
        currentState = np.asarray(logState, dtype=float)
        previousState = self._activePreviousState
        if previousState is not None:
            self._recordCrossings(previousState, currentState)
            self._pendingBoundaryStepCap = self._boundaryStepCap(
                previousState,
                currentState,
            )
        self._activePreviousState = currentState.copy()
        return point, direction

    def _acceptedStepSize(
        self,
        currentStep: float,
        correctorEvaluations: int,
    ) -> float:
        """
        Apply predictive event and pending inset-edge step caps.
        """

        selected = super()._acceptedStepSize(currentStep, correctorEvaluations)
        if self._pendingBoundaryStepCap is not None:
            selected = min(selected, self._pendingBoundaryStepCap)
        self._pendingBoundaryStepCap = None
        return selected

    def _boundaryStepCap(
        self,
        previousState: np.ndarray,
        currentState: np.ndarray,
    ) -> float | None:
        """
        Place the next predictor between the inset and outer boundaries.
        """

        if self._insetBounds is None or self._outerBounds is None:
            return None
        insetLower, insetUpper = self._insetBounds
        tolerance = _EDGE_TOL_FACTOR * self.config.stepTol
        if np.any(currentState <= insetLower + tolerance) or np.any(
            currentState >= insetUpper - tolerance
        ):
            return None
        secant = currentState - previousState
        norm = float(np.linalg.norm(secant))
        if not math.isfinite(norm) or norm <= self.config.stepTol:
            return None
        tangent = secant / norm
        lower, upper = self._outerBounds
        candidates: list[tuple[float, float]] = []
        for axis in range(currentState.size):
            if tangent[axis] < 0.0:
                distance = (
                    insetLower[axis] - currentState[axis]
                ) / tangent[axis]
                outerGap = (lower[axis] - insetLower[axis]) / tangent[axis]
            elif tangent[axis] > 0.0:
                distance = (
                    insetUpper[axis] - currentState[axis]
                ) / tangent[axis]
                outerGap = (upper[axis] - insetUpper[axis]) / tangent[axis]
            else:
                continue
            if 0.0 < distance <= self.stepConfig.nominalMaximumStep:
                candidates.append((distance, outerGap))
        if not candidates:
            return None
        distance, outerGap = min(candidates, key=lambda item: item[0])
        return min(
            self.stepConfig.nominalMaximumStep,
            distance + _OUTER_GAP_FRACTION * outerGap,
        )

    def _recordCrossings(
        self,
        previousState: np.ndarray,
        currentState: np.ndarray,
    ) -> None:
        """
        Localize every outward inset-edge crossing in one accepted segment.
        """

        if (
            self._coverageComposition is None
            or self._outerBounds is None
            or self._insetBounds is None
        ):
            return
        insetLower, insetUpper = self._insetBounds
        tolerance = _EDGE_TOL_FACTOR * self.config.stepTol
        crossedEdges: list[tuple[int, float]] = []
        for axis in range(currentState.size):
            if previousState[axis] > insetLower[axis] + tolerance and (
                currentState[axis] <= insetLower[axis]
            ):
                crossedEdges.append((axis, float(insetLower[axis])))
            if previousState[axis] < insetUpper[axis] - tolerance and (
                currentState[axis] >= insetUpper[axis]
            ):
                crossedEdges.append((axis, float(insetUpper[axis])))
        if not crossedEdges:
            return

        def evaluateManifold(
            state: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Evaluate the dimensionless spinodal residual and Jacobian.
            """

            evaluation = self.solver.kernel.stabilityEvaluation(
                state,
                self._coverageComposition,
            )
            return self._stabilityArrays(evaluation)

        lower, upper = self._outerBounds
        for fixedAxis, logEdge in crossedEdges:
            crossing = _locateBoundaryCrossing(
                evaluateManifold,
                previousState,
                currentState,
                fixedAxis,
                logEdge,
                lower,
                upper,
                resTol=self.config.resTol,
                stepTol=self.config.stepTol,
                maxCorrectorEval=self.config.maximumCorrectorEvaluations,
                acceptanceRatio=self.config.acceptanceRatio,
                maxRadius=self.stepConfig.nominalMaximumStep,
            )
            self._crossingRecords.append(crossing)
            if crossing.resolved and crossing.logState is not None:
                self._activeCrossings.append(
                    np.asarray(crossing.logState, dtype=float)
                )

    def _contourSeedCovered(
        self,
        seed: np.ndarray,
        contours: list[CriticalDiscoveryResult],
    ) -> bool:
        """
        Use conservative distance, then corrected edge intersections.
        """

        if super()._contourSeedCovered(seed, contours):
            return True
        return any(
            np.linalg.norm(seed - crossing) <= _CROSSING_SEED_LOG_TOL
            for contourCrossings in self._crossingsByContour
            for crossing in contourCrossings
        )


__all__ = [
    "AdaptiveBoundaryBellFinder",
]
