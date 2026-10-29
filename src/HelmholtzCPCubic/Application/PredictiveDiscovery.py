"""
Adjust spinodal tracing steps using a linear forecast of the critical condition.
Estimate the next zero of cStar from two accepted points and limit the next step.
Use smaller steps during startup, near crossings and after difficult corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..Kernel.Critical import DensityStabilityState
from .Discovery import (
    BellCriticalFinder,
    CriticalContourPoint,
    CriticalDiscoveryConfig,
    CriticalDiscoveryResult,
)


@dataclass(frozen=True)
class LinearForecastStepConfig:
    """
    Controls for opt-in predictive spinodal tracing.

    Parameters
    ----------
    conservativeMaximumStep:
        Dimensionless logarithmic arc-step cap used for startup, crossings or difficult corrections.
    nominalMaximumStep:
        Dimensionless logarithmic arc-step ceiling on smooth segments.
    reserveSteps:
        Number of forecast steps retained before a predicted event.  Values
        below one are invalid because the mapped step could exceed the event
        distance.
    startupAcceptedPoints:
        Count of initial accepted contour samples kept under the conservative
        cap; at least two are required to form the secant forecast.
    """

    conservativeMaximumStep: float = 0.04
    nominalMaximumStep: float = 0.12
    reserveSteps: float = 1.0
    startupAcceptedPoints: int = 2

    def __post_init__(self) -> None:
        """
        Reject non-finite, unordered, or structurally invalid controls.
        """

        finiteValues = (
            self.conservativeMaximumStep,
            self.nominalMaximumStep,
            self.reserveSteps,
        )
        if any(isinstance(value, bool) for value in finiteValues) or not all(
            math.isfinite(value) for value in finiteValues
        ):
            raise ValueError("linear forecast step controls must be finite")
        if not (
            0.0
            < self.conservativeMaximumStep
            < self.nominalMaximumStep
        ):
            raise ValueError("require 0 < conservative step < nominal step")
        if self.reserveSteps < 1.0:
            raise ValueError("reserveSteps must be at least one")
        if (
            not isinstance(self.startupAcceptedPoints, int)
            or isinstance(self.startupAcceptedPoints, bool)
            or self.startupAcceptedPoints < 2
        ):
            raise ValueError("startupAcceptedPoints must be an integer >= 2")


@dataclass(frozen=True)
class PredictiveStepDecision:
    """
    One accepted contour point and the step cap selected for its successor.

    Arc length, forecast distance, slope denominator coordinate, and selected
    maximum step use the dimensionless logarithmic contour coordinate.
    ``cStar`` is the dimensionless third-order critical event scalar.
    """

    arcLength: float
    cStar: float
    selectedMaximumStep: float
    secantSlope: float
    predictedZeroDistance: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PredictiveTraceRecord:
    """
    Predictive step decisions retained for one traced spinodal component.
    """

    decisions: tuple[PredictiveStepDecision, ...]


class PredictiveBellCriticalFinder(BellCriticalFinder):
    """
    Spinodal finder with linear zero forecasting and adaptive step caps.

    Two accepted cStar samples give a linear zero-distance forecast. The step cap
    is clip(distance / reserveSteps, conservativeMaximumStep, nominalMaximumStep).
    The base finder keeps its own step policy. Finite sampling does not establish
    complete branch coverage, so rootSetCertified remains False.
    """

    def __init__(
        self,
        solver,
        config: CriticalDiscoveryConfig | None = None,
        stepConfig: LinearForecastStepConfig | None = None,
    ) -> None:
        """
        Create a predictive finder over an existing critical-point solver.

        Raises
        ------
        ValueError
            If the discovery nominal cap differs from the predictive nominal
            cap or its initial step exceeds the conservative cap.
        """

        self.stepConfig = stepConfig or LinearForecastStepConfig()
        activeConfig = config or CriticalDiscoveryConfig(
            maximumStep=self.stepConfig.nominalMaximumStep,
        )
        if not math.isclose(
            activeConfig.maximumStep,
            self.stepConfig.nominalMaximumStep,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "discovery maximumStep must equal predictive nominalMaximumStep"
            )
        if activeConfig.initialStep > self.stepConfig.conservativeMaximumStep:
            raise ValueError(
                "initialStep must not exceed predictive conservativeMaximumStep"
            )
        super().__init__(solver, activeConfig)
        self._eventHistory: list[tuple[float, float, int]] = []
        self._activeDecisions: list[PredictiveStepDecision] = []
        self._nextMaximumStep = self.stepConfig.conservativeMaximumStep
        self.tracePolicyRecords: list[PredictiveTraceRecord] = []

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
        Run one base trace while isolating per-component forecast state.
        """

        self._eventHistory = []
        self._activeDecisions = []
        self._nextMaximumStep = self.stepConfig.conservativeMaximumStep
        try:
            return super()._traceSeed(
                x,
                seed,
                seedEvaluations,
                orientationAxis=orientationAxis,
                orientationSign=orientationSign,
                coverageKind=coverageKind,
            )
        finally:
            self.tracePolicyRecords.append(
                PredictiveTraceRecord(tuple(self._activeDecisions))
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
        Create a base contour point and select its successor step cap.
        """

        point, direction = super()._contourPoint(
            logState,
            composition,
            arcLength,
            evaluations,
            referenceDirection=referenceDirection,
            stabilityState=stabilityState,
        )
        self._eventHistory.append(
            (point.arcLength, point.cStar, point.correctorEvaluations)
        )
        decision = self._selectStepDecision()
        self._activeDecisions.append(decision)
        self._nextMaximumStep = decision.selectedMaximumStep
        return point, direction

    def _acceptedStepSize(
        self,
        currentStep: float,
        correctorEvaluations: int,
    ) -> float:
        """
        Grow after acceptance and enforce the selected predictive cap.
        """

        return min(
            currentStep * self.config.growthFactor,
            self._nextMaximumStep,
        )

    def _contourSeedCovered(self, seed, contours) -> bool:
        """
        Use conservative resolution for boundary-component identity.
        """

        tolerance = 1.25 * self.stepConfig.conservativeMaximumStep
        return any(
            min(
                math.dist(
                    seed,
                    (math.log(point.T), math.log(point.rho)),
                )
                for point in result.contour
            )
            <= tolerance
            for result in contours
            if result.contour
        )

    def _selectStepDecision(self) -> PredictiveStepDecision:
        """
        Select a hard-risk cap or map a sole linear zero forecast.
        """

        arcLength, cStar, correctorEvaluations = self._eventHistory[-1]
        reasons: list[str] = []
        slope = math.nan
        predictedDistance = math.inf
        horizon = (
            self.stepConfig.reserveSteps
            * self.stepConfig.nominalMaximumStep
        )

        if len(self._eventHistory) < self.stepConfig.startupAcceptedPoints:
            reasons.append("startup")
        if correctorEvaluations > self.config.fastEval:
            reasons.append("corrector_difficulty")

        if len(self._eventHistory) >= 2:
            previousArc, previousCStar, _ = self._eventHistory[-2]
            deltaArc = arcLength - previousArc
            if deltaArc <= 0.0 or not math.isfinite(deltaArc):
                raise ValueError("accepted contour arc lengths must increase")
            slope = (cStar - previousCStar) / deltaArc
            if previousCStar * cStar <= 0.0 and previousCStar != cStar:
                reasons.append("sampled_crossing")
            if math.isfinite(slope) and slope != 0.0 and cStar * slope < 0.0:
                predictedDistance = abs(cStar / slope)
                if predictedDistance <= horizon:
                    reasons.append("linear_zero_forecast")

        uniqueReasons = tuple(dict.fromkeys(reasons))
        if uniqueReasons == ("linear_zero_forecast",):
            selectedMaximumStep = min(
                self.stepConfig.nominalMaximumStep,
                max(
                    self.stepConfig.conservativeMaximumStep,
                    predictedDistance / self.stepConfig.reserveSteps,
                ),
            )
            uniqueReasons += ("distance_mapped_forecast",)
        elif uniqueReasons:
            selectedMaximumStep = self.stepConfig.conservativeMaximumStep
        else:
            selectedMaximumStep = self.stepConfig.nominalMaximumStep

        return PredictiveStepDecision(
            arcLength=arcLength,
            cStar=cStar,
            selectedMaximumStep=selectedMaximumStep,
            secantSlope=slope,
            predictedZeroDistance=predictedDistance,
            reasons=uniqueReasons,
        )


__all__ = [
    "LinearForecastStepConfig",
    "PredictiveBellCriticalFinder",
    "PredictiveStepDecision",
    "PredictiveTraceRecord",
]
