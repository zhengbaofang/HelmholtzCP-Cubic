r"""
Classify critical-point candidates using local curvature and TPD searches.
Search trial composition and density jointly, then check the original TPD.
Return stable, unstable or failed, with the lowest TPD and any failure details.
State inputs use K, bar and mol/L; TPD is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Common.Model import FreeEnergyModel
from ..Equilibrium.MolarTPD import findMolarTpdMinima
from ..Equilibrium.TPD import TPDResult
from ..Kernel.LocalStability import LocalStabilityKernel, LocalStabilityResult
from ..Numerics.ScalarTrustRegion import ScalarTrustRegionConfig
from ..Transform.WorkspaceUnits import asWorkspaceModel
from .CriticalStability import CriticalState, CriticalStabilityStatus

# Dimensionless TPD tolerance: only TPD < -tolerance indicates instability.
TPD_STABILITY_TOLERANCE = 1.0e-10
FEED_NORM_TOL = 1.0e-10


def _isFeedStart(
    trialComposition: np.ndarray,
    feedComposition: np.ndarray,
) -> bool:
    """
    Return whether normalized compositions are the same numerical state.

    This composition-only compatibility helper is not used to discard starts
    in the joint density search: equal compositions can have competing density
    branches. Derivative evaluations at nearby states remain independent.
    """

    trial = np.asarray(trialComposition, dtype=float)
    feed = np.asarray(feedComposition, dtype=float)
    if trial.ndim != 1 or trial.shape != feed.shape:
        raise ValueError("composition vectors must have equal one-dimensional shape")
    if not np.all(np.isfinite(trial)) or not np.all(np.isfinite(feed)):
        raise ValueError("composition vectors must be finite")
    return bool(
        np.linalg.norm(trial - feed, ord=2)
        <= FEED_NORM_TOL
    )


@dataclass(frozen=True)
class TPDStabilityResult:
    """
    One finite stability assessment and its calculated local and TPD results.

    ``lowestTpdValue`` is dimensionless. ``local`` uses the classifier's
    K/bar/mol/L model; ``tpd`` retains its search counts, minima, witness and
    failures without recomputation. Either can be absent if its stage failed.
    Pure-fluid identity classification also has no TPD search result.
    ``failureReason`` describes a failed assessment; individual search failures
    remain in ``tpd`` even when a negative witness establishes instability.
    """

    status: CriticalStabilityStatus
    lowestTpdValue: float | None
    local: LocalStabilityResult | None = None
    tpd: TPDResult | None = None
    failureReason: str | None = None


@dataclass(frozen=True)
class TPDStabilityClassifier:
    """
    Classify ordinary critical points by finite multistart TPD searches.

    Args:
        model: Free-energy-density model used for the TPD evaluation.
        tolerance: ``-tolerance <= D_min`` classifies a completed search as
            operationally stable.
        config: Optional scalar trust-region configuration forwarded to the
            TPD minimization.

    The callable returns ``(CriticalStabilityStatus, lowest assessed D | None)``.
    The metric can come from the first sufficiently negative feasible witness
    rather than a converged minimum. Classification stops at that witness.
    Use ``evaluate(point)`` to retain the evidence in the same assessment.
    """

    model: FreeEnergyModel
    tolerance: float = TPD_STABILITY_TOLERANCE
    config: ScalarTrustRegionConfig | None = None

    def __post_init__(self) -> None:
        """
        Normalize the EOS to the public critical-state units and validate tolerance.
        """

        object.__setattr__(self, "model", asWorkspaceModel(self.model))
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("TPD stability tolerance must be finite and positive")

    def __call__(
        self,
        point: CriticalState,
    ) -> tuple[CriticalStabilityStatus, float | None]:
        """
        Return (status, lowest TPD) for a critical point in K, bar and mol/L.
        """

        result = self.evaluate(point)
        return result.status, result.lowestTpdValue

    def evaluate(self, point: CriticalState) -> TPDStabilityResult:
        """
        Assess a critical point in K/bar/mol/L using local curvature and TPD.
        Local qualification runs once. A qualified reference skips only its matching
        density root; other stable roots at the feed composition remain starts.
        A qualified TPD below -tolerance returns UNSTABLE immediately. Otherwise,
        STABLE requires a completed search; an incomplete search returns FAILED.
        A locally stable pure fluid returns STABLE without a multistart search.

        Zero fractions require an active submodel. Numerical failures return FAILED
        with their stage and cause; invalid composition or residual input raises
        ValueError. Noncritical states use findMolarTpdMinima directly. This classifier
        reports metastable critical states as UNSTABLE when a negative TPD is found.
        """

        composition = np.asarray(point.composition, dtype=float)
        if (
            composition.shape != (self.model.componentCount,)
            or not np.all(np.isfinite(composition))
            or np.any(composition < 0.0)
            or not np.isfinite(point.rNorm)
            or point.rNorm < 0.0
        ):
            raise ValueError("critical state contains invalid composition or residual")
        if not np.all(composition > 0.0):
            return TPDStabilityResult(
                CriticalStabilityStatus.FAILED, None,
                failureReason="local qualification: active-submodel treatment required for zero fractions",
            )
        try:
            local = LocalStabilityKernel(self.model).evaluate(
                point.T,
                point.rho,
                point.composition,
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            return TPDStabilityResult(
                CriticalStabilityStatus.FAILED, None,
                failureReason=f"local qualification: {type(error).__name__}: {error}",
            )
        try:
            if local.isLocallyStable is True:
                if len(point.composition) == 1:

                    # A locally stable pure-fluid critical point has only the trivial composition.
                    return TPDStabilityResult(CriticalStabilityStatus.STABLE, 0.0, local)
            result: TPDResult = findMolarTpdMinima(
                self.model,
                point.composition,
                point.T,
                point.P,
                config=self.config,
                referenceDensityMolL=point.rho,
                stopBelow=-self.tolerance,
                qualifiedReference=local.isLocallyStable is True,
            )
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:

            # Also retain failures before the start loop, such as an invalid reference state.
            return TPDStabilityResult(
                CriticalStabilityStatus.FAILED, None, local,
                failureReason=f"TPD evaluation: {type(error).__name__}: {error}",
            )
        values = [item.value for item in (result.minimum, result.witness) if item is not None]
        if not values:
            return TPDStabilityResult(
                CriticalStabilityStatus.FAILED, None, local, result,
                failureReason=result.message,
            )
        minimum = float(min(values))
        if minimum < -self.tolerance:
            return TPDStabilityResult(CriticalStabilityStatus.UNSTABLE, minimum, local, result)
        if result.minimum is None or not result.completed:
            return TPDStabilityResult(
                CriticalStabilityStatus.FAILED, minimum, local, result,
                failureReason=result.message,
            )
        return TPDStabilityResult(CriticalStabilityStatus.STABLE, minimum, local, result)


__all__ = [
    "FEED_NORM_TOL",
    "TPDStabilityClassifier",
    "TPDStabilityResult",
    "TPD_STABILITY_TOLERANCE",
]
