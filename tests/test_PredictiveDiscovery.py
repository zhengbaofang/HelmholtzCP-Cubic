"""
Check predictive step controls and recovery of nearby critical points.
The full-search comparison is marked slow and excluded by default.
Fluid states use K, bar and mol/L; logarithmic tracing steps are dimensionless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from HelmholtzCPCubic import (
    CubicBellModel,
    CriticalPointSolver,
    PredictiveBellCriticalFinder,
)
from HelmholtzCPCubic.Application import (
    BellCriticalFinder,
    CriticalDiscoveryConfig,
    LinearForecastStepConfig,
)


COMPOSITION = (0.52, 0.48)


def _buildSolver() -> CriticalPointSolver:
    """
    Return Bell and Jäger (2017), Table-1 C1--H2S/SRK inputs.
    """

    model = CubicBellModel(
        criticalTemp=(190.4, 373.2),
        criticalPres=(46.0, 89.4),
        acentricFactor=(0.011, 0.097),
        cubicKij=((0.0, 0.08), (0.08, 0.0)),
        mode="SRK",
    )
    return CriticalPointSolver(model)


def _algorithmEvaluations(result) -> int:
    """
    Count seed, contour, and event-corrector evaluations.
    """

    return int(result.seedEvaluations) + sum(
        int(point.correctorEvaluations) for point in result.contour
    ) + sum(
        int(attempt.point.nEval or 0)
        for attempt in result.attempts
        if attempt.point is not None
    )


def test_predictive_config_rejects_unsafe_or_inconsistent_controls() -> None:
    """
    Keep the opt-in policy explicit and internally ordered.
    """

    for invalidReserve in (True, 0.5, math.nan, math.inf):
        with pytest.raises(ValueError):
            LinearForecastStepConfig(reserveSteps=invalidReserve)
    with pytest.raises(ValueError, match="conservative step"):
        LinearForecastStepConfig(
            conservativeMaximumStep=0.12,
            nominalMaximumStep=0.12,
        )
    with pytest.raises(ValueError, match="startupAcceptedPoints"):
        LinearForecastStepConfig(startupAcceptedPoints=1)

    solver = _buildSolver()
    with pytest.raises(ValueError, match="maximumStep"):
        PredictiveBellCriticalFinder(
            solver,
            CriticalDiscoveryConfig(maximumStep=0.08),
        )
    with pytest.raises(ValueError, match="initialStep"):
        PredictiveBellCriticalFinder(
            solver,
            CriticalDiscoveryConfig(
                initialStep=0.05,
                maximumStep=0.12,
            ),
        )


def test_base_accepted_step_hook_preserves_existing_growth_behavior() -> None:
    """
    Check step growth on both sides of the fast-correction evaluation limit.
    """

    finder = BellCriticalFinder(_buildSolver())
    assert finder._acceptedStepSize(0.03, finder.config.fastEval) == pytest.approx(
        0.0375
    )
    assert finder._acceptedStepSize(
        0.03,
        finder.config.fastEval + 1,
    ) == pytest.approx(0.03)


@pytest.mark.slow
def test_predictive_finder_recovers_known_coarse_step_omission() -> None:
    """
    Recover both nearby ordinary roots without enumeration or fallback.
    """

    solver = _buildSolver()
    reference = BellCriticalFinder(
        solver,
        CriticalDiscoveryConfig(initialStep=0.015, maximumStep=0.02),
    ).find(COMPOSITION)
    coarse = BellCriticalFinder(
        solver,
        CriticalDiscoveryConfig(maximumStep=0.12),
    ).find(COMPOSITION)
    finder = PredictiveBellCriticalFinder(solver)
    predictive = finder.find(COMPOSITION)

    assert reference.completed
    assert coarse.completed
    assert predictive.completed
    assert len(reference.roots) == len(predictive.roots) == 2
    assert len(coarse.roots) == 1
    np.testing.assert_allclose(
        sorted(point.T for point in predictive.roots),
        sorted(point.T for point in reference.roots),
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        sorted(point.P for point in predictive.roots),
        sorted(point.P for point in reference.roots),
        rtol=0.0,
        atol=1.0e-4,
    )
    assert predictive.rejectedSteps == 0
    assert max(point.rNorm for point in predictive.roots) <= 1.0e-8
    assert _algorithmEvaluations(predictive) == 97
    assert _algorithmEvaluations(predictive) < _algorithmEvaluations(reference)
    reasons = {
        reason
        for record in finder.tracePolicyRecords
        for decision in record.decisions
        for reason in decision.reasons
    }
    assert "linear_zero_forecast" in reasons
    assert "distance_mapped_forecast" in reasons
