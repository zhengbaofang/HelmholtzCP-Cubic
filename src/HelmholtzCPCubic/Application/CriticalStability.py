"""
Define the critical-state inputs and phase-stability classification interface.
A classifier reads temperature [K], pressure [bar], density [mol/L] and
composition [-], and returns a status with an optional stability measure.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol, TypeAlias


class CriticalStabilityStatus(str, Enum):
    """
    Operational phase-stability verdict returned by a classifier.
    """

    NOT_EVALUATED = "not_evaluated"
    STABLE = "stable"
    METASTABLE = "metastable"
    UNSTABLE = "unstable"
    FAILED = "failed"


class CriticalState(Protocol):
    """
    Minimum corrected-state fields required for stability evaluation.
    """

    @property
    def T(self) -> float:
        """
        Temperature in kelvin.
        """

    @property
    def P(self) -> float:
        """
        Pressure in bar.
        """

    @property
    def rho(self) -> float:
        """
        Candidate-state molar density in mol/L.
        """

    @property
    def composition(self) -> tuple[float, ...]:
        """
        Mole fractions in model component order.
        """

    @property
    def rNorm(self) -> float:
        """
        Dimensionless corrected critical-equation residual norm.
        """


StabilityClassifier: TypeAlias = Callable[
    [CriticalState],
    tuple[CriticalStabilityStatus, float | None],
]


__all__ = ["CriticalState", "CriticalStabilityStatus", "StabilityClassifier"]
