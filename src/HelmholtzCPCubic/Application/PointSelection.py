"""
Model-independent selection utilities for critical-point candidates.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Callable, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .CriticalPoint import CriticalPointResult, CriticalSolverConfig


PointType = TypeVar("PointType")


def isDomainValidCP(
    point: CriticalPointResult,
    *,
    requirePositivePressure: bool = True,
) -> bool:
    """
    Return whether a traced point passes basic model-domain filters.

    Temperature, density and the model-domain margin must always be positive.
    Pressure must be finite and is positive by default; callers tracing a
    mathematical locus may retain negative-pressure states explicitly.  This
    check does not evaluate local fourth-order stability, TPD, parameter
    validity ranges or application engineering limits.
    """

    return (
        math.isfinite(point.T)
        and math.isfinite(point.P)
        and math.isfinite(point.rho)
        and point.T > 0.0
        and point.rho > 0.0
        and (not requirePositivePressure or point.P > 0.0)
        and point.domainMargin > 0.0
    )


def contiguousValidSegment(
    points: tuple[PointType, ...],
    seedIndex: int,
    keep: Callable[[PointType], bool],
) -> tuple[tuple[PointType, ...], bool, bool]:
    """
    Keep only the valid branch segment connected to its corrected seed.

    Returns the connected points and whether the reverse or forward side was
    cut by an invalid point. This prevents later valid states from being
    spliced across a failed or out-of-domain continuation state.
    """

    if not 0 <= seedIndex < len(points):
        raise ValueError("seedIndex must identify an input point")
    if not keep(points[seedIndex]):
        return (), seedIndex > 0, seedIndex < len(points) - 1
    lower = seedIndex
    while lower > 0 and keep(points[lower - 1]):
        lower -= 1
    upper = seedIndex + 1
    while upper < len(points) and keep(points[upper]):
        upper += 1
    return points[lower:upper], lower > 0, upper < len(points)


def deduplicatePoints(
    points: Iterable[CriticalPointResult],
    config: CriticalSolverConfig,
) -> list[CriticalPointResult]:
    """
    Deduplicate corrected roots and retain the lower-residual representative.
    """

    ordered = sorted(points, key=lambda point: (point.T, point.P))
    unique: list[CriticalPointResult] = []
    for point in ordered:
        match = None
        for index, existing in enumerate(unique):
            densityDifference = abs(point.rho - existing.rho) / max(
                point.rho,
                existing.rho,
                1.0,
            )
            if (
                abs(point.T - existing.T) <= config.dedupeTemperatureK
                and abs(point.P - existing.P) <= config.dedupePressureBar
                and densityDifference <= config.dedupeDensityRelative
            ):
                match = index
                break
        if match is None:
            unique.append(point)
        elif point.rNorm < unique[match].rNorm:
            unique[match] = point
    return unique
