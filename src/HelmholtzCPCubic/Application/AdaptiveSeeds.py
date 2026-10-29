"""
Find spinodal starting points on the four edges of the search region.
Use the smallest stability eigenvalue, remove duplicate corner roots and
assign an inward tracing direction. Edge states use log(T/K), log(rho/(mol/L)).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .AdaptiveBoundary import (
    AdaptiveBoundaryDiagnostic,
    _AdaptiveBoundaryScanner,
)
from .CriticalPoint import CriticalPointSolver
from .Discovery import CriticalDiscoveryConfig


_BoundarySeed = tuple[np.ndarray, int, int]
_SEED_LOG_TOL_FLOOR = 1.0e-9


@dataclass(frozen=True)
class _AdaptiveBoundarySeedResult:
    """
    Oriented seeds, work, and failures for one complete inset state box.
    """

    seeds: tuple[_BoundarySeed, ...]
    nEval: int
    failures: tuple[str, ...]
    diagnostics: tuple[AdaptiveBoundaryDiagnostic, ...]


def _findAdaptiveBoundarySeeds(
    solver: CriticalPointSolver,
    config: CriticalDiscoveryConfig,
    scanner: _AdaptiveBoundaryScanner,
    composition: np.ndarray,
    insetLower: np.ndarray,
    insetUpper: np.ndarray,
) -> _AdaptiveBoundarySeedResult:
    """
    Find oriented spinodal starting points on all four inset logarithmic state-box edges.

    ``composition`` is a dimensionless mole-fraction vector. ``insetLower``
    and ``insetUpper`` contain dimensionless ``(ln(T), ln(rho))`` bounds. Each
    returned seed stores its log state, the fixed edge axis, and the inward
    orientation sign required by the contour tracer. Roots from resolved
    panels remain usable when another panel is unresolved; the edge failure
    is retained in both ``failures`` and ``diagnostics``.
    """

    # Each edge fixes one log coordinate; the sign points tracing into the search region.
    edges = (
        ("lower_temperature", 0, insetLower[0], 1, 1),
        ("upper_temperature", 0, insetUpper[0], 1, -1),
        ("lower_density", 1, insetLower[1], 0, 1),
        ("upper_density", 1, insetUpper[1], 0, -1),
    )
    cache: dict[tuple[float, float], float] = {}
    seeds: list[_BoundarySeed] = []
    failures: list[str] = []
    diagnostics: list[AdaptiveBoundaryDiagnostic] = []

    def qMinAt(state: np.ndarray) -> float:
        """
        Evaluate and cache the dimensionless stability residual.
        """

        key = (float(state[0]), float(state[1]))
        if key not in cache:
            value = solver.kernel.stabilityResidual(state, composition)
            residual = float(value.detach().cpu().item())
            if not math.isfinite(residual):
                raise FloatingPointError("boundary stability residual is nonfinite")
            cache[key] = residual
        return cache[key]

    def addSeed(
        state: np.ndarray,
        orientationAxis: int,
        orientationSign: int,
    ) -> None:
        """
        Append one oriented seed unless a box-corner duplicate exists.
        """

        seedTol = max(config.stepTol, _SEED_LOG_TOL_FLOOR)
        if any(
            np.linalg.norm(state - existing[0]) <= seedTol for existing in seeds
        ):
            return
        seeds.append(
            (
                np.asarray(state, dtype=float),
                orientationAxis,
                orientationSign,
            )
        )

    for edgeName, fixedAxis, fixedValue, variableAxis, orientationSign in edges:
        def stateAt(coordinate: float) -> np.ndarray:
            """
            Map one scalar edge coordinate to ``(ln(T), ln(rho))``.
            """

            state = np.empty(2, dtype=float)
            state[fixedAxis] = fixedValue
            state[variableAxis] = coordinate
            return state

        nEvalBefore = len(cache)
        scan = scanner.scan(
            lambda coordinate: qMinAt(stateAt(coordinate)),
            float(insetLower[variableAxis]),
            float(insetUpper[variableAxis]),
        )
        diagnostics.append(
            AdaptiveBoundaryDiagnostic(
                edgeName=edgeName,
                nRoots=len(scan.roots),
                resolved=scan.resolved,
                nEval=len(cache) - nEvalBefore,
                maxDepthReached=scan.maxDepthReached,
                termination=scan.termination,
            )
        )
        if not scan.resolved:
            failures.append(f"{edgeName}_unresolved: {scan.termination}")

        # Each returned root has passed refinement. An unresolved panel does
        # not invalidate qualified roots elsewhere on the same edge.
        for root in scan.roots:
            addSeed(stateAt(root.coordinate), fixedAxis, orientationSign)

    seeds.sort(key=lambda item: (item[1], item[2], item[0][0], item[0][1]))
    return _AdaptiveBoundarySeedResult(
        seeds=tuple(seeds),
        nEval=len(cache),
        failures=tuple(failures),
        diagnostics=tuple(diagnostics),
    )
