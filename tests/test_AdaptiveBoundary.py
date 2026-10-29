"""
Check adaptive edge scans for crossings, close roots and tangencies.
Synthetic function coordinates and residuals are dimensionless.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from HelmholtzCPCubic.Application.AdaptiveBoundary import AdaptiveBoundaryConfig, _AdaptiveBoundaryScanner


def _rootCoordinates(scan) -> np.ndarray:
    """
    Return sorted scalar coordinates from one internal edge scan.
    """

    return np.asarray(
        sorted(root.coordinate for root in scan.roots),
        dtype=float,
    )


def test_adaptive_scanner_resolves_synthetic_root_topologies() -> None:
    """
    Resolve exclusion, crossings, a close pair, and an even tangency.
    """

    scanner = _AdaptiveBoundaryScanner()
    noRoot = scanner.scan(lambda value: value**2 + 0.25, -1.0, 1.0)
    crossing = scanner.scan(lambda value: value - 0.2, -1.0, 1.0)
    closePair = scanner.scan(
        lambda value: (value - 0.13) ** 2 - 0.008**2,
        -1.0,
        1.0,
    )
    tangent = scanner.scan(lambda value: (value + 0.27) ** 2, -1.0, 1.0)

    assert noRoot.resolved and not noRoot.roots
    assert crossing.resolved
    assert np.allclose(_rootCoordinates(crossing), (0.2,), atol=1.0e-9)
    assert closePair.resolved
    assert np.allclose(
        _rootCoordinates(closePair),
        (0.122, 0.138),
        atol=1.0e-8,
    )
    assert tangent.resolved
    assert np.allclose(_rootCoordinates(tangent), (-0.27,), atol=1.0e-7)


@pytest.mark.xfail(
    strict=True,
    reason="Known baseline failure: the high-frequency fixture returns resolved; see docs/Development.md",
)
def test_adaptive_scanner_reports_exhausted_resolution() -> None:
    """
    Expose a hard work-budget failure instead of silently omitting roots.
    """

    scan = _AdaptiveBoundaryScanner(
        AdaptiveBoundaryConfig(maxDepth=1)
    ).scan(lambda value: math.sin(95.0 * value), -1.0, 1.0)

    assert not scan.resolved
    assert "unresolved" in scan.termination
    assert scan.maxDepthReached == 1

