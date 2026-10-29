"""
Find starting points for spinodal tracing on the edges of the search region.
Use Chebyshev interpolation and adaptive subdivision to locate roots.
Refine roots using the original residual and report unresolved intervals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from numpy.polynomial import chebyshev
from scipy.optimize import minimize_scalar

from ..Numerics.BracketRoot import solveBracket

_ScalarFunction = Callable[[float], float]
_LOWER_DEGREE = 3
_HIGHER_DEGREE = 6
_SENTINEL_COORDS = (-0.37,)
_REAL_ROOT_IMAG_TOL = 1.0e-6
_NORMALIZED_BOUND_TOL = 1.0e-10
_CACHE_GRID_TOL = 1.0e-12
_ROOT_DEDUP_REL_TOL = 1.0e-8
_MARGIN_SAMPLE_COUNT = 129
_MAX_BRACKET_ITE = 100


@dataclass(frozen=True)
class AdaptiveBoundaryConfig:
    """
    Controls for adaptive root searches along a search-region edge.

    Parameters
    ----------
    maxDepth:
        Maximum interval-bisection depth; unresolved intervals are reported.
    rootResTol:
        Absolute tolerance for the dimensionless stability residual.
    rootCoordTol:
        Absolute tolerance in the dimensionless logarithmic edge coordinate.
    topologyTol:
        Maximum root-position mismatch between degree-3 and degree-6
        interpolants, measured on the normalized panel.
    relInterpTol:
        Maximum interpolation error relative to the local residual scale.
    exclusionFactor:
        Safety factor for deciding whether an interval contains no root.

    Polynomial degrees and off-grid check points are fixed in this module.
    """

    maxDepth: int = 9
    rootResTol: float = 1.0e-8
    rootCoordTol: float = 1.0e-10
    topologyTol: float = 0.15
    relInterpTol: float = 1.0e-2
    exclusionFactor: float = 1.0

    def __post_init__(self) -> None:
        """
        Reject invalid work budgets and numerical controls.
        """

        if (
            not isinstance(self.maxDepth, int)
            or isinstance(self.maxDepth, bool)
            or self.maxDepth < 0
        ):
            raise ValueError("maxDepth must be a nonnegative integer")
        for name in (
            "rootResTol",
            "rootCoordTol",
            "topologyTol",
            "relInterpTol",
            "exclusionFactor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.topologyTol >= 2.0:
            raise ValueError("topologyTol must be smaller than the panel width")
        if self.relInterpTol >= 1.0:
            raise ValueError("relInterpTol must be smaller than one")
        if self.exclusionFactor < 1.0:
            raise ValueError("exclusionFactor must be at least one")


@dataclass(frozen=True)
class AdaptiveBoundaryDiagnostic:
    """
    Root count, completion status and failures for one logarithmic search edge.

    Parameters
    ----------
    edgeName:
        Lower/upper temperature or density edge.
    nRoots:
        Number of refined roots accepted on this edge.
    resolved:
        Whether every subinterval was resolved.
    nEval:
        Number of distinct stability-residual evaluations.
    maxDepthReached:
        Deepest bisection level visited, starting from zero.
    termination:
        Completion or failure reason.
    """

    edgeName: str
    nRoots: int
    resolved: bool
    nEval: int
    maxDepthReached: int
    termination: str


@dataclass(frozen=True)
class _BoundaryRoot:
    """
    One refined scalar boundary root in natural-log state coordinate.
    """

    coordinate: float
    residual: float


@dataclass(frozen=True)
class _BoundaryPanelResult:
    """
    Roots and resolution decision for one adaptive edge panel.
    """

    roots: tuple[_BoundaryRoot, ...]
    resolved: bool


@dataclass(frozen=True)
class _BoundaryScanResult:
    """
    Roots, work, and resolution status for one complete edge.
    """

    roots: tuple[_BoundaryRoot, ...]
    resolved: bool
    nEval: int
    maxDepthReached: int
    termination: str


def _realPolynomialRoots(coefficients: np.ndarray) -> tuple[float, ...]:
    """
    Return real Chebyshev roots inside the normalized panel.
    """

    roots = chebyshev.chebroots(coefficients)
    values = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) <= _REAL_ROOT_IMAG_TOL
        and -1.0 - _NORMALIZED_BOUND_TOL
        <= root.real
        <= 1.0 + _NORMALIZED_BOUND_TOL
    )
    return tuple(float(np.clip(value, -1.0, 1.0)) for value in values)


def _rootTopologiesMatch(
    lowerRoots: tuple[float, ...],
    higherRoots: tuple[float, ...],
    tolerance: float,
) -> bool:
    """
    Return whether nested interpolants resolve the same normalized roots.
    """

    return len(lowerRoots) == len(higherRoots) and all(
        abs(lower - higher) <= tolerance
        for lower, higher in zip(lowerRoots, higherRoots, strict=True)
    )


class _AdaptiveBoundaryScanner:
    """
    Resolve scalar edge roots by local spectral checks and bisection.
    """

    def __init__(self, config: AdaptiveBoundaryConfig | None = None) -> None:
        """
        Create a scanner with one set of adaptive scan controls.
        """

        self.config = config or AdaptiveBoundaryConfig()

    def scan(
        self,
        evaluate: _ScalarFunction,
        lower: float,
        upper: float,
    ) -> _BoundaryScanResult:
        """
        Return all numerically resolved roots on one finite interval.

        ``evaluate`` returns the dimensionless scalar residual at a natural-log
        edge coordinate. Bounds must be finite and strictly increasing. A
        non-finite model evaluation or exhausted panel depth returns an
        unresolved scan with a termination reason.
        """

        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("boundary scan bounds must be finite and increasing")

        cache: dict[float, float] = {}
        leafDepths: list[int] = []
        roots: list[_BoundaryRoot] = []
        nUnresolved = 0

        def cachedEvaluate(coordinate: float) -> float:
            """
            Evaluate one scalar coordinate once and reject non-finite data.
            """

            key = float(coordinate)
            if key not in cache:
                value = float(evaluate(key))
                if not math.isfinite(value):
                    raise FloatingPointError("boundary function is nonfinite")
                cache[key] = value
            return cache[key]

        def sentinelCoords(panelLower: float, panelUpper: float) -> np.ndarray:
            """
            Reuse cached off-grid ancestor samples when available.
            """

            midpoint = 0.5 * (panelLower + panelUpper)
            halfSpan = 0.5 * (panelUpper - panelLower)
            highNormalized = np.cos(
                np.arange(_HIGHER_DEGREE + 1) * np.pi / _HIGHER_DEGREE
            )
            candidates = []
            for coordinate in cache:
                normalized = (coordinate - midpoint) / halfSpan
                if not -1.0 < normalized < 1.0:
                    continue
                if (
                    np.min(np.abs(highNormalized - normalized))
                    <= _CACHE_GRID_TOL
                ):
                    continue
                candidates.append((coordinate, normalized))

            selected: list[float] = []
            unused = list(candidates)
            for desired in _SENTINEL_COORDS:
                if unused:
                    index = min(
                        range(len(unused)),
                        key=lambda item: abs(unused[item][1] - desired),
                    )
                    selected.append(float(unused.pop(index)[0]))
                else:
                    selected.append(float(midpoint + halfSpan * desired))
            return np.asarray(selected, dtype=float)

        def recurse(panelLower: float, panelUpper: float, depth: int) -> bool:
            """
            Resolve one panel or bisect it until the work budget is exhausted.
            """

            nonlocal nUnresolved
            panel = self._analyzePanel(
                cachedEvaluate,
                panelLower,
                panelUpper,
                sentinelCoords(panelLower, panelUpper),
            )
            if panel.resolved:
                roots.extend(panel.roots)
                leafDepths.append(depth)
                return True
            if depth >= self.config.maxDepth:
                leafDepths.append(depth)
                nUnresolved += 1
                return False
            midpoint = 0.5 * (panelLower + panelUpper)
            leftResolved = recurse(panelLower, midpoint, depth + 1)
            rightResolved = recurse(midpoint, panelUpper, depth + 1)
            return leftResolved and rightResolved

        try:
            resolved = recurse(lower, upper, 0)
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            return _BoundaryScanResult(
                roots=(),
                resolved=False,
                nEval=len(cache),
                maxDepthReached=max(leafDepths, default=0),
                termination=f"evaluation_failed: {type(error).__name__}: {error}",
            )

        uniqueRoots: list[_BoundaryRoot] = []
        for root in sorted(roots, key=lambda item: item.coordinate):
            duplicateTol = max(
                self.config.rootCoordTol,
                _ROOT_DEDUP_REL_TOL * max(1.0, abs(root.coordinate)),
            )
            if (
                uniqueRoots
                and abs(root.coordinate - uniqueRoots[-1].coordinate)
                <= duplicateTol
            ):
                if root.residual < uniqueRoots[-1].residual:
                    uniqueRoots[-1] = root
                continue
            uniqueRoots.append(root)
        maxDepthReached = max(leafDepths, default=0)
        termination = "resolved" if resolved else f"{nUnresolved}_panels_unresolved"
        return _BoundaryScanResult(
            roots=tuple(uniqueRoots),
            resolved=resolved,
            nEval=len(cache),
            maxDepthReached=maxDepthReached,
            termination=termination,
        )

    def _analyzePanel(
        self,
        evaluate: _ScalarFunction,
        lower: float,
        upper: float,
        sentinelCoords: np.ndarray,
    ) -> _BoundaryPanelResult:
        """
        Classify one panel as excluded, root-resolved, or unresolved.
        """

        lowNormalized = np.cos(
            np.arange(_LOWER_DEGREE + 1) * np.pi / _LOWER_DEGREE
        )
        highNormalized = np.cos(
            np.arange(_HIGHER_DEGREE + 1) * np.pi / _HIGHER_DEGREE
        )

        def mapCoordinate(normalized: np.ndarray | float) -> np.ndarray | float:
            """
            Map the standard Chebyshev interval onto the current panel.
            """

            return 0.5 * (lower + upper) + 0.5 * (upper - lower) * normalized

        lowCoordinates = np.asarray(mapCoordinate(lowNormalized), dtype=float)
        highCoordinates = np.asarray(mapCoordinate(highNormalized), dtype=float)
        lowValues = np.asarray([evaluate(value) for value in lowCoordinates])
        highValues = np.asarray([evaluate(value) for value in highCoordinates])
        lowCoefficients = chebyshev.chebfit(
            lowNormalized,
            lowValues,
            _LOWER_DEGREE,
        )
        highCoefficients = chebyshev.chebfit(
            highNormalized,
            highValues,
            _HIGHER_DEGREE,
        )
        sentinelNormalized = (
            np.asarray(sentinelCoords, dtype=float) - 0.5 * (lower + upper)
        ) / (0.5 * (upper - lower))
        sentinelValues = np.asarray([evaluate(value) for value in sentinelCoords])
        interpolationError = max(
            float(
                np.max(
                    np.abs(
                        highValues
                        - chebyshev.chebval(highNormalized, lowCoefficients)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        sentinelValues
                        - chebyshev.chebval(
                            sentinelNormalized,
                            highCoefficients,
                        )
                    )
                )
            ),
            float(np.sum(np.abs(highCoefficients[-2:]))),
        )
        valueScale = max(
            1.0,
            float(np.max(np.abs(highValues))),
            float(np.max(np.abs(sentinelValues))),
        )
        lowerRoots = _realPolynomialRoots(lowCoefficients)
        higherRoots = _realPolynomialRoots(highCoefficients)
        denseCoordinates = np.linspace(-1.0, 1.0, _MARGIN_SAMPLE_COUNT)
        polynomialMargin = float(
            np.min(np.abs(chebyshev.chebval(denseCoordinates, highCoefficients)))
        )
        exclusionResolved = (
            not higherRoots
            and polynomialMargin
            > self.config.exclusionFactor * interpolationError
            + self.config.rootResTol
        )
        topologyResolved = (
            bool(higherRoots)
            and _rootTopologiesMatch(
                lowerRoots,
                higherRoots,
                self.config.topologyTol,
            )
        )

        if exclusionResolved:
            return _BoundaryPanelResult(roots=(), resolved=True)

        if topologyResolved:
            refined = self._refinePanelRoots(
                evaluate,
                lower,
                upper,
                highCoordinates,
                higherRoots,
                allowUnbracketed=(
                    interpolationError <= self.config.relInterpTol * valueScale
                ),
            )
            if len(refined) == len(higherRoots):
                return _BoundaryPanelResult(roots=refined, resolved=True)

        return _BoundaryPanelResult(roots=(), resolved=False)

    def _refinePanelRoots(
        self,
        evaluate: _ScalarFunction,
        lower: float,
        upper: float,
        sampledCoordinates: np.ndarray,
        polynomialRoots: tuple[float, ...],
        *,
        allowUnbracketed: bool = True,
    ) -> tuple[_BoundaryRoot, ...]:
        """
        Refine candidates with the original callback and residual tolerance.

        Without verified interpolation accuracy, only candidates supported by
        an actual sampled sign change may be accepted. Same-sign refinement
        retains the original interpolation requirement.
        """

        refined: list[_BoundaryRoot] = []
        mappedRoots = [
            0.5 * (lower + upper) + 0.5 * (upper - lower) * normalized
            for normalized in polynomialRoots
        ]

        for candidateIndex, candidate in enumerate(mappedRoots):
            left = (
                lower
                if candidateIndex == 0
                else 0.5 * (mappedRoots[candidateIndex - 1] + candidate)
            )
            right = (
                upper
                if candidateIndex == len(mappedRoots) - 1
                else 0.5 * (candidate + mappedRoots[candidateIndex + 1])
            )
            internalSamples = [
                float(value) for value in sampledCoordinates if left < value < right
            ]
            probeCoordinates = np.asarray(
                sorted((left, *internalSamples, right)),
                dtype=float,
            )
            probeValues = np.asarray(
                [evaluate(value) for value in probeCoordinates],
                dtype=float,
            )
            sampledRootIndex = next(
                (
                    index
                    for index, value in enumerate(probeValues)
                    if abs(value) <= self.config.rootResTol
                ),
                None,
            )
            brackets = [
                (index, index + 1)
                for index in range(probeCoordinates.size - 1)
                if probeValues[index] * probeValues[index + 1] < 0.0
            ]

            # A genuine bracket does not depend on the interpolant's value
            # accuracy. Its root must still pass the original EOS residual.
            if not allowUnbracketed and not brackets:
                continue
            if sampledRootIndex is not None:
                rootCoordinate = float(probeCoordinates[sampledRootIndex])
                rootValue = float(probeValues[sampledRootIndex])
            elif brackets:
                bracketIndex = min(
                    brackets,
                    key=lambda indices: abs(
                        0.5
                        * (
                            probeCoordinates[indices[0]]
                            + probeCoordinates[indices[1]]
                        )
                        - candidate
                    ),
                )
                rootCoordinate = float(
                    solveBracket(
                        evaluate,
                        float(probeCoordinates[bracketIndex[0]]),
                        float(probeCoordinates[bracketIndex[1]]),
                        xtol=self.config.rootCoordTol,
                        maxiter=_MAX_BRACKET_ITE,
                    )
                )
                rootValue = float(evaluate(rootCoordinate))
            else:
                minimum = minimize_scalar(
                    lambda coordinate: abs(evaluate(float(coordinate))),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": self.config.rootCoordTol},
                )
                if not minimum.success or not math.isfinite(float(minimum.fun)):
                    raise RuntimeError("same-sign boundary basin refinement failed")
                rootCoordinate = float(minimum.x)
                rootValue = float(evaluate(rootCoordinate))
            if abs(rootValue) <= self.config.rootResTol:
                refined.append(
                    _BoundaryRoot(
                        coordinate=rootCoordinate,
                        residual=abs(rootValue),
                    )
                )
        return tuple(refined)


__all__ = [
    "AdaptiveBoundaryConfig",
    "AdaptiveBoundaryDiagnostic",
]
