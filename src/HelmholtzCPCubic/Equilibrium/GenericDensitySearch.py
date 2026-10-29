r"""
Find pressure roots for models without an analytic density-root formula.
Scan pressure and its density derivative, then refine stationary points and roots.
Use the model density limit and return candidates for independent root checks.
Temperature uses K; pressure and density use the model's native units.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
from scipy.optimize import minimize_scalar

from ..Numerics.BracketRoot import solveBracket
import torch

from ..Common.Model import FreeEnergyModel
from ..Kernel.Thermo import ThermoKernel
from ..Numerics.BatchedBracket import solveBatchedBrackets


MIN_DENSITY_SCALE_FACTOR = 1.0e-12
HIGH_DENSITY_FRACTION = 0.999
LOGARITHMIC_GRID_FRACTION = 0.5
GRID_TRANSITION_FRACTION = 0.135
STATIONARY_DERIVATIVE_RELATIVE_TOL = 1.0e-8
STATIONARY_COARSE_DENSITY_RELATIVE_TOL = 1.0e-6
STATIONARY_PRESSURE_PROXIMITY_RELATIVE_TOL = 1.0e-6
DENSITY_REFINEMENT_RELATIVE_TOL = 1.0e-11
ROOT_DEDUP_RELATIVE_TOL = 1.0e-9
PRESSURE_ROOT_ABSOLUTE_DENSITY_TOL = 1.0e-12  # Model density unit (mol/L for PR/SRK).
PRESSURE_ROOT_RELATIVE_DENSITY_TOL = 8.0 * np.finfo(float).eps
MAX_ROOT_ITERATIONS = 100


def findGenericDensityValues(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    scanPointCount: int,
    pressureResidualTol: float,
    *,
    pressureFunctions: tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ] | None = None,
) -> tuple[float, ...]:
    """
    Return generic Helmholtz density roots before state inspection.

    Args:
        model: Helmholtz free-energy model owning the density domain.
        kernel: Thermodynamic derivative kernel belonging to ``model``.
        temperatureTensor: Scalar temperature tensor in the model unit system.
        temperature: The same temperature as a finite positive float.
        targetPressure: Positive target pressure in the model pressure unit.
        fractions: Normalized, strictly positive mole-fraction vector.
        scanPointCount: Target number of mixed-grid density points.
        pressureResidualTol: Absolute pressure-root acceptance tolerance in the
            model pressure unit.
        pressureFunctions: Optional pressure-only and pressure/slope callbacks
            prepared for this exact model, temperature, composition and units.
            The caller may retain them for subsequent root inspection; density
            caches remain private to this search.

    Returns:
        Increasing total molar densities in the model density unit.

    Raises:
        ValueError: If model density bounds are invalid or refinement fails.
        FloatingPointError: If pressure or derivative evaluations are non-finite.
        RuntimeError: If a stationary-point or pressure-root solve fails.

    A mixed logarithmic/linear grid samples pressure and dP/drho. Refined
    stationary points split grid cells to expose tangent roots and narrow loops.
    Independent brackets share batched evaluations when supported; one bracket
    uses the scalar wrapper. All refinements use the original callbacks, and
    the caller checks final roots independently. A finite scan may miss roots.
    """

    upperDensity = float(model.densityRootUpperBound(temperature, fractions))
    if not math.isfinite(upperDensity) or upperDensity <= 0.0:
        raise ValueError("model densityRootUpperBound must be finite and positive")
    densityScale = float(model.densityScale)
    if not math.isfinite(densityScale) or densityScale <= 0.0:
        raise ValueError("model densityScale must be finite and positive")

    lowerDensity = max(
        MIN_DENSITY_SCALE_FACTOR * densityScale,
        np.finfo(float).tiny,
    )
    scanUpperDensity = HIGH_DENSITY_FRACTION * upperDensity
    if lowerDensity >= scanUpperDensity:
        raise ValueError("density-root scan bounds must be strictly increasing")
    grid = _densityScanGrid(lowerDensity, scanUpperDensity, scanPointCount)
    pressureCache: dict[float, float] = {}
    derivativeCache: dict[float, float] = {}
    supportsBatchedProperties = True
    if pressureFunctions is None:
        pressureFunctions = kernel.fixedCompositionPressureFunctions(
            temperatureTensor, fractions,
        )
    pressureFunction, pressureAndSlope = pressureFunctions

    def pressureAt(density: float) -> float:
        """
        Evaluate pressure in the native model unit at a total molar density.
        """

        cached = pressureCache.get(density)
        if cached is not None:
            return cached
        value = pressureFunction(model.asTensor(density))
        result = float(value.detach().cpu().item())
        pressureCache[density] = result
        return result

    def pressureDerivativeAt(density: float) -> float:
        """
        Evaluate the native pressure derivative at fixed temperature and composition.
        """

        cached = derivativeCache.get(density)
        if cached is not None:
            return cached
        pressure, value = pressureAndSlope(model.asTensor(density))
        pressureCache[density] = float(pressure.detach().cpu().item())
        result = float(value.detach().cpu().item())
        derivativeCache[density] = result
        return result

    def pressuresAt(densities: np.ndarray) -> np.ndarray:
        """
        Evaluate uncached pressures together when the model supports it.
        """

        densityValues = np.asarray(densities, dtype=float)
        if densityValues.ndim != 1 or densityValues.size == 0:
            raise ValueError(
                "batched pressure densities must be a nonempty vector"
            )
        missing = tuple(
            dict.fromkeys(
                float(density)
                for density in densityValues
                if float(density) not in pressureCache
            )
        )
        if missing:
            if supportsBatchedProperties:
                densityTensor = model.asTensor(np.asarray(missing))
                batch = torch.vmap(pressureFunction)(densityTensor)
                batchValues = batch.detach().cpu().numpy().astype(
                    float,
                    copy=False,
                )
                for density, pressure in zip(missing, batchValues):
                    pressureCache[density] = float(pressure)
            else:
                for density in missing:
                    pressureAt(density)
        return np.asarray(
            [pressureCache[float(density)] for density in densityValues],
            dtype=float,
        )

    def pressureDerivativesAt(densities: np.ndarray) -> np.ndarray:
        """
        Evaluate uncached density derivatives together when supported.
        """

        densityValues = np.asarray(densities, dtype=float)
        if densityValues.ndim != 1 or densityValues.size == 0:
            raise ValueError(
                "batched derivative densities must be a nonempty vector"
            )
        missing = tuple(
            dict.fromkeys(
                float(density)
                for density in densityValues
                if float(density) not in derivativeCache
            )
        )
        if missing:
            if supportsBatchedProperties:
                densityTensor = model.asTensor(np.asarray(missing))
                pressures, batch = torch.vmap(pressureAndSlope)(densityTensor)
                batchValues = batch.detach().cpu().numpy().astype(
                    float,
                    copy=False,
                )
                for density, derivative in zip(missing, batchValues):
                    derivativeCache[density] = float(derivative)
                for density, pressure in zip(missing, pressures.detach().cpu().numpy()):
                    pressureCache[density] = float(pressure)
            else:
                for density in missing:
                    pressureDerivativeAt(density)
        return np.asarray(
            [derivativeCache[float(density)] for density in densityValues],
            dtype=float,
        )

    gridTensor = model.asTensor(grid)
    try:
        scanPressures, scanDerivatives = torch.vmap(pressureAndSlope)(gridTensor)
    except (NotImplementedError, RuntimeError):

        # Some third-party models cannot be transformed by vmap.  The scalar
        # path preserves the same grid, stationary rules, and failure checks.
        supportsBatchedProperties = False
        scanPressures = model.asTensor(
            [pressureAt(float(density)) for density in grid]
        )
        scanDerivatives = model.asTensor(
            [pressureDerivativeAt(float(density)) for density in grid]
        )
    else:
        for density, pressureValue, derivativeValue in zip(
            grid,
            scanPressures,
            scanDerivatives,
        ):
            densityValue = float(density)
            pressureCache[densityValue] = float(
                pressureValue.detach().cpu().item()
            )
            derivativeCache[densityValue] = float(
                derivativeValue.detach().cpu().item()
            )

    pressures = scanPressures.detach().cpu().numpy().astype(float, copy=False)
    derivatives = scanDerivatives.detach().cpu().numpy().astype(float, copy=False)
    if not np.all(np.isfinite(pressures)):
        nInvalid = int(np.count_nonzero(~np.isfinite(pressures)))
        raise FloatingPointError(
            f"pressure scan returned {nInvalid} non-finite values"
        )
    if not np.all(np.isfinite(derivatives)):
        nInvalid = int(np.count_nonzero(~np.isfinite(derivatives)))
        raise FloatingPointError(
            f"pressure-derivative scan returned {nInvalid} non-finite values"
        )

    stationaryDensities = _findPressureStationaryDensities(
        grid,
        derivatives,
        pressureDerivativeAt,
        pressureDerivativesAt,
        pressureAt,
        targetPressure,
        pressureResidualTol,
    )
    nodes = _deduplicateDensityValues(
        [*(float(value) for value in grid), *stationaryDensities],
        DENSITY_REFINEMENT_RELATIVE_TOL,
    )
    residuals = [pressureAt(density) - targetPressure for density in nodes]
    if not all(math.isfinite(residual) for residual in residuals):
        nInvalid = sum(not math.isfinite(residual) for residual in residuals)
        raise FloatingPointError(
            f"pressure node evaluation returned {nInvalid} non-finite residuals"
        )

    rootDensities: list[float] = []
    bracketLeft: list[float] = []
    bracketRight: list[float] = []
    for left, right, leftResidual, rightResidual in zip(
        nodes[:-1],
        nodes[1:],
        residuals[:-1],
        residuals[1:],
    ):
        leftIsRoot = abs(leftResidual) <= pressureResidualTol
        rightIsRoot = abs(rightResidual) <= pressureResidualTol
        if leftIsRoot:
            rootDensities.append(left)
        if rightIsRoot:
            rootDensities.append(right)
        if leftIsRoot or rightIsRoot:
            continue
        if leftResidual * rightResidual < 0.0:
            bracketLeft.append(left)
            bracketRight.append(right)

    if bracketLeft:
        rootDensities.extend(
            _refinePressureBrackets(
                pressureAt,
                pressuresAt,
                np.asarray(bracketLeft),
                np.asarray(bracketRight),
                targetPressure,
                supportsBatchedProperties,
            )
        )

    return tuple(
        _deduplicateDensityValues(
            rootDensities,
            ROOT_DEDUP_RELATIVE_TOL,
        )
    )


def _refinePressureBrackets(
    pressureAt: Callable[[float], float],
    pressuresAt: Callable[[np.ndarray], np.ndarray],
    leftDensities: np.ndarray,
    rightDensities: np.ndarray,
    targetPressure: float,
    supportsBatchedProperties: bool,
) -> list[float]:
    """
    Refine final pressure roots without changing their bracket order.
    """

    if leftDensities.shape != rightDensities.shape or leftDensities.size == 0:
        raise ValueError("pressure-root brackets must be equal nonempty vectors")

    if leftDensities.size >= 2 and supportsBatchedProperties:
        def pressureResiduals(densities: np.ndarray) -> np.ndarray:
            """
            Return vector pressure residuals in the model pressure unit.
            """

            return pressuresAt(densities) - targetPressure

        try:
            densities = solveBatchedBrackets(
                pressureResiduals,
                leftDensities,
                rightDensities,
                relativeTolerance=PRESSURE_ROOT_RELATIVE_DENSITY_TOL,
                absoluteTolerance=PRESSURE_ROOT_ABSOLUTE_DENSITY_TOL,
                maxIterations=MAX_ROOT_ITERATIONS,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "batched pressure-root refinement failed for "
                f"{leftDensities.size} brackets"
            ) from error
        return [float(density) for density in densities]

    densities: list[float] = []
    for leftDensity, rightDensity in zip(leftDensities, rightDensities):
        try:
            density = solveBracket(
                lambda value: pressureAt(value) - targetPressure,
                float(leftDensity),
                float(rightDensity),
                xtol=PRESSURE_ROOT_ABSOLUTE_DENSITY_TOL,
                rtol=PRESSURE_ROOT_RELATIVE_DENSITY_TOL,
                maxiter=MAX_ROOT_ITERATIONS,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                "pressure-root refinement failed in bracket "
                f"[{leftDensity:.9g}, {rightDensity:.9g}]"
            ) from error
        densities.append(float(density))
    return densities


def _densityScanGrid(
    lowerDensity: float,
    upperDensity: float,
    pointCount: int,
) -> np.ndarray:
    """
    Combine dilute logarithmic coverage with dense-region resolution.
    """

    transitionDensity = max(
        lowerDensity * (1.0 + 16.0 * np.finfo(float).eps),
        GRID_TRANSITION_FRACTION * upperDensity,
    )
    transitionDensity = min(transitionDensity, upperDensity)
    logarithmicCount = max(
        2,
        int(round(LOGARITHMIC_GRID_FRACTION * pointCount)),
    )
    linearCount = max(2, pointCount - logarithmicCount + 1)
    diluteGrid = np.geomspace(
        lowerDensity,
        transitionDensity,
        logarithmicCount,
    )
    denseGrid = np.linspace(
        transitionDensity,
        upperDensity,
        linearCount,
    )
    return np.unique(np.concatenate((diluteGrid, denseGrid)))


def _findPressureStationaryDensities(
    grid: np.ndarray,
    derivatives: np.ndarray,
    derivativeAt: Callable[[float], float],
    derivativesAt: Callable[[np.ndarray], np.ndarray],
    pressureAt: Callable[[float], float],
    targetPressure: float,
    pressureResidualTol: float,
) -> list[float]:
    """
    Locate sign-changing, tangential, and hidden zeros of ``dP/drho``.
    """

    stationaryDensities: list[float] = []
    bracketLeft: list[float] = []
    bracketRight: list[float] = []
    for left, right, leftDerivative, rightDerivative in zip(
        grid[:-1],
        grid[1:],
        derivatives[:-1],
        derivatives[1:],
    ):
        if leftDerivative == 0.0:
            stationaryDensities.append(float(left))
        if leftDerivative * rightDerivative < 0.0:
            bracketLeft.append(float(left))
            bracketRight.append(float(right))
    if derivatives[-1] == 0.0:
        stationaryDensities.append(float(grid[-1]))
    if bracketLeft:
        stationaryDensities.extend(
            _refineStationaryBracketsForPressure(
                derivativeAt,
                derivativesAt,
                pressureAt,
                np.asarray(bracketLeft),
                np.asarray(bracketRight),
                targetPressure,
                pressureResidualTol,
            )
        )

    # A narrow opposite-sign derivative pocket, or a derivative that merely
    # touches zero, can lie wholly inside one coarse grid cell.  A strict
    # sampled basin is refined before deciding whether extra brackets exist.
    for index in range(1, len(grid) - 1):
        localDerivatives = derivatives[index - 1 : index + 2]
        if (
            localDerivatives[0] * localDerivatives[1] <= 0.0
            or localDerivatives[1] * localDerivatives[2] <= 0.0
        ):
            continue
        middleMagnitude = abs(localDerivatives[1])
        neighbourMagnitude = min(
            abs(localDerivatives[0]),
            abs(localDerivatives[2]),
        )
        basinScale = max(1.0, middleMagnitude, neighbourMagnitude)
        if (
            middleMagnitude >= neighbourMagnitude
            or neighbourMagnitude - middleMagnitude <= 1.0e-10 * basinScale
        ):
            continue

        left = float(grid[index - 1])
        right = float(grid[index + 1])
        localSign = 1.0 if localDerivatives[1] > 0.0 else -1.0
        optimumDensity = _refineStationaryBasin(
            derivativeAt,
            left,
            float(grid[index]),
            right,
            localSign,
            STATIONARY_COARSE_DENSITY_RELATIVE_TOL
            * max(1.0, float(grid[index])),
        )
        optimumDerivative = derivativeAt(optimumDensity)
        if not math.isfinite(optimumDerivative):
            raise FloatingPointError(
                "pressure-stationary refinement returned a non-finite derivative"
            )
        derivativeScale = max(
            1.0,
            *(abs(float(value)) for value in localDerivatives),
            abs(targetPressure) / max(1.0, abs(optimumDensity)),
        )
        derivativeTolerance = (
            STATIONARY_DERIVATIVE_RELATIVE_TOL * derivativeScale
        )
        if _stationaryPressureIsNearTarget(
            pressureAt,
            optimumDensity,
            targetPressure,
            pressureResidualTol,
        ):
            optimumDensity = _refineStationaryBasin(
                derivativeAt,
                left,
                float(grid[index]),
                right,
                localSign,
                DENSITY_REFINEMENT_RELATIVE_TOL
                * max(1.0, float(grid[index])),
            )
            optimumDerivative = derivativeAt(optimumDensity)
            if not math.isfinite(optimumDerivative):
                raise FloatingPointError(
                    "pressure-stationary refinement returned a non-finite "
                    "derivative"
                )
        if abs(optimumDerivative) <= derivativeTolerance:
            stationaryDensities.append(optimumDensity)
            continue

        for bracketLeft, bracketRight, endpointDerivative in (
            (left, optimumDensity, float(localDerivatives[0])),
            (optimumDensity, right, float(localDerivatives[2])),
        ):
            if endpointDerivative * optimumDerivative < 0.0:
                stationaryDensities.append(
                    _refineStationaryDensityForPressure(
                        derivativeAt,
                        pressureAt,
                        bracketLeft,
                        bracketRight,
                        targetPressure,
                        pressureResidualTol,
                    )
                )

    return _deduplicateDensityValues(
        stationaryDensities,
        DENSITY_REFINEMENT_RELATIVE_TOL,
    )


def _refineStationaryDensity(
    derivativeAt: Callable[[float], float],
    leftDensity: float,
    rightDensity: float,
    relativeTolerance: float = DENSITY_REFINEMENT_RELATIVE_TOL,
) -> float:
    """
    Refine a sign-changing dP/drho bracket with the Chandrupatla solver.
    """

    try:
        return float(
            solveBracket(
                derivativeAt,
                leftDensity,
                rightDensity,
                xtol=np.finfo(float).eps * max(1.0, abs(leftDensity)),
                rtol=relativeTolerance,
                maxiter=MAX_ROOT_ITERATIONS,
            )
        )
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            "pressure-stationary refinement failed in bracket "
            f"[{leftDensity:.9g}, {rightDensity:.9g}]"
        ) from error


def _refineStationaryDensityForPressure(
    derivativeAt: Callable[[float], float],
    pressureAt: Callable[[float], float],
    leftDensity: float,
    rightDensity: float,
    targetPressure: float,
    pressureResidualTol: float,
) -> float:
    """
    Refine a stationary bracket only as tightly as the target pressure needs.
    """

    density = _refineStationaryDensity(
        derivativeAt,
        leftDensity,
        rightDensity,
        STATIONARY_COARSE_DENSITY_RELATIVE_TOL,
    )
    if _stationaryPressureIsNearTarget(
        pressureAt,
        density,
        targetPressure,
        pressureResidualTol,
    ):
        return _refineStationaryDensity(
            derivativeAt,
            leftDensity,
            rightDensity,
            DENSITY_REFINEMENT_RELATIVE_TOL,
        )
    return density


def _refineStationaryBracketsForPressure(
    derivativeAt: Callable[[float], float],
    derivativesAt: Callable[[np.ndarray], np.ndarray],
    pressureAt: Callable[[float], float],
    leftDensities: np.ndarray,
    rightDensities: np.ndarray,
    targetPressure: float,
    pressureResidualTol: float,
) -> list[float]:
    """
    Refine simultaneous sign-change brackets with shared batch evaluations.
    """

    if leftDensities.size == 1:
        return [
            _refineStationaryDensityForPressure(
                derivativeAt,
                pressureAt,
                float(leftDensities[0]),
                float(rightDensities[0]),
                targetPressure,
                pressureResidualTol,
            )
        ]
    densities = solveBatchedBrackets(
        derivativesAt,
        leftDensities,
        rightDensities,
        relativeTolerance=STATIONARY_COARSE_DENSITY_RELATIVE_TOL,
        absoluteTolerance=np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(leftDensities)))),
        maxIterations=MAX_ROOT_ITERATIONS,
    )
    tightIndices = [
        index
        for index, density in enumerate(densities)
        if _stationaryPressureIsNearTarget(
            pressureAt,
            float(density),
            targetPressure,
            pressureResidualTol,
        )
    ]
    if tightIndices:
        tightLeft = leftDensities[tightIndices]
        tightRight = rightDensities[tightIndices]
        if len(tightIndices) == 1:
            index = tightIndices[0]
            densities[index] = _refineStationaryDensity(
                derivativeAt,
                float(tightLeft[0]),
                float(tightRight[0]),
                DENSITY_REFINEMENT_RELATIVE_TOL,
            )
        else:
            densities[tightIndices] = solveBatchedBrackets(
                derivativesAt,
                tightLeft,
                tightRight,
                relativeTolerance=DENSITY_REFINEMENT_RELATIVE_TOL,
                absoluteTolerance=np.finfo(float).eps
                * max(1.0, float(np.max(np.abs(tightLeft)))),
                maxIterations=MAX_ROOT_ITERATIONS,
            )
    return [float(density) for density in densities]


def _stationaryPressureIsNearTarget(
    pressureAt: Callable[[float], float],
    density: float,
    targetPressure: float,
    pressureResidualTol: float,
) -> bool:
    """
    Return whether a stationary pressure needs final-density precision.
    """

    pressure = pressureAt(density)
    if not math.isfinite(pressure):
        raise FloatingPointError(
            "pressure-stationary refinement returned a non-finite pressure"
        )
    tolerance = max(
        pressureResidualTol,
        STATIONARY_PRESSURE_PROXIMITY_RELATIVE_TOL
        * max(1.0, abs(targetPressure)),
    )
    return abs(pressure - targetPressure) <= tolerance


def _refineStationaryBasin(
    derivativeAt: Callable[[float], float],
    leftDensity: float,
    middleDensity: float,
    rightDensity: float,
    localSign: float,
    absoluteTolerance: float,
) -> float:
    """
    Refine one sampled derivative basin from its strict three-point bracket.
    """

    def basinObjective(density: float) -> float:
        """
        Evaluate only inside the sampled bounded basin.
        """

        densityValue = float(density)
        if not leftDensity <= densityValue <= rightDensity:
            raise ValueError("Scalar trial left its pressure-stationary bracket")
        return localSign * derivativeAt(densityValue)

    try:
        optimum = minimize_scalar(
            basinObjective,
            bounds=(leftDensity, rightDensity),
            method="bounded",
            options={
                "xatol": absoluteTolerance,
                "maxiter": MAX_ROOT_ITERATIONS,
            },
        )
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            "pressure-stationary basin refinement failed in bracket "
            f"[{leftDensity:.9g}, {rightDensity:.9g}]"
        ) from error
    optimumDensity = float(optimum.x)
    if (
        not optimum.success
        or not math.isfinite(float(optimum.fun))
        or not leftDensity <= optimumDensity <= rightDensity
    ):
        raise RuntimeError(
            "pressure-stationary basin refinement failed in bracket "
            f"[{leftDensity:.9g}, {rightDensity:.9g}]"
        )
    return optimumDensity


def _deduplicateDensityValues(
    densities: list[float],
    relativeTolerance: float,
) -> list[float]:
    """
    Return sorted density values unique under a relative tolerance.
    """

    uniqueDensities: list[float] = []
    for density in sorted(densities):
        if uniqueDensities and abs(density - uniqueDensities[-1]) <= (
            relativeTolerance
            * max(1.0, abs(density), abs(uniqueDensities[-1]))
        ):
            continue
        uniqueDensities.append(density)
    return uniqueDensities


__all__ = ["findGenericDensityValues"]
