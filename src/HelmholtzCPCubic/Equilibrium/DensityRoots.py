r"""
Find density roots at fixed temperature, pressure and composition.
Use the PR/SRK cubic equation or a pressure-and-slope scan for other models.
Check each root independently and retain its stability and fugacity data.
Inputs and roots use K, bar and mol/L.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..Common.Model import FreeEnergyModel, TensorLike
from ..Kernel.Thermo import ThermoKernel
from ..Transform.CubicBell import CubicBellModel, cubicConstants
from .GenericDensitySearch import findGenericDensityValues

DEFAULT_SCAN_POINT_COUNT = 48
PRESSURE_RESIDUAL_TOL = 1.0e-8  # bar; shared by root acceptance and molar TPD.


@dataclass(frozen=True)
class DensityRoot:
    """
    One pressure-matching root with its mechanical-stability result.

    Density is in mol/L, pressure and its residual are in bar, and
    ``pressureDensityDerivative`` is the fixed-composition dP/drho in
    bar L/mol. A root is mechanically stable exactly when this derivative is
    positive.
    """

    density: float
    pressure: float
    pressureResidual: float
    pressureDensityDerivative: float
    lnphi: tuple[float, ...]
    gibbsLike: float

    @property
    def mechanicalStable(self) -> bool:
        """
        Return whether fixed-composition dP/drho is positive.
        """

        return self.pressureDensityDerivative > 0.0

    def asDict(self) -> dict[str, object]:
        """
        Return a JSON-ready record of this root.
        """

        return {
            "densityMolL": self.density,
            "pressureBar": self.pressure,
            "pressureResidualBar": self.pressureResidual,
            "pressureDensityDerivativeBarLPerMol": (
                self.pressureDensityDerivative
            ),
            "mechanicalStable": self.mechanicalStable,
            "lnphi": list(self.lnphi),
            "sumYlnphi": self.gibbsLike,
        }


@dataclass(frozen=True)
class _DensityRootEvaluation:
    """
    Qualified root and chemical potentials, with an optional full Hessian.

    TPD screening qualifies the scalar mechanical slope first and defers the
    matrix until a selected branch needs an optimization step. Public density
    queries retain full-state inspection by default.
    """

    root: DensityRoot
    chemicalPotentials: np.ndarray
    hessian: np.ndarray | None


@dataclass(frozen=True)
class DensityRootFailure:
    """
    One rejected density candidate (mol/L) and its numerical cause.
    """

    density: float
    reason: str

    def asDict(self) -> dict[str, object]:
        """
        Return the rejected candidate without presenting it as a root.
        """

        return {"candidateDensityMolL": self.density, "reason": self.reason}


class DensityRootEvaluationError(FloatingPointError):
    """
    Incomplete root qualification, retaining valid roots and all failures.

    ``roots`` can supply feasible trial states, but cannot be treated as a
    complete root set for minimum-Gibbs phase selection. ``evaluations``
    retains their already computed derivatives for the internal TPD caller.
    """

    def __init__(
        self,
        evaluations: tuple[_DensityRootEvaluation, ...],
        failures: tuple[DensityRootFailure, ...],
    ) -> None:
        """
        Retain qualified roots in density order and rejected candidates.
        """

        self.evaluations = evaluations
        self.roots = tuple(item.root for item in evaluations)
        self.failures = failures
        super().__init__(
            f"density-root qualification incomplete: {len(evaluations)} valid, "
            f"{len(failures)} rejected; " + "; ".join(item.reason for item in failures)
        )


def findDensityRoots(
    model: FreeEnergyModel,
    temperature: TensorLike,
    pressure: TensorLike,
    composition: TensorLike,
    *,
    scanPointCount: int = DEFAULT_SCAN_POINT_COUNT,
) -> tuple[DensityRoot, ...]:
    """
    Return qualified density roots ordered by increasing density.

    Args:
        model: Free-energy-density model using K, bar and mol/L.
        temperature: Scalar temperature (K).
        pressure: Scalar target pressure in bar.
        composition: Strictly positive mole fractions.
        scanPointCount: Number of generic logarithmic density-scan points.
            Cubic PR/SRK models use their analytic polynomial, but the value
            is still validated to keep the public input contract uniform.

    This entry requires a K/bar/mol/L model and performs no implicit unit
    conversion. Adapt a native SI model with ``asWorkspaceModel`` before use.

    Returns:
        Density roots sorted by density; empty when no admissible root is found.

    Raises:
        ValueError: If the state arguments are invalid or the composition is
            not strictly positive.
        DensityRootEvaluationError: If any candidate fails qualification;
            its ``roots`` and ``failures`` retain the partial result.

    PR/SRK roots come from the compressibility cubic. Other models use a joint
    pressure/slope scan bounded by densityRootUpperBound. Stationary points
    split scan intervals before refinement, including possible tangent roots.
    Root checks retain mechanical stability and fugacity coefficients. Batched
    full-state evaluation is used only when the model supports it.
    """

    return _findDensityRoots(
        model,
        temperature,
        pressure,
        composition,
        scanPointCount=scanPointCount,
        kernel=ThermoKernel(model),
    )


def _findDensityRoots(
    model: FreeEnergyModel,
    temperature: TensorLike,
    pressure: TensorLike,
    composition: TensorLike,
    *,
    scanPointCount: int,
    kernel: ThermoKernel,
) -> tuple[DensityRoot, ...]:
    """
    Implement density-root discovery with a caller-owned thermo kernel.
    """

    evaluations = _findDensityRootEvaluations(
        model,
        temperature,
        pressure,
        composition,
        scanPointCount=scanPointCount,
        kernel=kernel,
    )
    return tuple(evaluation.root for evaluation in evaluations)


def _findDensityRootEvaluations(
    model: FreeEnergyModel,
    temperature: TensorLike,
    pressure: TensorLike,
    composition: TensorLike,
    *,
    scanPointCount: int,
    kernel: ThermoKernel,
    withHessian: bool = True,
) -> tuple[_DensityRootEvaluation, ...]:
    """
    Find roots with either full or deferred component-density Hessians.

    ``withHessian=False`` retains chemical potentials and a scalar mechanical
    slope for TPD screening. Domain and pressure tolerances are unchanged.
    Pure fluids retain their scalar Hessian and tangent-root rounding.
    """

    if kernel.model is not model:
        raise ValueError("reused ThermoKernel must belong to the supplied model")
    temperatureTensor = model.asTensor(temperature)
    pressureTensor = model.asTensor(pressure)
    fractions = model.asTensor(composition)
    if (
        not isinstance(scanPointCount, int)
        or isinstance(scanPointCount, bool)
        or scanPointCount < 3
    ):
        raise ValueError("scanPointCount must be an integer of at least three")
    if temperatureTensor.ndim != 0 or pressureTensor.ndim != 0:
        raise ValueError("temperature and pressure must be scalars")
    temperatureValue = float(temperatureTensor.detach().cpu())
    pressureValue = float(pressureTensor.detach().cpu())
    if not math.isfinite(temperatureValue) or temperatureValue <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pressureValue) or pressureValue <= 0.0:
        raise ValueError("pressure must be finite and positive")
    if fractions.shape != (model.componentCount,):
        raise ValueError(
            f"composition must have shape ({model.componentCount},)"
        )
    if not bool(torch.all(torch.isfinite(fractions)).detach().cpu().item()):
        raise ValueError("composition must be finite")
    if bool(torch.any(fractions.detach() <= 0.0).cpu().item()):
        raise ValueError("all active-component mole fractions must be positive")
    compositionTotal = torch.sum(fractions)
    if not bool(torch.isfinite(compositionTotal).detach().cpu().item()):
        raise ValueError("composition sum must be finite")
    fractions = fractions / compositionTotal

    temperatureK = temperatureValue
    targetPressure = pressureValue
    fractionsNumpy = fractions.detach().cpu().numpy()
    if isinstance(model, CubicBellModel):
        return _findCubicDensityRootEvaluations(
            model,
            kernel,
            temperatureTensor,
            temperatureK,
            targetPressure,
            fractionsNumpy,
            withHessian=withHessian,
        )

    return _findGenericDensityRootEvaluations(
        model,
        kernel,
        temperatureTensor,
        temperatureK,
        targetPressure,
        fractionsNumpy,
        scanPointCount,
        withHessian=withHessian,
    )


def _findCubicDensityRootEvaluations(
    model: CubicBellModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    *,
    withHessian: bool = True,
) -> tuple[_DensityRootEvaluation, ...]:
    """
    Solve the PR/SRK compressibility cubic and inspect physical roots.
    """

    attraction, coVolume = model.mixtureParameters(temperatureTensor, fractions)
    gasTemperature = model.gasConstant * temperature
    reducedAttraction = (
        float(attraction.detach().cpu().item())
        * targetPressure
        / gasTemperature**2
    )
    reducedCoVolume = (
        float(coVolume.detach().cpu().item())
        * targetPressure
        / gasTemperature
    )
    _omegaA, _omegaB, delta1, delta2 = cubicConstants(model.mode)
    deltaSum = delta1 + delta2
    deltaProduct = delta1 * delta2
    coefficients = (
        1.0,
        (deltaSum - 1.0) * reducedCoVolume - 1.0,
        reducedAttraction
        + deltaProduct * reducedCoVolume**2
        - deltaSum * reducedCoVolume * (reducedCoVolume + 1.0),
        -(
            reducedAttraction * reducedCoVolume
            + deltaProduct
            * reducedCoVolume**2
            * (reducedCoVolume + 1.0)
        ),
    )
    polynomialRoots = np.roots(coefficients)
    imaginaryTolerance = 1.0e-8
    compressibilityRoots = sorted(
        float(root.real)
        for root in polynomialRoots
        if abs(root.imag) <= imaginaryTolerance
        and root.real > reducedCoVolume + 1.0e-12
    )
    densities = sorted(
        targetPressure / (compressibility * gasTemperature)
        for compressibility in compressibilityRoots
    )
    uniqueDensities: list[float] = []
    for density in densities:
        if uniqueDensities and abs(density - uniqueDensities[-1]) <= (
            1.0e-9 * max(1.0, density, uniqueDensities[-1])
        ):
            continue
        uniqueDensities.append(density)
    return _rootEvaluationsAt(
        model,
        kernel,
        temperatureTensor,
        temperature,
        targetPressure,
        fractions,
        uniqueDensities,
        withHessian=withHessian,
    )


def _findGenericDensityRootEvaluations(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    scanPointCount: int,
    *,
    withHessian: bool = True,
) -> tuple[_DensityRootEvaluation, ...]:
    """
    Find generic roots by joint pressure/derivative scanning and refinement.
    """

    # Keep fixed-T/y preparation local to one query, including root inspection.
    # Share functions, not cached scan values: roots still have independent
    # component-pressure acceptance and their original scalar/batch slope path.
    pressureFunctions = kernel.fixedCompositionPressureFunctions(
        temperatureTensor, fractions,
    )
    rootDensities = findGenericDensityValues(
        model,
        kernel,
        temperatureTensor,
        temperature,
        targetPressure,
        fractions,
        scanPointCount,
        PRESSURE_RESIDUAL_TOL,
        pressureFunctions=pressureFunctions,
    )
    return _rootEvaluationsAt(
        model,
        kernel,
        temperatureTensor,
        temperature,
        targetPressure,
        fractions,
        rootDensities,
        withHessian=withHessian,
        pressureAndSlope=pressureFunctions[1],
    )


def _rootAt(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    density: float,
    *,
    withHessian: bool = True,
) -> DensityRoot:
    """
    Evaluate one density root: pressure, stability and ln(phi).
    """

    return _rootEvaluationAt(
        model,
        kernel,
        temperatureTensor,
        temperature,
        targetPressure,
        fractions,
        density,
        withHessian=withHessian,
    ).root


def _rootEvaluationAt(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    density: float,
    *,
    withHessian: bool = True,
    pressureAndSlope: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] | None = None,
) -> _DensityRootEvaluation:
    """
    Inspect one root, optionally reusing this query's fixed-T/y slope callback.

    The callback uses the model's native units and must match this temperature
    and composition. It is called only after independent pressure acceptance.
    """

    densities = model.asTensor(density * fractions)
    if not withHessian and model.componentCount > 1:
        model.validateState(temperatureTensor, densities)
        chemicalPotentials, pressure = kernel.chemicalPotentialsAndPressure(
            temperatureTensor, densities, validate=False,
        )

        # Reject a bad pressure candidate before even requesting its slope.
        _validateRootPressure(model, pressure, targetPressure, density)
        if pressureAndSlope is None:
            _, pressureAndSlope = kernel.fixedCompositionPressureFunctions(
                temperatureTensor, fractions,
            )
        _, slope = pressureAndSlope(model.asTensor(density))
        return _rootEvaluationFromState(
            model, temperature, targetPressure, fractions, density, densities,
            pressure, chemicalPotentials, None, float(slope.detach().cpu()),
        )

    # For a pure fluid the Hessian is scalar already. Preserve its operation
    # order and exact tangent-root sign instead of changing rounding for no
    # dimension reduction.
    state = kernel.state(
        temperatureTensor,
        densities,
        validate=False,
    )
    return _rootEvaluationFromState(
        model,
        temperature,
        targetPressure,
        fractions,
        density,
        densities,
        state.pressure,
        state.chemicalPotentials,
        state.hessian,
    )


def _rootEvaluationsAt(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperatureTensor: torch.Tensor,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    densities: tuple[float, ...] | list[float],
    *,
    withHessian: bool = True,
    pressureAndSlope: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] | None = None,
) -> tuple[_DensityRootEvaluation, ...]:
    """
    Qualify every candidate, retaining valid roots if another one fails.

    Full-state inspection is the default. Deferred inspection batches first
    derivatives and scalar slopes when the model declares batched AD support;
    declared batch failures are reported without retrying scalar inspection.
    A supplied slope callback must belong to this model and fixed T/y in the
    same units. Full-Hessian and pure-fluid inspection do not use it.
    """

    densityValues = tuple(float(density) for density in densities)
    evaluations: list[_DensityRootEvaluation] = []
    failures: list[DensityRootFailure] = []
    if len(densityValues) < 2 or not model.supportsBatchedThermoState:
        for density in densityValues:
            try:
                evaluations.append(_rootEvaluationAt(
                    model, kernel, temperatureTensor, temperature,
                    targetPressure, fractions, density,
                    withHessian=withHessian,
                    pressureAndSlope=pressureAndSlope,
                ))
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failures.append(DensityRootFailure(density, f"{type(error).__name__}: {error}"))
        if failures:
            raise DensityRootEvaluationError(tuple(evaluations), tuple(failures))
        return tuple(evaluations)

    componentDensities = model.asTensor(
        np.asarray(densityValues)[:, np.newaxis] * fractions
    )
    try:
        if withHessian or model.componentCount == 1:
            states = kernel.stateBatch(
                temperatureTensor, componentDensities, validate=False,
            )
            pressures, chemicalPotentials = states.pressure, states.chemicalPotentials
            hessians, slopes = states.hessian, None
        else:
            def firstOrderValues(localDensities: torch.Tensor):
                """
                Batch only the first derivatives needed to compare roots.
                """

                return kernel.chemicalPotentialsAndPressure(
                    temperatureTensor, localDensities, validate=False,
                )

            chemicalPotentials, pressures = torch.vmap(firstOrderValues)(componentDensities)
            if pressureAndSlope is None:
                _, pressureAndSlope = kernel.fixedCompositionPressureFunctions(
                    temperatureTensor, fractions,
                )
            _, slopes = torch.vmap(pressureAndSlope)(model.asTensor(densityValues))
            hessians = None
    except (
        FloatingPointError,
        NotImplementedError,
        RuntimeError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "batched density-root state evaluation failed for "
            f"{len(densityValues)} roots"
        ) from error
    for index, density in enumerate(densityValues):
        try:
            evaluations.append(_rootEvaluationFromState(
                model, temperature, targetPressure, fractions, density,
                componentDensities[index], pressures[index],
                chemicalPotentials[index], None if hessians is None else hessians[index],
                None if slopes is None else float(slopes[index].detach().cpu()),
            ))
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(DensityRootFailure(density, f"{type(error).__name__}: {error}"))
    if failures:
        raise DensityRootEvaluationError(tuple(evaluations), tuple(failures))
    return tuple(evaluations)


def _validateRootPressure(
    model: FreeEnergyModel,
    pressureTensor: torch.Tensor,
    targetPressure: float,
    density: float,
) -> tuple[float, float]:
    """
    Check positive finite pressure and the shared 1e-8 bar residual limit.
    """

    pressure = float(pressureTensor.detach().cpu().item())
    model.validatePres(pressure)
    pressureResidual = abs(pressure - targetPressure)
    if not math.isfinite(pressureResidual) or pressureResidual > PRESSURE_RESIDUAL_TOL:
        raise FloatingPointError(
            "refined density root exceeds pressure residual tolerance: "
            f"density={density:.9g} mol/L, residual={pressureResidual:.3e} bar"
        )
    return pressure, pressureResidual


def _rootEvaluationFromState(
    model: FreeEnergyModel,
    temperature: float,
    targetPressure: float,
    fractions: np.ndarray,
    density: float,
    densities: torch.Tensor,
    pressureTensor: torch.Tensor,
    chemicalPotentialTensor: torch.Tensor,
    hessianTensor: torch.Tensor | None,
    pressureDensityDerivative: float | None = None,
) -> _DensityRootEvaluation:
    """
    Build one density-root record from an already evaluated local state.
    """

    model.validateState(temperature, densities)
    pressure, pressureResidual = _validateRootPressure(
        model, pressureTensor, targetPressure, density,
    )
    totalDensity = float(torch.sum(densities).detach().cpu().item())
    molarVolume = 1.0 / totalDensity
    compressibility = pressure * molarVolume / model.gasConstant / temperature

    chemicalPotentials = chemicalPotentialTensor.detach().cpu().numpy()
    hessian = None if hessianTensor is None else hessianTensor.detach().cpu().numpy()
    if not np.all(np.isfinite(chemicalPotentials)) or (
        hessian is not None and not np.all(np.isfinite(hessian))
    ):
        raise FloatingPointError("density-root derivatives contain NaN or Inf")

    # dP/d(rho)|_{T,y} = rho * y^T H y  (fixed-composition mechanical stability)
    if hessian is None:
        if pressureDensityDerivative is None:
            raise ValueError("a root without Hessian requires its mechanical slope")
        stability = pressureDensityDerivative
    else:
        stability = totalDensity * float(fractions @ (hessian @ fractions))

    lnphi = tuple(
        float(
            (
                chemicalPotentials[index]
                - model.gasConstant
                * temperature
                * math.log(density * fractions[index])
            )
            / (model.gasConstant * temperature)
            - math.log(compressibility)
        )
        for index in range(model.componentCount)
    )
    gibbsLike = float(np.dot(fractions, lnphi))
    if not all(math.isfinite(value) for value in (*lnphi, gibbsLike, stability)):
        raise FloatingPointError("density-root thermodynamic record is nonfinite")
    return _DensityRootEvaluation(
        root=DensityRoot(
            density=float(density),
            pressure=pressure,
            pressureResidual=pressureResidual,
            pressureDensityDerivative=stability,
            lnphi=lnphi,
            gibbsLike=gibbsLike,
        ),
        chemicalPotentials=chemicalPotentials,
        hessian=hessian,
    )


__all__ = ["DensityRoot", "DensityRootFailure", "DensityRootEvaluationError", "findDensityRoots"]
