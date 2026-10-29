r"""
Minimize extended molar TPD jointly over composition and density.
Use a trust-region solver with all qualified stable density roots as starts.
Apply the instability threshold to pressure-qualified original TPD values.
Inputs use K, bar and mol/L; the objective and TPD are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..Common.Model import FreeEnergyModel, TensorLike
from ..Kernel.Thermo import ThermoKernel
from ..Numerics.ScalarTrustRegion import (
    ScalarTrustRegionConfig, _curvatureTolerance, minimizeScalar,
)
from ..Transform.CubicBell import CubicBellModel
from .DensityRoots import (
    DEFAULT_SCAN_POINT_COUNT, PRESSURE_RESIDUAL_TOL,
    DensityRootEvaluationError, _DensityRootEvaluation,
    _findDensityRootEvaluations, _rootEvaluationFromState,
)
from .GenericDensitySearch import (
    MIN_DENSITY_SCALE_FACTOR, HIGH_DENSITY_FRACTION, ROOT_DEDUP_RELATIVE_TOL,
)
from .TPD import (
    TPDResult, TPDStartFailure, _normalizeFeed, _normalizeStart,
    _referenceState, defaultTpdStarts,
)
from .TPDMinimization import (
    TPDMinimum, TPDDensityFailure, _TPDEvidence, _TPDWitnessFound,
    _ILR_COORDINATE_BOUND, _compositionFromIlr, _helmertBasis,
    _ilrCoordinates, _ilrProjected, _trialDerivativesFromEvaluation,
)


@dataclass
class _MolarEvaluation:
    """
    One joint TPD evaluation with density [mol/L] and dimensionless J and TPD.
    value stores J; tpdValue stores the original TPD expression. The latter is
    used for classification only after pressure acceptance. root is set only
    after independent root checks; compositionHessian uses the fixed-P map.
    """

    fractions: np.ndarray
    density: float
    value: float
    tpdValue: float
    gradient: np.ndarray
    hessian: np.ndarray
    root: _DensityRootEvaluation | None
    compositionHessian: np.ndarray | None


def _molarDerivatives(
    model: FreeEnergyModel, kernel: ThermoKernel, temperature: float,
    pressure: float, constant: np.ndarray, basis: np.ndarray,
    coordinates: np.ndarray,
) -> _MolarEvaluation:
    """
    Evaluate the joint objective and derivatives in (ILR, log(r/densityScale)).
    J = (Psi + P0)/(RT*r) - y.T@b and c = y.T@(mu/(RT) - b).
    Using P = rho_i.T@mu - Psi gives c = J + (P-P0)/(RT*r).
    The instability threshold applies to pressure-qualified c, not J.
    This uses Michelsen (1982) TPD and the volume formulation of Nichita (2018).

    Full component differentiation retains composition-dependent EOS parameters.
    Non-cubic trials require positive pressure and mechanical slope; cubic trials
    may cross zero within their density domain. Final roots must pass domain,
    pressure and mechanical-stability checks. Inputs use K, bar and mol/L.
    """

    fractions = _compositionFromIlr(coordinates[:-1], basis)
    density = float(model.densityScale) * math.exp(float(coordinates[-1]))
    lower = max(MIN_DENSITY_SCALE_FACTOR * model.densityScale, np.finfo(float).tiny)
    upper = HIGH_DENSITY_FRACTION * float(model.densityRootUpperBound(temperature, fractions))
    if not math.isfinite(upper) or not lower < density < upper:
        raise ValueError("molar TPD trial lies outside the density search bounds")
    densities = model.asTensor(density * fractions)

    # Final EOS pressure limits and the target-pressure residual belong to
    # root acceptance; the model-specific trial restriction is applied below.
    model.validateState(temperature, densities)
    state = kernel.state(temperature, densities, validate=False)
    energy = float(state.freeEnergy.detach().cpu().item())
    chemical = state.chemicalPotentials.detach().cpu().numpy()
    densityHessian = state.hessian.detach().cpu().numpy()
    localPressure = float(state.pressure.detach().cpu().item())
    if not (math.isfinite(energy) and math.isfinite(localPressure)
            and np.all(np.isfinite(chemical)) and np.all(np.isfinite(densityHessian))):
        raise FloatingPointError("molar TPD thermodynamic state is nonfinite")
    slope = density * float(fractions @ densityHessian @ fractions)
    if not math.isfinite(slope):
        raise FloatingPointError("molar TPD mechanical slope is nonfinite")

    # PR/SRK derivatives remain finite when pressure or its density slope
    # crosses zero. Permit those trial steps; final roots must pass both checks.
    # Other models retain positive-pressure and positive-slope trial limits.
    if not isinstance(model, CubicBellModel) and (localPressure <= 0.0 or slope <= 0.0):
        raise ValueError("molar TPD trial requires positive pressure and mechanical slope")

    # RT [bar L/mol]; both J and TPD are dimensionless.
    rt = model.gasConstant * temperature
    reduced = chemical / rt - constant
    tpdValue = float(fractions @ reduced)
    radialGradient = (localPressure - pressure) / (rt * density)

    # J = TPD - (P-P0)/(RT*r). The cutoff is applied to pressure-qualified TPD.
    value = tpdValue - radialGradient
    compositionGradient, compositionBlock = _ilrProjected(
        reduced, density / rt * densityHessian, fractions, basis,
    )
    jacobian = (np.diag(fractions) - np.outer(fractions, fractions)) @ basis
    cross = density / rt * (jacobian.T @ densityHessian @ fractions)
    gradient = np.concatenate((compositionGradient, [radialGradient]))

    # Assemble composition, density and cross-derivative blocks in log coordinates.
    hessian = np.empty((fractions.size, fractions.size))
    hessian[:-1, :-1] = compositionBlock
    hessian[:-1, -1] = cross
    hessian[-1, :-1] = cross
    hessian[-1, -1] = slope / rt - radialGradient

    qualified = None
    fixedPressureHessian = None
    if abs(localPressure - pressure) <= PRESSURE_RESIDUAL_TOL and slope > 0.0:

        # Reuse the independently defined root acceptance, including original
        # model validity, absolute pressure, finite mu/H and mechanical slope.
        qualified = _rootEvaluationFromState(
            model, temperature, pressure, fractions, density, densities,
            state.pressure, state.chemicalPotentials, state.hessian,
        )
        _, oldGradient, oldHessian, _ = _trialDerivativesFromEvaluation(
            model, temperature, fractions, constant, qualified,
        )
        _, fixedPressureHessian = _ilrProjected(
            oldGradient, oldHessian, fractions, basis,
        )
    return _MolarEvaluation(
        fractions, density, value, tpdValue, gradient, hessian,
        qualified, fixedPressureHessian,
    )


def _minimizeMolarPrepared(
    model: FreeEnergyModel, kernel: ThermoKernel, temperature: float,
    pressure: float, start: np.ndarray, initial: _DensityRootEvaluation,
    constant: np.ndarray, basis: np.ndarray, config: ScalarTrustRegionConfig,
    evidence: _TPDEvidence,
) -> TPDMinimum:
    """
    Minimize one discovered density basin with one scalar trust region.

    A local record of evaluated coordinates supplies qualification and the
    final state without repeating EOS work. No record is reused across starts.
    Failed pressure/curvature, tiny steps and exhausted budgets stay failures.
    """

    coordinates = np.concatenate((
        _ilrCoordinates(start, basis),
        [math.log(initial.root.density / model.densityScale)],
    ))
    evaluated: dict[bytes, _MolarEvaluation] = {}

    def objective(values: np.ndarray):
        """
        Return J derivatives; retain qualified evidence at any trial.
        """

        try:
            point = _molarDerivatives(
                model, kernel, temperature, pressure, constant, basis, values,
            )
        except (FloatingPointError, RuntimeError, ValueError, OverflowError, np.linalg.LinAlgError):
            return None
        evaluated[values.tobytes()] = point
        if point.root is not None:
            evidence.observe(model, temperature, point.fractions, constant, point.root)
        return point.value, point.gradient, point.hessian

    def converged(values, _value, _gradient, _hessian):
        """
        Require composition stationarity, a qualified pressure root and curvature.
        """

        point = evaluated[values.tobytes()]
        curvature = point.compositionHessian
        if point.root is None or not point.root.root.mechanicalStable or curvature is None:
            return False
        return bool(
            np.linalg.norm(point.gradient[:-1]) <= config.stationarityTol
            and (curvature.size == 0 or
                 np.linalg.eigvalsh(curvature)[0] >= -_curvatureTolerance(curvature))
        )

    lower = np.full(coordinates.size, -_ILR_COORDINATE_BOUND)
    upper = np.full(coordinates.size, _ILR_COORDINATE_BOUND)
    lower[-1] = math.log(max(
        MIN_DENSITY_SCALE_FACTOR,
        np.finfo(float).tiny / model.densityScale,
    ))
    upper[-1] = math.inf  # The actual upper bound depends on the trial composition.
    result = minimizeScalar(
        objective, coordinates, lower, upper, config=config,
        convergenceCheck=converged,
    )
    point = evaluated[result.state.tobytes()]
    curvature = point.compositionHessian
    stationarityNorm = float(np.linalg.norm(point.gradient[:-1]))
    if curvature is None:
        minimumHessianEigenvalue = None
    elif curvature.size:
        minimumHessianEigenvalue = float(np.linalg.eigvalsh(curvature)[0])
    else:
        minimumHessianEigenvalue = 0.0
    return TPDMinimum(
        composition=tuple(float(item) for item in point.fractions),
        density=point.density, value=point.tpdValue,
        converged=bool(result.converged), evaluations=result.evaluations,
        iterations=result.iterations,
        stationarityNorm=stationarityNorm,
        minimumHessianEigenvalue=minimumHessianEigenvalue,
        termination=result.termination, witness=evidence.witness,
    )


def findMolarTpdMinima(
    model: FreeEnergyModel, feedComposition: TensorLike,
    temperature: float, pressure: float, *,
    starts: Sequence[np.ndarray] | None = None,
    config: ScalarTrustRegionConfig | None = None,
    distanceTolerance: float = 2.0e-3,
    referenceDensityMolL: float | None = None,
    referencePhase: str = "stable", stopBelow: float | None = None,
    qualifiedReference: bool = False,
) -> TPDResult:
    """
    Search all discovered stable density starts using extended molar TPD.

    Inputs use K/bar/mol/L; compositions and TPD are dimensionless. ``starts``
    selects compositions, each expanded into ALL qualified stable pressure
    roots without a Gibbs ranking filter. Root discovery failures survive even
    when other roots can be optimized. ``attemptedStarts`` counts inspected
    density starts, including an already qualified reference and an immediate
    negative witness; ``failures`` can additionally contain discovery failures.

    ``qualifiedReference`` is only for a reference independently qualified by
    the local critical-stability kernel. It requires an explicit reference
    density and skips only an unambiguously matching root under the existing
    density-discovery identity tolerance, never its entire composition.
    Other roots at the feed remain starts; ambiguous matches are all searched.
    ``stopBelow`` acts on the original, pressure-qualified TPD expression.
    A negative witness can end an incomplete search; a nonnegative incomplete
    search is never reported as completed. No automatic alternative solver is
    called. Finite multistart completion does not certify a global minimum.
    """

    feed = _normalizeFeed(np.asarray(feedComposition, dtype=float))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and positive")
    if not math.isfinite(distanceTolerance) or distanceTolerance <= 0.0:
        raise ValueError("distanceTolerance must be finite and positive")
    if stopBelow is not None and (not math.isfinite(stopBelow) or stopBelow >= 0.0):
        raise ValueError("stopBelow must be finite and strictly negative")
    if qualifiedReference and referenceDensityMolL is None:
        raise ValueError("qualifiedReference requires an explicit reference density")
    controls = config or ScalarTrustRegionConfig()

    # Each starting composition is later expanded into all qualified stable density roots.
    catalog = defaultTpdStarts(model, feed, temperature, pressure) if starts is None else starts
    catalog = tuple(_normalizeStart(start) for start in catalog)
    if not catalog or any(start.shape != feed.shape for start in catalog):
        raise ValueError("at least one composition start matching the feed is required")
    if starts is None:

        # Wilson's first entry is the supplied feed. Preserve that provenance
        # rather than asking repeated normalization to reproduce identical bits.
        catalog = (feed.copy(), *catalog[1:])
    referenceLnPhi, referenceDensity, referenceKind, _ = _referenceState(
        model, temperature, pressure, feed,
        referenceDensityMolL=referenceDensityMolL, referencePhase=referencePhase,
    )
    constant = np.log(pressure / (model.gasConstant * temperature)) + np.log(feed) + referenceLnPhi
    kernel = ThermoKernel(model)
    basis = _helmertBasis(feed.size) if feed.size > 1 else np.zeros((1, 0))
    evidence = _TPDEvidence(stopBelow=stopBelow)
    found: list[TPDMinimum] = []
    failures: list[TPDStartFailure] = []
    attempted = 0
    convergedCount = 0
    stoppedEarly = False
    try:
        for start in catalog:
            try:
                roots = _findDensityRootEvaluations(
                    model, temperature, pressure, start,
                    scanPointCount=DEFAULT_SCAN_POINT_COUNT, kernel=kernel,
                    withHessian=False,
                )
            except DensityRootEvaluationError as error:
                roots = error.evaluations
                failure = TPDDensityFailure(tuple(float(v) for v in start), error.failures)
                evidence.densityFailures.append(failure)
                failures.append(TPDStartFailure(
                    tuple(float(v) for v in start),
                    "initial density discovery: unresolved_density_candidates", (failure,),
                ))
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failures.append(TPDStartFailure(
                    tuple(float(v) for v in start), f"initial density discovery: {error}",
                ))
                continue
            stableRoots = tuple(item for item in roots if item.root.mechanicalStable)
            if not stableRoots:
                failures.append(TPDStartFailure(
                    tuple(float(v) for v in start), "initial density discovery: no stable root",
                ))
                continue
            referenceMatches = tuple(index for index, initial in enumerate(stableRoots) if
                qualifiedReference and np.array_equal(start, feed)
                and abs(initial.root.density - referenceDensity)
                <= ROOT_DEDUP_RELATIVE_TOL * max(1.0, initial.root.density, referenceDensity)
            )
            for rootIndex, initial in enumerate(stableRoots):
                attempted += 1
                evidence.observe(model, temperature, start, constant, initial)
                sameReference = referenceMatches == (rootIndex,)
                try:
                    if sameReference:

                        # Use discovery's existing numerical root identity,
                        # never the much wider basin-deduplication tolerance.
                        # Multiple matches are ambiguous and none are skipped.
                        item = TPDMinimum(
                            composition=tuple(float(v) for v in start),
                            density=initial.root.density,
                            value=float(start @ (initial.chemicalPotentials / (model.gasConstant * temperature) - constant)),
                            converged=True,
                            evaluations=0,
                            iterations=0,
                            stationarityNorm=None,
                            minimumHessianEigenvalue=None,
                            termination="local_reference_qualification",
                        )
                    else:
                        item = _minimizeMolarPrepared(
                            model, kernel, temperature, pressure, start, initial,
                            constant, basis, controls, evidence,
                        )
                except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    failures.append(TPDStartFailure(
                        tuple(float(v) for v in start),
                        f"density start {initial.root.density:.17g} mol/L: {error}",
                    ))
                    continue
                if not item.converged:
                    failures.append(TPDStartFailure(
                        tuple(float(v) for v in start),
                        f"density start {initial.root.density:.17g} mol/L: {item.termination}",
                    ))
                    continue
                convergedCount += 1

                # Density is part of basin identity: equal composition alone
                # must not erase a competing pressure branch.
                duplicate = None
                for index, old in enumerate(found):
                    compositionDistance = np.linalg.norm(
                        np.asarray(item.composition) - np.asarray(old.composition),
                        ord=1,
                    )
                    if compositionDistance < distanceTolerance:
                        densityDistance = abs(math.log(item.density / old.density))
                        if densityDistance < distanceTolerance:
                            duplicate = index
                            break
                if duplicate is None:
                    found.append(item)
                elif item.value < found[duplicate].value:
                    found[duplicate] = item
    except _TPDWitnessFound:
        stoppedEarly = True
    ordered = tuple(sorted(found, key=lambda item: item.value))
    completed = bool(ordered) and not failures and not stoppedEarly
    message = (
        f"{convergedCount}/{attempted} density starts converged from {len(catalog)} composition starts; "
        f"{len(failures)} discovery/search failure(s); finite multistart is not a global certificate"
    )
    if stoppedEarly:
        message = f"qualified TPD witness below {stopBelow:.9g}; search stopped incomplete; " + message
    return TPDResult(
        feedComposition=tuple(float(v) for v in feed),
        temperature=float(temperature),
        pressure=float(pressure),
        minimum=ordered[0] if ordered else None,
        minima=ordered,
        completed=completed,
        message=message,
        attemptedStarts=attempted, convergedStarts=convergedCount,
        failedStarts=len(failures), failures=tuple(failures),
        referenceDensityMolL=referenceDensity, referenceSelection=referenceKind,
        formulation="molar_npt", witness=evidence.witness, stoppedEarly=stoppedEarly,
    )


__all__ = ["findMolarTpdMinima"]
