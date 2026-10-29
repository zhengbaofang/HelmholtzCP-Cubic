r"""
Evaluate tangent-plane distance (TPD) and search from multiple compositions.
Build the reference state and Wilson/component-biased starting compositions.
Use Michelsen's (1982) TPD criterion and return minima, negative trials and failures.
Inputs use K, bar and mol/L; compositions and TPD are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..Common.Model import FreeEnergyModel, TensorLike
from ..Kernel.Thermo import ThermoKernel
from ..Numerics.ScalarTrustRegion import ScalarTrustRegionConfig
from .DensityRoots import (
    DEFAULT_SCAN_POINT_COUNT,
    _DensityRootEvaluation,
    _findDensityRootEvaluations,
    _rootAt,
)
from .TPDMinimization import (
    TPDMinimum,
    TPDDensityFailure,
    TPDWitness,
    _TPDEvidence,
    _TPDWitnessFound,
    _helmertBasis,
    _minimizeTpdPrepared,
    _selectRootEvaluation,
)

_EPS = 1.0e-30


@dataclass(frozen=True)
class TPDStartFailure:
    """
    One failed TPD start with its normalized composition and cause.
    """

    startComposition: tuple[float, ...]
    reason: str
    densityFailures: tuple[TPDDensityFailure, ...] = ()

    def asDict(self) -> dict[str, object]:
        """
        Return a JSON-ready failed-start record.
        """

        return {
            "startComposition": list(self.startComposition),
            "reason": self.reason,
            "densityFailures": [item.asDict() for item in self.densityFailures],
        }


@dataclass(frozen=True)
class TPDResult:
    """
    Finite multistart tangent-plane-distance result for one feed state.

    ``minimum`` is the lowest value found among converged local searches.  It
    is not a mathematical certificate of the global TPD minimum.
    ``completed`` requires at least one converged basin and no failed start;
    ``failures`` preserves each incomplete start and its cause.
    ``witness`` retains the lowest qualified feasible trial, including trials
    from failed searches. It does not assert stationarity. An unresolved
    density candidate prevents completion even when a usable branch exists.
    ``stoppedEarly`` identifies an intentional negative-witness exit, which
    leaves ``completed=False``. ``attemptedStarts`` counts only started
    searches; the interrupted start is not failed unless it had unresolved
    density candidates. Unevaluated starts do not enter the failure count.
    ``referenceDensityMolL`` records the actual pressure-matching reference
    density, while ``referenceSelection`` records whether it was explicitly
    specified or selected as a stable, vapor, or liquid root. ``formulation``
    distinguishes composition-only ``composition_npt`` from joint
    ``molar_npt`` searches. Both report the original pressure-qualified TPD
    expression for accepted minima and witnesses. Molar-NPT start counters
    count density basins; its failures also include initial root discovery.
    """

    feedComposition: tuple[float, ...]
    temperature: float
    pressure: float
    minimum: TPDMinimum | None
    minima: tuple[TPDMinimum, ...]
    completed: bool
    message: str
    attemptedStarts: int = 0
    convergedStarts: int = 0
    failedStarts: int = 0
    failures: tuple[TPDStartFailure, ...] = ()
    referenceDensityMolL: float | None = None
    referenceSelection: str = "stable"
    formulation: str = "composition_npt"
    witness: TPDWitness | None = None
    stoppedEarly: bool = False

    def asDict(self) -> dict[str, object]:
        """
        Return a JSON-ready record.
        """

        return {
            "feedComposition": list(self.feedComposition),
            "temperatureK": self.temperature,
            "pressureBar": self.pressure,
            "minimum": None if self.minimum is None else self.minimum.asDict(),
            "minima": [item.asDict() for item in self.minima],
            "completed": self.completed,
            "message": self.message,
            "attemptedStarts": self.attemptedStarts,
            "convergedStarts": self.convergedStarts,
            "failedStarts": self.failedStarts,
            "failures": [item.asDict() for item in self.failures],
            "referenceDensityMolL": self.referenceDensityMolL,
            "referenceSelection": self.referenceSelection,
            "formulation": self.formulation,
            "witness": None if self.witness is None else self.witness.asDict(),
            "stoppedEarly": self.stoppedEarly,
        }


def _normalizeFeed(values: np.ndarray) -> np.ndarray:
    """
    Normalize a strictly positive active-component feed composition.
    """

    fractions = np.asarray(values, dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("feed composition must be a nonempty vector")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("feed composition must be finite")
    if np.any(fractions <= 0.0):
        raise ValueError("active-component feed fractions must be positive")
    total = float(np.sum(fractions))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("feed composition sum must be finite and positive")
    return fractions / total


def _normalizeStart(values: np.ndarray) -> np.ndarray:
    """
    Normalize a nonnegative search start and regularize exact zeros.
    """

    fractions = np.asarray(values, dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("TPD start must be a nonempty vector")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("TPD start must be finite")
    if np.any(fractions < 0.0):
        raise ValueError("TPD start fractions cannot be negative")
    total = float(np.sum(fractions))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("TPD start sum must be finite and positive")
    fractions = np.maximum(fractions / total, _EPS)
    return fractions / np.sum(fractions)


def wilsonStarts(
    composition: TensorLike,
    temperature: float,
    pressure: float,
    *,
    criticalTemperatures: Sequence[float] | None = None,
    criticalPressures: Sequence[float] | None = None,
    acentricFactors: Sequence[float] | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Return vapor/liquid and component-biased TPD starts.

    With the pure critical properties the Wilson K distribution
    (Wilson, 1959) generates the standard vapor/liquid directions; the
    lightest and heaviest (by critical temperature) components receive a 0.95
    biased start.  Without critical temperatures every component receives a
    biased start, which keeps the coverage generic for Helmholtz models that
    do not expose Tc/Pc/omega.
    """

    fractions = _normalizeFeed(np.asarray(composition, dtype=float))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and positive")
    starts = [fractions]
    if (
        criticalTemperatures is not None
        and criticalPressures is not None
        and acentricFactors is not None
    ):
        temperatures = np.asarray(criticalTemperatures, dtype=float)
        pressures = np.asarray(criticalPressures, dtype=float)
        acentric = np.asarray(acentricFactors, dtype=float)
        if (
            temperatures.shape != fractions.shape
            or pressures.shape != fractions.shape
            or acentric.shape != fractions.shape
        ):
            raise ValueError("critical-property arrays must match composition")
        if (
            not np.all(np.isfinite(temperatures))
            or not np.all(np.isfinite(pressures))
            or not np.all(np.isfinite(acentric))
        ):
            raise ValueError("critical properties must be finite")
        if np.any(temperatures <= 0.0) or np.any(pressures <= 0.0):
            raise ValueError("critical temperatures and pressures must be positive")
        kValues = pressures / pressure * np.exp(
            5.373 * (1.0 + acentric) * (1.0 - temperatures / temperature)
        )
        if not np.all(np.isfinite(kValues)) or np.any(kValues <= 0.0):
            raise FloatingPointError("Wilson K values must be finite and positive")
        starts.extend(
            (
                _normalizeStart(fractions * kValues),
                _normalizeStart(fractions / kValues),
                _normalizeStart(fractions * np.sqrt(kValues)),
                _normalizeStart(fractions / np.sqrt(kValues)),
            )
        )
        biasedIndices = np.unique(
            (int(np.argmin(temperatures)), int(np.argmax(temperatures)))
        )
    else:
        biasedIndices = np.arange(fractions.size)

    if fractions.size == 1:
        return (fractions,)
    count = fractions.size
    for index in biasedIndices:
        biased = np.full(count, 0.05 / max(1, count - 1))
        biased[index] = 0.95
        starts.append(_normalizeStart(biased))
    return tuple(starts)


def defaultTpdStarts(
    model: FreeEnergyModel,
    feedComposition: TensorLike,
    temperatureK: float,
    pressureBar: float,
) -> tuple[np.ndarray, ...]:
    """
    Return the model-aware default TPD start catalog.

    ``model`` must use K/bar/mol/L; no unit conversion is performed here.
    ``temperatureK`` is in kelvin and ``pressureBar`` is in bar. Models that
    expose pure critical properties receive Wilson-K starts; other models use
    the generic component-biased catalog from :func:`wilsonStarts`.
    """

    feed = _normalizeFeed(np.asarray(feedComposition, dtype=float))
    return wilsonStarts(
        feed,
        temperatureK,
        pressureBar,
        **_criticalProperties(model),
    )


def _referenceState(
    model: FreeEnergyModel,
    temperature: float,
    pressure: float,
    fractions: np.ndarray,
    *,
    referenceDensityMolL: float | None = None,
    referencePhase: str = "stable",
) -> tuple[
    np.ndarray,
    float,
    str,
    _DensityRootEvaluation | None,
]:
    """
    Return reference fugacity coefficients, density [mol/L] and the checked root.
    """

    if referencePhase not in {"stable", "vapor", "liquid"}:
        raise ValueError("referencePhase must be 'stable', 'vapor', or 'liquid'")
    kernel = ThermoKernel(model)
    if referenceDensityMolL is None:
        evaluations = _findDensityRootEvaluations(
            model,
            temperature,
            pressure,
            fractions,
            scanPointCount=DEFAULT_SCAN_POINT_COUNT,
            kernel=kernel,
        )
        evaluation = _selectRootEvaluation(evaluations, referencePhase)
        if evaluation is None:
            raise ValueError(
                f"reference state has no mechanically stable {referencePhase} root"
            )
        root = evaluation.root
        return (
            np.asarray(root.lnphi, dtype=float),
            root.density,
            referencePhase,
            evaluation,
        )

    density = float(referenceDensityMolL)
    if referencePhase != "stable":
        raise ValueError(
            "referenceDensityMolL cannot be combined with referencePhase"
        )
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("referenceDensityMolL must be finite and positive")
    try:
        root = _rootAt(
            model,
            kernel,
            model.asTensor(temperature),
            temperature,
            pressure,
            np.asarray(fractions, dtype=float),
            density,
            withHessian=False,
        )
    except FloatingPointError as error:
        raise ValueError(
            "referenceDensityMolL does not satisfy the supplied pressure: "
            f"{error}"
        ) from error
    return (
        np.asarray(root.lnphi, dtype=float),
        root.density,
        "specified_density",
        None,
    )


def minimizeTpd(
    model: FreeEnergyModel,
    feedComposition: TensorLike,
    temperature: float,
    pressure: float,
    *,
    start: np.ndarray,
    config: ScalarTrustRegionConfig | None = None,
    referenceLnPhi: np.ndarray | None = None,
    referenceDensityMolL: float | None = None,
    referencePhase: str = "stable",
) -> TPDMinimum:
    """
    Minimize TPD from one orthonormal ILR composition start.

    The model must use K/bar/mol/L; temperature is K, pressure is bar, and
    ``referenceDensityMolL`` is mol/L. This entry does not convert units.
    The reference state is selected by ``referencePhase`` unless its molar
    density is supplied explicitly in ``referenceDensityMolL``. Supplying
    ``referenceLnPhi`` bypasses both reference-state routes.

    The dimensionless objective is D(y) = sum(y_i * (ln(y_i) + ln(phi_i(y))
    - ln(z_i) - ln(phi_i(z)))). Fugacity coefficients follow from the Helmholtz
    chemical potentials: ln(phi_i) = mu_i/(RT) - ln(rho_i) - ln(Z).
    The solver uses y = softmax(Q*u), where Q.T@Q = I and Q.T@ones = 0,
    and exact projected derivatives with the More--Sorensen trust-region method.
    """

    feed = _normalizeFeed(np.asarray(feedComposition, dtype=float))
    start = _normalizeStart(start)
    if start.shape != feed.shape:
        raise ValueError("TPD start must match feed composition")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and positive")
    referenceEvaluation: _DensityRootEvaluation | None = None
    referenceKind = "provided_lnphi"
    if referenceLnPhi is None:
        referenceLnPhi, _, referenceKind, referenceEvaluation = _referenceState(
            model,
            temperature,
            pressure,
            feed,
            referenceDensityMolL=referenceDensityMolL,
            referencePhase=referencePhase,
        )
    else:
        if referenceDensityMolL is not None or referencePhase != "stable":
            raise ValueError(
                "referenceLnPhi cannot be combined with reference density or phase"
            )
        referenceLnPhi = np.asarray(referenceLnPhi, dtype=float)
        if referenceLnPhi.shape != feed.shape:
            raise ValueError("referenceLnPhi must match feed composition")
        if not np.all(np.isfinite(referenceLnPhi)):
            raise ValueError("referenceLnPhi must be finite")
    kernel = ThermoKernel(model)
    basis = None if feed.size == 1 else _helmertBasis(feed.size)
    return _minimizeTpdPrepared(
        model,
        feed,
        temperature,
        pressure,
        start,
        referenceLnPhi,
        kernel,
        basis,
        config=config,
        initialEvaluation=(
            referenceEvaluation
            if referenceKind == "stable" and np.array_equal(start, feed)
            else None
        ),
    )


def findTpdMinima(
    model: FreeEnergyModel,
    feedComposition: TensorLike,
    temperature: float,
    pressure: float,
    *,
    starts: Sequence[np.ndarray] | None = None,
    config: ScalarTrustRegionConfig | None = None,
    distanceTolerance: float = 2.0e-3,
    referenceDensityMolL: float | None = None,
    referencePhase: str = "stable",
    stopBelow: float | None = None,
) -> TPDResult:
    """
    Return converged minima and any separately accepted TPD trials.

    The model must use K/bar/mol/L; temperature is K, pressure is bar, and
    ``referenceDensityMolL`` is mol/L. This entry does not convert units.
    Each supplied start drives one local Michelsen-type minimization.  A
    negative value proves that the feed is unstable, while a nonnegative
    lowest value only means that this finite set of starts found no negative
    basin. ``referenceDensityMolL`` evaluates TPD relative to that exact
    pressure-matching state; otherwise ``referencePhase`` selects a
    mechanically stable vapor, liquid, or minimum-Gibbs root.

    ``stopBelow=None`` retains the full multistart minimization. A finite
    negative ``stopBelow`` requests an instability screen: stop both the
    current minimization and remaining starts when a pressure-qualified
    witness satisfies the strict inequality ``TPD < stopBelow``. Equality
    continues the search. This exit supplies evidence, not a converged
    minimum, and preserves any unresolved density candidates.
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
    if starts is None:
        starts = defaultTpdStarts(model, feed, temperature, pressure)
    starts = tuple(_normalizeStart(start) for start in starts)
    if not starts:
        raise ValueError("at least one TPD start is required")
    if any(start.shape != feed.shape for start in starts):
        raise ValueError("each TPD start must match feed composition")
    (
        referenceLnPhi,
        referenceDensity,
        referenceKind,
        referenceEvaluation,
    ) = _referenceState(
        model,
        temperature,
        pressure,
        feed,
        referenceDensityMolL=referenceDensityMolL,
        referencePhase=referencePhase,
    )
    found: list[TPDMinimum] = []
    attemptedStarts = 0
    stoppedEarly = False
    convergedStarts = 0
    failedStarts = 0
    failures: list[TPDStartFailure] = []
    witness: TPDWitness | None = None
    kernel = ThermoKernel(model)
    basis = None if feed.size == 1 else _helmertBasis(feed.size)
    for start in starts:
        attemptedStarts += 1
        evidence = _TPDEvidence(stopBelow=stopBelow)
        try:
            item = _minimizeTpdPrepared(
                model,
                feed,
                temperature,
                pressure,
                start,
                referenceLnPhi,
                kernel,
                basis,
                config,
                initialEvaluation=(
                    referenceEvaluation
                    if referenceKind == "stable" and np.array_equal(start, feed)
                    else None
                ),
                evidence=evidence,
            )
        except _TPDWitnessFound:
            stoppedEarly = True

            # A valid negative witness survives unresolved competing roots,
            # but stopping must not erase those failures or claim completion.
            if evidence.densityFailures:
                failedStarts += 1
                failures.append(TPDStartFailure(
                    tuple(float(value) for value in start),
                    "unresolved_density_candidates before negative-witness exit",
                    tuple(evidence.densityFailures),
                ))
            break
        except (
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            failedStarts += 1
            failures.append(
                TPDStartFailure(
                    tuple(float(value) for value in start),
                    f"{type(error).__name__}: {error}",
                    tuple(evidence.densityFailures),
                )
            )
            continue
        finally:

            # A failed optimizer or unresolved competing root does not erase
            # a feasible lower-free-energy trial already inspected.
            if evidence.witness is not None and (
                witness is None or evidence.witness.value < witness.value
            ):
                witness = evidence.witness
        if not item.converged:
            failedStarts += 1
            failures.append(
                TPDStartFailure(
                    tuple(float(value) for value in start),
                    f"optimizer stopped: {item.termination}",
                    item.densityFailures,
                )
            )
            continue
        convergedStarts += 1
        composition = np.asarray(item.composition, dtype=float)
        if any(
            np.linalg.norm(composition - np.asarray(old.composition), ord=1)
            < distanceTolerance
            for old in found
        ):
            continue
        found.append(item)
    ordered = tuple(sorted(found, key=lambda item: item.value))
    converged = tuple(item for item in ordered if item.converged)
    minimum = converged[0] if converged else None
    completed = not stoppedEarly and minimum is not None and failedStarts == 0
    message = (
        (
            f"lowest converged TPD value found from {convergedStarts}/"
            f"{len(starts)} starts; finite multistart search is not a global "
            "minimum certificate"
            + (
                ""
                if completed
                else f"; {failedStarts} start(s) failed"
            )
        )
        if minimum is not None
        else f"no converged TPD basin from {len(starts)} starts"
    )
    if stoppedEarly:
        message = (
            f"stopped on a qualified TPD witness below {stopBelow:.9g} after "
            f"{attemptedStarts}/{len(starts)} starts; multistart minimization "
            f"is incomplete; {failedStarts} start(s) had failures"
        )
    return TPDResult(
        feedComposition=tuple(float(value) for value in feed),
        temperature=float(temperature),
        pressure=float(pressure),
        minimum=minimum,
        minima=ordered,
        completed=completed,
        message=message,
        attemptedStarts=attemptedStarts,
        convergedStarts=convergedStarts,
        failedStarts=failedStarts,
        failures=tuple(failures),
        referenceDensityMolL=referenceDensity,
        referenceSelection=referenceKind,
        witness=witness,
        stoppedEarly=stoppedEarly,
    )


def _criticalProperties(
    model: FreeEnergyModel,
) -> dict[str, tuple[float, ...] | None]:
    """
    Read optional pure critical properties exposed by the model.
    """

    temperatures = getattr(model, "pureCriticalTemp", None)
    pressures = getattr(model, "pureCriticalPres", None)
    acentric = getattr(model, "pureAcentricFactor", None)
    if temperatures is None or pressures is None or acentric is None:
        return {
            "criticalTemperatures": None,
            "criticalPressures": None,
            "acentricFactors": None,
        }
    return {
        "criticalTemperatures": tuple(
            float(value) for value in temperatures
        ),
        "criticalPressures": tuple(float(value) for value in pressures),
        "acentricFactors": tuple(float(value) for value in acentric),
    }


__all__ = [
    "TPDMinimum",
    "TPDResult",
    "TPDStartFailure",
    "TPDWitness",
    "TPDDensityFailure",
    "defaultTpdStarts",
    "findTpdMinima",
    "minimizeTpd",
    "wilsonStarts",
]
