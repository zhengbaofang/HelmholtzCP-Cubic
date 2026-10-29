r"""
Fixed-composition critical-point solving on the Helmholtz-density kernel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
import math

import numpy as np

from ..Common.Composition import (
    compositionDensityScale,
    compositionTemperatureScale,
    normalizeComposition,
    normalizeFullComposition,
)
from ..Common.Model import FreeEnergyModel
from ..Kernel.Critical import (
    DensityCriticalEvaluation,
    DensityCriticalKernel,
    DensityCriticalState,
)
from ..Numerics.LeastSquaresTrustRegion import solveLeastSquares
from ..Transform.CubicBell import CubicBellModel
from ..Transform.WorkspaceUnits import asWorkspaceModel
from .DirectBackend import (
    ADCriticalEvaluator,
    CriticalBackend,
    CriticalEvaluator,
    selectCriticalBackend,
)
from .PointSelection import deduplicatePoints


@dataclass(frozen=True)
class CriticalSolverConfig:
    """
    Controls for critical-point correction and duplicate removal.
    Temperature bounds and deduplication use K; density bounds use mol/L.
    Pressure deduplication uses bar. Residual tolerances, log-coordinate steps
    and radii, relative density tolerances and ratios are dimensionless.
    """

    resTol: float = 1.0e-8
    stepTol: float = 1.0e-10
    maxEval: int = 120
    temperatureBoundsK: tuple[float, float] | None = None
    densityBoundsMolL: tuple[float, float] | None = None
    dedupeTemperatureK: float = 1.0e-4
    dedupePressureBar: float = 1.0e-2
    dedupeDensityRelative: float = 1.0e-7
    requirePositivePressure: bool = True
    initialRadius: float = 10.0
    maxRadius: float = 100.0
    acceptanceRatio: float = 0.1
    stagnationRelativeTol: float = 1.0e-12
    maxStagnantEvaluations: int = 12

    def __post_init__(self) -> None:
        for name in (
            "resTol",
            "stepTol",
            "dedupeTemperatureK",
            "dedupePressureBar",
            "dedupeDensityRelative",
            "initialRadius",
            "maxRadius",
            "stagnationRelativeTol",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not isinstance(self.maxEval, int)
            or isinstance(self.maxEval, bool)
            or self.maxEval < 1
        ):
            raise ValueError("maxEval must be a positive integer")
        if (
            not isinstance(self.maxStagnantEvaluations, int)
            or isinstance(self.maxStagnantEvaluations, bool)
            or self.maxStagnantEvaluations < 1
        ):
            raise ValueError("maxStagnantEvaluations must be a positive integer")
        for name in ("temperatureBoundsK", "densityBoundsMolL"):
            bounds = getattr(self, name)
            if bounds is not None and (
                len(bounds) != 2
                or any(
                    isinstance(value, bool) or not math.isfinite(value)
                    for value in bounds
                )
                or not (0.0 < bounds[0] < bounds[1])
            ):
                raise ValueError(
                    f"{name} must be a positive finite increasing pair"
                )
        if not isinstance(self.requirePositivePressure, bool):
            raise TypeError("requirePositivePressure must be a boolean")
        if self.initialRadius > self.maxRadius:
            raise ValueError("trust-region radii must satisfy initial <= max")
        if not math.isfinite(self.acceptanceRatio) or not (
            0.0 <= self.acceptanceRatio < 1.0
        ):
            raise ValueError("acceptanceRatio must lie in [0, 1)")


@dataclass(frozen=True)
class CriticalPointResult:
    """
    One corrected ordinary CP candidate with units and solver diagnostics.

    ``T``, ``P``, and ``rho`` use K, bar, and mol/L. Protection counters refer
    to the local least-squares correction that produced this point; points
    reconstructed from a traced locus retain zero when no per-point correction
    record exists. ``backend`` is required provenance for the implementation
    that evaluated the accepted point.
    """

    T: float
    P: float
    rho: float
    composition: tuple[float, ...]
    qMin: float
    cStar: float
    rNorm: float
    eigenvalues: tuple[float, ...]
    nNegativeEigenvalues: int
    domainMargin: float
    converged: bool
    source: str
    nEval: int | None
    solverTermination: str
    message: str
    backend: str
    rejectedSteps: int = 0
    infeasibleEvaluations: int = 0
    subproblemFailures: int = 0
    cauchySteps: int = 0
    boundaryReflections: int = 0

    def asDict(self) -> dict[str, object]:
        """
        Return a JSON-ready critical-point record.
        """

        return {
            "temperatureK": self.T,
            "pressureBar": self.P,
            "densityMolL": self.rho,
            "composition": self.composition,
            "qMin": self.qMin,
            "cStar": self.cStar,
            "rNorm": self.rNorm,
            "eigenvalues": self.eigenvalues,
            "nNegativeEigenvalues": self.nNegativeEigenvalues,
            "domainMargin": self.domainMargin,
            "converged": self.converged,
            "source": self.source,
            "nEval": self.nEval,
            "solverTermination": self.solverTermination,
            "message": self.message,
            "rejectedSteps": self.rejectedSteps,
            "infeasibleEvaluations": self.infeasibleEvaluations,
            "subproblemFailures": self.subproblemFailures,
            "cauchySteps": self.cauchySteps,
            "boundaryReflections": self.boundaryReflections,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class CriticalSolveAttempt:
    """
    One explicit enumeration attempt, including rejected candidates.
    """

    startT: float
    startRho: float
    accepted: bool
    reason: str
    point: CriticalPointResult | None = None


@dataclass(frozen=True)
class CriticalEnumerationResult:
    """
    Deduplicated roots plus complete explicit-enumeration diagnostics.
    """

    roots: tuple[CriticalPointResult, ...]
    attempts: tuple[CriticalSolveAttempt, ...]

    def primary(self) -> CriticalPointResult:
        """
        Return the highest-temperature accepted critical root.
        """

        if not self.roots:
            raise RuntimeError("no converged physical critical point was found")
        return max(self.roots, key=lambda point: (point.T, -point.rNorm))


class CriticalPointSolver:
    """
    Refine one ordinary CP root or explicitly enumerate candidate roots.

    ``backend`` selects only how the local ``[qMin, cStar]`` residual and its
    Jacobian are evaluated. The model remains the sole source of EOS identity:
    ``"ad"`` works for every supported Helmholtz model, ``"analytic"``
    requires a supported exact model-specific implementation, ``"pymcx"``
    uses multicomplex PR/SRK derivatives, and ``"auto"``
    selects analytic support when available. A runtime failure never switches
    backend implicitly.

    Inputs, bounds, returned points and ``solver.model`` use K/bar/mol/L.
    Native-unit EOS objects are adapted once at construction; their own
    low-level kernels continue to use their documented native units.
    """

    def __init__(
        self,
        model: FreeEnergyModel,
        config: CriticalSolverConfig | None = None,
        *,
        backend: CriticalBackend | str = CriticalBackend.AD,
    ) -> None:
        self.model = asWorkspaceModel(model)
        self.kernel = DensityCriticalKernel(self.model)
        self.config = config or CriticalSolverConfig()
        self._adEvaluator = ADCriticalEvaluator(self.kernel)
        self.directBackendMode, self._directEvaluator = selectCriticalBackend(
            self.model,
            self._adEvaluator,
            backend,
        )
        self.directBackendName = self._directEvaluator.name

    def correct(
        self,
        composition: Sequence[float] | np.ndarray,
        T: float,
        rho: float,
        *,
        source: str = "user_start",
    ) -> CriticalPointResult:
        """
        Refine one CP guess in logarithmic temperature-density coordinates.
        """

        activeSolver, fractions, fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            point = activeSolver.correct(fractions, T, rho, source=source)
            return replace(
                point,
                composition=tuple(float(value) for value in fullComposition),
            )
        if not (math.isfinite(T) and math.isfinite(rho) and T > 0.0 and rho > 0.0):
            raise ValueError("critical start T and rho must be finite and positive")
        bounds = self._bounds(fractions)

        # Solve in log(T/K) and log(rho/(mol/L)) so temperature and density stay positive.
        initial = np.log(np.array([T, rho], dtype=float))
        lower = np.log(np.array([bounds[0][0], bounds[1][0]], dtype=float))
        upper = np.log(np.array([bounds[0][1], bounds[1][1]], dtype=float))
        if np.any(initial <= lower) or np.any(initial >= upper):
            raise ValueError(
                "critical start must lie strictly inside the active "
                "temperature and density bounds"
            )

        latestEvaluation: DensityCriticalEvaluation | None = None

        def evaluate(logState: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """
            Return the local residual and Jacobian for the shared numerical solver.
            """

            nonlocal latestEvaluation
            latestEvaluation = self._directEvaluator.criticalEvaluation(
                logState,
                fractions,
            )
            return (
                latestEvaluation.residual.cpu().numpy(),
                latestEvaluation.jacobian.cpu().numpy(),
            )

        result = solveLeastSquares(
            evaluate,
            initial,
            lower,
            upper,
            resTol=self.config.resTol,
            stepTol=self.config.stepTol,
            maxEval=self.config.maxEval,
            initialRadius=self.config.initialRadius,
            maxRadius=self.config.maxRadius,
            acceptanceRatio=self.config.acceptanceRatio,
            stagnationRelativeTol=self.config.stagnationRelativeTol,
            maxStagnantEvaluations=self.config.maxStagnantEvaluations,
        )
        termination = result.termination
        retainedCritical = None
        if latestEvaluation is not None and np.array_equal(
            latestEvaluation.logState.cpu().numpy(),
            result.state,
        ):
            retainedCritical = latestEvaluation.critical
        return self.pointFromLogState(
            result.state,
            fractions,
            source=source,
            nEval=result.evaluations,
            optimizerSuccess=result.converged,
            solverTermination=termination,
            message=self._terminationMessage(termination),
            rejectedSteps=result.rejectedSteps,
            infeasibleEvaluations=result.infeasibleEvaluations,
            subproblemFailures=result.subproblemFailures,
            cauchySteps=result.cauchySteps,
            boundaryReflections=result.boundaryReflections,
            _criticalState=retainedCritical,
            _evaluator=self._directEvaluator,
        )

    def solveDirect(
        self,
        composition: Sequence[float] | np.ndarray,
    ) -> CriticalPointResult:
        """
        Solve once from the documented model-specific direct initial state.

        Classic PR/SRK uses the ``K=3`` initial state
        ``T0=sum(x_i*Tc_i)`` and ``rho0=1/(3*b)``. Other Helmholtz models use
        their existing composition temperature and density scales. The method
        performs exactly one local correction and never invokes spinodal discovery,
        ``solveAll()``, or another initial state after failure.
        """

        activeSolver, fractions, fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            point = activeSolver.solveDirect(fractions)
            return replace(
                point,
                composition=tuple(float(value) for value in fullComposition),
            )
        T, rho, source = self._directStart(fractions)
        return self.correct(fractions, T, rho, source=source)

    def solveAll(
        self,
        composition: Sequence[float] | np.ndarray,
        starts: Iterable[tuple[float, float]] | None = None,
    ) -> CriticalEnumerationResult:
        """
        Run deterministic explicit enumeration and retain every attempt.

        ``starts=None`` selects the documented 5-by-6 model-aware default
        grid. Supplying ``starts`` uses exactly that ordered iterable. This is
        a local multi-start candidate search, not a certified all-root proof.
        """

        fractions = normalizeFullComposition(
            composition,
            self.model.componentCount,
        )
        candidates: list[CriticalPointResult] = []
        attempts: list[CriticalSolveAttempt] = []
        searchStarts = list(starts) if starts is not None else self.defaultStarts(fractions)
        for index, start in enumerate(searchStarts):
            source = f"multistart_{index}"
            if len(start) != 2:
                attempts.append(
                    CriticalSolveAttempt(
                        math.nan,
                        math.nan,
                        False,
                        "start must contain (T, rho)",
                    )
                )
                continue
            temperature, density = float(start[0]), float(start[1])
            if not (
                math.isfinite(temperature)
                and math.isfinite(density)
                and temperature > 0.0
                and density > 0.0
            ):
                attempts.append(
                    CriticalSolveAttempt(
                        temperature,
                        density,
                        False,
                        "invalid_start",
                    )
                )
                continue
            try:
                point = self.correct(
                    fractions,
                    temperature,
                    density,
                    source=source,
                )
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exception:
                attempts.append(
                    CriticalSolveAttempt(
                        temperature,
                        density,
                        False,
                        f"{type(exception).__name__}: {exception}",
                    )
                )
                continue
            accepted = point.converged and (
                not self.config.requirePositivePressure or point.P > 0.0
            )
            reason = "accepted" if accepted else "not_converged_or_nonphysical"
            attempts.append(
                CriticalSolveAttempt(
                    temperature,
                    density,
                    accepted,
                    reason,
                    point,
                )
            )
            if accepted:
                candidates.append(point)
        return CriticalEnumerationResult(
            roots=tuple(deduplicatePoints(candidates, self.config)),
            attempts=tuple(attempts),
        )

    def solvePrimary(
        self,
        composition: Sequence[float] | np.ndarray,
        starts: Iterable[tuple[float, float]] | None = None,
    ) -> CriticalPointResult:
        """
        Enumerate explicitly and return the highest-temperature root.
        """

        return self.solveAll(composition, starts=starts).primary()

    def defaultStarts(
        self,
        composition: Sequence[float] | np.ndarray,
    ) -> list[tuple[float, float]]:
        """
        Return the ordered 5-by-6 geometric grid used by ``solveAll``.
        """

        activeSolver, fractions, _fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            return activeSolver.defaultStarts(fractions)
        tempBounds, densityBounds = self._bounds(fractions)
        temperatures = np.geomspace(
            tempBounds[0] * 1.05,
            tempBounds[1] / 1.05,
            5,
        )
        densities = np.geomspace(
            densityBounds[0] * 1.2,
            densityBounds[1] / 1.2,
            6,
        )
        return [
            (float(temperature), float(density))
            for temperature in temperatures
            for density in densities
        ]

    def stateBounds(
        self,
        composition: Sequence[float] | np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Return the active physical bounds for one composition.
        """

        activeSolver, fractions, _fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            return activeSolver.stateBounds(fractions)
        return self._bounds(fractions)

    def initialScales(
        self,
        composition: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        """
        Return composition temperature and density scales in active space.
        """

        activeSolver, fractions, _fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            return activeSolver.initialScales(fractions)
        return (
            compositionTemperatureScale(self.model, fractions),
            compositionDensityScale(self.model, fractions),
        )

    def pointFromLogState(
        self,
        logState: np.ndarray,
        composition: Sequence[float] | np.ndarray,
        *,
        source: str,
        nEval: int | None,
        optimizerSuccess: bool,
        solverTermination: str,
        message: str,
        rejectedSteps: int = 0,
        infeasibleEvaluations: int = 0,
        subproblemFailures: int = 0,
        cauchySteps: int = 0,
        boundaryReflections: int = 0,
        _criticalState: DensityCriticalState | None = None,
        _evaluator: CriticalEvaluator | None = None,
    ) -> CriticalPointResult:
        """
        Build a public CP result from an already corrected logarithmic state.

        ``_criticalState`` is internal retained evidence from the corrector's
        final accepted evaluation. Direct callers omit it and receive the same
        independent state evaluation as before.
        """

        activeSolver, fractions, fullComposition = self._activeContext(composition)
        if activeSolver is not self:
            point = activeSolver.pointFromLogState(
                logState,
                fractions,
                source=source,
                nEval=nEval,
                optimizerSuccess=optimizerSuccess,
                solverTermination=solverTermination,
                message=message,
                rejectedSteps=rejectedSteps,
                infeasibleEvaluations=infeasibleEvaluations,
                subproblemFailures=subproblemFailures,
                cauchySteps=cauchySteps,
                boundaryReflections=boundaryReflections,
                _criticalState=_criticalState,
                _evaluator=_evaluator,
            )
            return replace(
                point,
                composition=tuple(float(value) for value in fullComposition),
            )
        T, rho = np.exp(np.asarray(logState, dtype=float))
        evaluator = self._adEvaluator if _evaluator is None else _evaluator
        state = (
            _criticalState
            if _criticalState is not None
            else evaluator.evaluate(T, rho, fractions)
        )
        densities = state.densities
        pressureTensor = evaluator.pressure(T, densities)
        qMin = float(state.qMin.detach().cpu().item())
        cStar = float(state.cStar.detach().cpu().item())
        rNorm = float(math.hypot(qMin, cStar))
        eigenvalues = tuple(
            float(value)
            for value in state.eigenvalues.detach().cpu().tolist()
        )
        eigenTol = max(self.config.resTol * 10.0, 1.0e-7)
        nNegativeEigenvalues = sum(value < -eigenTol for value in eigenvalues)
        margin = float(
            self.model.domainMargin(T, densities).detach().cpu().item()
        )
        P = float(pressureTensor.detach().cpu().item())
        converged = bool(
            optimizerSuccess
            and rNorm <= self.config.resTol
            and margin > 0.0
            and all(math.isfinite(value) for value in eigenvalues)
            and (not self.config.requirePositivePressure or P > 0.0)
        )
        return CriticalPointResult(
            T=float(T),
            P=P,
            rho=float(rho),
            composition=tuple(float(value) for value in fractions),
            qMin=qMin,
            cStar=cStar,
            rNorm=rNorm,
            eigenvalues=eigenvalues,
            nNegativeEigenvalues=nNegativeEigenvalues,
            domainMargin=margin,
            converged=converged,
            source=source,
            nEval=nEval,
            solverTermination=solverTermination,
            message=message,
            rejectedSteps=rejectedSteps,
            infeasibleEvaluations=infeasibleEvaluations,
            subproblemFailures=subproblemFailures,
            cauchySteps=cauchySteps,
            boundaryReflections=boundaryReflections,
            backend=evaluator.name,
        )

    def pointFromLocusState(
        self,
        state: Sequence[float] | np.ndarray,
        *,
        source: str,
    ) -> CriticalPointResult:
        """
        Build a critical-point record from ``[ln T, ln rho, x1]``.
        """

        u = np.asarray(state, dtype=float)
        if self.model.componentCount != 2 or u.shape != (3,):
            raise ValueError("binary critical-locus state must have shape (3,)")
        x = normalizeComposition(
            [u[2], 1.0 - u[2]],
            self.model.componentCount,
        )
        return self.pointFromLogState(
            u[:2],
            x,
            source=source,
            nEval=None,
            optimizerSuccess=True,
            solverTermination="pseudo_arclength_corrector",
            message="accepted by pseudo-arclength corrector",
        )

    @staticmethod
    def _terminationMessage(termination: str) -> str:
        messages = {
            "residual": "residual tolerance satisfied",
            "step_tolerance": "step tolerance reached",
            "maximum_evaluations": "maximum evaluations reached",
            "radius_tolerance": "trust-region radius tolerance reached",
            "insufficient_progress": "trust-region residual reduction stagnated",
            "linear_algebra_failure": "trust-region subproblem has no usable step",
        }
        return messages.get(termination, termination)

    def _bounds(self, composition: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        tempScale = compositionTemperatureScale(self.model, composition)
        if self.config.temperatureBoundsK is None:
            lowerT = 0.30 * tempScale
            upperT = 5.0 * tempScale
            parameters = getattr(self.model, "parameters", None)
            tripleT = getattr(parameters, "tripleTemp", None)
            maximumT = getattr(parameters, "maxTemp", None)
            if tripleT is not None:
                lowerT = max(lowerT, float(tripleT) * (1.0 + 1.0e-10))
            if maximumT is not None:
                upperT = min(upperT, float(maximumT) * (1.0 - 1.0e-10))
            tempBounds = (lowerT, upperT)
        else:
            tempBounds = self.config.temperatureBoundsK
        if self.config.densityBoundsMolL is None:
            upperDensity = float(
                self.model.densityRootUpperBound(tempScale, composition)
            )
            densityBounds = (0.005 * upperDensity, 0.95 * upperDensity)
        else:
            densityBounds = self.config.densityBoundsMolL
        return tempBounds, densityBounds

    def _directStart(self, composition: np.ndarray) -> tuple[float, float, str]:
        """
        Return the single documented initial state for the direct route.
        """

        if type(self.model) is CubicBellModel:
            T = compositionTemperatureScale(self.model, composition)
            rho = 1.0 / (3.0 * float(self.model.coVolume(composition)))
            return T, rho, "direct_cubic_K3"
        T, rho = self.initialScales(composition)
        return T, rho, "direct_model_scale"

    def _activeContext(
        self,
        composition: Sequence[float] | np.ndarray,
    ) -> tuple["CriticalPointSolver", np.ndarray, np.ndarray]:
        """
        Return an active-space solver, composition, and normalized full vector.
        """

        fullComposition = normalizeFullComposition(
            composition,
            self.model.componentCount,
        )
        activeIndices = tuple(
            int(index) for index in np.flatnonzero(fullComposition > 0.0)
        )
        if len(activeIndices) == self.model.componentCount:
            return self, fullComposition, fullComposition
        activeModel = self.model.activeSubmodel(activeIndices)
        activeComposition = fullComposition[np.asarray(activeIndices, dtype=int)]
        return (
            CriticalPointSolver(
                activeModel,
                config=self.config,
                backend=self.directBackendMode,
            ),
            activeComposition,
            fullComposition,
        )


__all__ = [
    "CriticalEnumerationResult",
    "CriticalPointResult",
    "CriticalSolveAttempt",
    "CriticalBackend",
    "CriticalSolverConfig",
    "CriticalPointSolver",
]
