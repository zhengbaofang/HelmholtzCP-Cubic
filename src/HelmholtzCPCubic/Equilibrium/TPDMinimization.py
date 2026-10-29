r"""
Minimize TPD from one composition at fixed temperature and pressure.
Use log-ratio coordinates to keep mole fractions positive and normalized.
Project exact derivatives into these coordinates for trust-region minimization.
Follow Michelsen (1982) and More and Sorensen (1983); inputs use K, bar and mol/L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..Common.Model import FreeEnergyModel
from ..Kernel.Thermo import ThermoKernel
from ..Numerics.ScalarTrustRegion import (
    ScalarTrustRegionConfig,
    minimizeScalar,
)
from .DensityRoots import (
    DEFAULT_SCAN_POINT_COUNT,
    DensityRoot,
    DensityRootEvaluationError,
    DensityRootFailure,
    _DensityRootEvaluation,
    _findDensityRootEvaluations,
)


_ILR_COORDINATE_BOUND = 80.0


@dataclass(frozen=True)
class TPDWitness:
    """
    A feasible TPD trial, without a stationarity or minimum claim.

    Composition and value are dimensionless; density is mol/L, pressure
    residual is bar, and its density derivative is bar L/mol. The producing
    root has passed model-domain, pressure, finite chemical-potential and
    mechanical-slope checks. A full component Hessian is not required.
    A sufficiently negative value proves instability (Michelsen, 1982).
    """

    composition: tuple[float, ...]
    density: float
    value: float
    pressureResidual: float
    pressureDensityDerivative: float

    def asDict(self) -> dict[str, object]:
        """
        Return a pressure-qualified trial separately from a TPD minimum.
        """

        return {
            "composition": list(self.composition), "densityMolL": self.density,
            "tpdValue": self.value, "pressureResidualBar": self.pressureResidual,
            "pressureDensityDerivativeBarLPerMol": self.pressureDensityDerivative,
        }


@dataclass(frozen=True)
class TPDDensityFailure:
    """
    Unresolved density candidates at one dimensionless trial composition.
    """

    composition: tuple[float, ...]
    candidates: tuple[DensityRootFailure, ...]

    def asDict(self) -> dict[str, object]:
        """
        Expose the affected composition and each rejected density candidate.
        """

        return {
            "composition": list(self.composition),
            "candidates": [item.asDict() for item in self.candidates],
        }


@dataclass
class _TPDEvidence:
    """
    Retain qualified TPD trials even if a later iteration fails.
    """

    witness: TPDWitness | None = None
    densityFailures: list[TPDDensityFailure] = field(default_factory=list)
    stopBelow: float | None = None

    def observe(
        self, model: FreeEnergyModel, temperature: float, fractions: np.ndarray,
        constant: np.ndarray, evaluation: _DensityRootEvaluation,
    ) -> None:
        """
        Retain the lowest finite TPD at an already qualified stable root.
        """

        root = evaluation.root
        value = float(fractions @ (
            evaluation.chemicalPotentials / (model.gasConstant * temperature) - constant
        ))
        if not math.isfinite(value):
            raise FloatingPointError("qualified trial TPD is nonfinite")
        if root.mechanicalStable and (self.witness is None or value < self.witness.value):
            self.witness = TPDWitness(
                tuple(float(item) for item in fractions), root.density, value,
                root.pressureResidual, root.pressureDensityDerivative,
            )

            # Qualified negative evidence suffices for instability; it need
            # not satisfy the optimizer's stationarity/curvature conditions.
            if self.stopBelow is not None and value < self.stopBelow:
                raise _TPDWitnessFound


class _TPDWitnessFound(Exception):
    """
    End a requested instability screen after retaining a qualified witness.

    This is intentional termination, not an infeasible optimizer trial. It
    must bypass the numerical-error handlers and be caught by the multistart
    driver, which retains any independent density failures already observed.
    """


@dataclass(frozen=True)
class TPDMinimum:
    """
    One local TPD search result at fixed temperature and pressure.

    ``composition`` contains dimensionless mole fractions, ``density`` is in
    mol/L and ``value`` is dimensionless. ``iterations`` and ``evaluations``
    retain the numerical work counters. For composition-NPT and extended
    molar-NPT TPD, ``stationarityNorm`` and ``minimumHessianEigenvalue`` are
    fixed-pressure log-ratio gradient and curvature. Molar-NPT additionally requires
    independent pressure acceptance; its reported value remains the original
    TPD expression, not the off-pressure optimization objective.
    The latter two are ``None`` for a single-component composition identity.
    ``converged=False`` retains the final inspected state and termination
    reason but does not represent a usable stationary basin.
    ``witness`` retains the lowest qualified trial independently. Any
    ``densityFailures`` leave root selection unresolved and force
    ``converged=False``, even if numerical stationarity was reached.
    """

    composition: tuple[float, ...]
    density: float
    value: float
    converged: bool
    evaluations: int
    iterations: int
    stationarityNorm: float | None
    minimumHessianEigenvalue: float | None
    termination: str = ""
    witness: TPDWitness | None = None
    densityFailures: tuple[TPDDensityFailure, ...] = ()

    def asDict(self) -> dict[str, object]:
        """
        Return a JSON-ready record with density in mol/L.
        """

        return {
            "composition": list(self.composition),
            "densityMolL": self.density,
            "tpdValue": self.value,
            "converged": self.converged,
            "evaluations": self.evaluations,
            "iterations": self.iterations,
            "stationarityNorm": self.stationarityNorm,
            "minimumHessianEigenvalue": self.minimumHessianEigenvalue,
            "termination": self.termination,
            "witness": None if self.witness is None else self.witness.asDict(),
            "densityFailures": [item.asDict() for item in self.densityFailures],
        }


def _softmax(values: np.ndarray) -> np.ndarray:
    """
    Return a normalized composition from unconstrained logits.
    """

    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(np.clip(shifted, -700.0, 0.0))
    return weights / np.sum(weights)


def _helmertBasis(componentCount: int) -> np.ndarray:
    r"""
    Return a Helmert basis for $\mathbf 1^\perp$ in composition space.

    One-based column $j$ contains $1/\sqrt{j(j+1)}$ in rows $1,\ldots,j$,
    $-j/\sqrt{j(j+1)}$ in row $j+1$, and zero below it. Thus the columns are
    orthonormal and every column sums to zero.
    """

    if componentCount < 2:
        raise ValueError("ILR coordinates require at least two components")
    basis = np.zeros((componentCount, componentCount - 1), dtype=float)
    for column in range(componentCount - 1):
        scale = 1.0 / math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = scale
        basis[column + 1, column] = -(column + 1) * scale
    return basis


def _compositionFromIlr(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    r"""
    Map $N-1$ ILR coordinates to an $N$-component composition.
    """

    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or basis.shape != (
        coordinates.size + 1,
        coordinates.size,
    ):
        raise ValueError("TPD ILR coordinates and basis are inconsistent")
    return _softmax(basis @ coordinates)


def _ilrCoordinates(fractions: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Return orthonormal ILR coordinates for a positive composition.
    """

    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or basis.shape != (
        fractions.size,
        fractions.size - 1,
    ):
        raise ValueError("TPD composition and ILR basis are inconsistent")
    return basis.T @ np.log(fractions)


def _selectRoot(
    roots: tuple[DensityRoot, ...],
    phase: str,
) -> DensityRoot | None:
    """
    Select among mechanically stable roots ordered by increasing density.
    "vapor" selects the lowest density and "liquid" the highest. "stable" selects
    the smallest sum(y_i*ln(phi_i)), which ranks molar Gibbs energy at fixed T/P/y.
    """

    mechanicallyStable = tuple(root for root in roots if root.mechanicalStable)
    if not mechanicallyStable:
        return None
    if phase == "vapor":
        return mechanicallyStable[0]
    if phase == "liquid":
        return mechanicallyStable[-1]
    if phase == "stable":
        return min(mechanicallyStable, key=lambda item: item.gibbsLike)
    raise ValueError(f"unknown phase selector: {phase}")


def _selectRootEvaluation(
    evaluations: tuple[_DensityRootEvaluation, ...],
    phase: str,
) -> _DensityRootEvaluation | None:
    """
    Select a root together with the local derivatives already evaluated.
    """

    root = _selectRoot(
        tuple(evaluation.root for evaluation in evaluations),
        phase,
    )
    if root is None:
        return None
    return next(evaluation for evaluation in evaluations if evaluation.root is root)


def _trialDerivatives(
    model: FreeEnergyModel,
    kernel: ThermoKernel,
    temperature: float,
    pressure: float,
    fractions: np.ndarray,
    constant: np.ndarray,
    *,
    evidence: _TPDEvidence | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Evaluate a qualified branch and explicitly retain incomplete selection.

    Without a result collector an incomplete root set raises. With
    one, a valid branch remains usable for exploration, while its unresolved
    competitors prevent declaring this local search fully converged.
    """

    try:
        evaluations = _findDensityRootEvaluations(
            model, temperature, pressure, fractions,
            scanPointCount=DEFAULT_SCAN_POINT_COUNT, kernel=kernel,
            withHessian=False,
        )
    except DensityRootEvaluationError as error:
        if evidence is None:
            raise
        evidence.densityFailures.append(TPDDensityFailure(
            tuple(float(value) for value in fractions), error.failures,
        ))
        evaluations = error.evaluations
    evaluation = _selectRootEvaluation(evaluations, "stable")
    if evaluation is None:
        raise FloatingPointError("trial composition has no density root")
    if evidence is not None:
        evidence.observe(model, temperature, fractions, constant, evaluation)
    if evaluation.hessian is None:

        # A negative screen has already returned. Only the selected branch
        # now needs a component Hessian for the trust-region model.
        try:
            hessian = kernel.hessian(
                temperature, evaluation.root.density * fractions, validate=False,
            ).detach().cpu().numpy()
            if not np.all(np.isfinite(hessian)) or float(fractions @ hessian @ fractions) <= 0.0:
                raise FloatingPointError("selected density-root Hessian is nonfinite or mechanically unstable")
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            if evidence is not None:
                evidence.densityFailures.append(TPDDensityFailure(
                    tuple(float(value) for value in fractions),
                    (DensityRootFailure(evaluation.root.density, f"{type(error).__name__}: {error}"),),
                ))
            raise
        evaluation = _DensityRootEvaluation(evaluation.root, evaluation.chemicalPotentials, hessian)
    return _trialDerivativesFromEvaluation(
        model,
        temperature,
        fractions,
        constant,
        evaluation,
    )


def _trialDerivativesFromEvaluation(
    model: FreeEnergyModel,
    temperature: float,
    fractions: np.ndarray,
    constant: np.ndarray,
    evaluation: _DensityRootEvaluation,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Calculate dimensionless TPD, composition derivatives and density [mol/L].
    At a pressure-matching root, g = mu/(RT) - b and D = y.T @ g.
    With H = d(mu)/d(rho_i), the fixed-pressure Hessian is
    H_D = rho/(RT) * (H - (H@y)(H@y).T / (y.T@H@y)).
    Eliminating total density enforces the pressure constraint (Michelsen, 1982).
    The root must supply its full component-density Hessian.
    """

    root = evaluation.root
    chemicalPotentials = evaluation.chemicalPotentials
    hessian = evaluation.hessian
    if hessian is None:
        raise ValueError("TPD derivative assembly requires a selected-root Hessian")
    reduced = chemicalPotentials / (model.gasConstant * temperature) - constant
    value = float(np.dot(fractions, reduced))
    gradient = reduced.copy()
    hessianVector = hessian @ fractions
    quadratic = float(fractions @ hessianVector)
    if not math.isfinite(quadratic) or abs(quadratic) <= np.finfo(float).tiny:
        raise FloatingPointError("trial stability quadratic is degenerate")
    hessianOfD = (
        root.density
        / (model.gasConstant * temperature)
        * (
            hessian
            - np.outer(hessianVector, hessianVector) / quadratic
        )
    )
    return value, gradient, hessianOfD, root.density


def _ilrProjected(
    gradient: np.ndarray,
    hessian: np.ndarray,
    fractions: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project exact composition derivatives onto orthonormal ILR coordinates.

    With ``y = softmax(Q u)``, the Jacobian is
    ``(diag(y) - y y^T) Q``.  The projected Hessian includes the exact second
    derivative of this map weighted by the composition-space gradient.
    """

    projectedFractions = basis.T @ fractions
    jacobian = np.diag(fractions) @ basis - np.outer(
        fractions,
        projectedFractions,
    )
    projectedHessian = jacobian.T @ hessian @ jacobian
    weightedGradient = gradient * fractions
    weightedSum = float(np.sum(weightedGradient))
    projectedWeightedGradient = basis.T @ weightedGradient
    diagonalCurvature = weightedGradient - weightedSum * fractions
    second = (
        basis.T @ (diagonalCurvature[:, None] * basis)
        - np.outer(projectedWeightedGradient, projectedFractions)
        - np.outer(projectedFractions, projectedWeightedGradient)
        + 2.0
        * weightedSum
        * np.outer(projectedFractions, projectedFractions)
    )
    return jacobian.T @ gradient, projectedHessian + second


def _minimizeTpdPrepared(
    model: FreeEnergyModel,
    feed: np.ndarray,
    temperature: float,
    pressure: float,
    start: np.ndarray,
    referenceLnPhi: np.ndarray,
    kernel: ThermoKernel,
    basis: np.ndarray | None,
    config: ScalarTrustRegionConfig | None,
    initialEvaluation: _DensityRootEvaluation | None = None,
    *,
    evidence: _TPDEvidence | None = None,
) -> TPDMinimum:
    """
    Minimize one validated start while retaining accepted-state density.

    The caller owns feed/start normalization, reference-state consistency,
    and multistart failure reporting. ``initialEvaluation`` is valid only for
    the initial coordinates of the same stable reference composition. Model
    domain errors become infeasible objective trials; an invalid initial
    state or an unevaluated final state is reported explicitly.
    """

    evidence = evidence if evidence is not None else _TPDEvidence()
    constant = np.log(pressure / (model.gasConstant * temperature)) + np.log(
        feed
    ) + referenceLnPhi

    if feed.size == 1:
        if initialEvaluation is None:
            finalValue, _gradient, _hessian, finalDensity = _trialDerivatives(
                model,
                kernel,
                temperature,
                pressure,
                np.ones(1, dtype=float),
                constant,
                evidence=evidence,
            )
        else:
            evidence.observe(model, temperature, np.ones(1), constant, initialEvaluation)
            finalValue, _gradient, _hessian, finalDensity = (
                _trialDerivativesFromEvaluation(
                    model,
                    temperature,
                    np.ones(1, dtype=float),
                    constant,
                    initialEvaluation,
                )
            )
        return TPDMinimum(
            composition=(1.0,),
            density=finalDensity,
            value=float(finalValue),
            converged=not evidence.densityFailures,
            evaluations=1,
            iterations=0,
            stationarityNorm=None,
            minimumHessianEigenvalue=None,
            termination=("unresolved_density_candidates" if evidence.densityFailures
                         else "single_component_identity"),
            witness=evidence.witness,
            densityFailures=tuple(evidence.densityFailures),
        )

    if basis is None:
        raise ValueError("multicomponent TPD requires an ILR basis")
    initialCoordinates = _ilrCoordinates(start, basis)
    initialKey = initialCoordinates.tobytes()
    evaluatedDensities: dict[bytes, float] = {}

    def objective(values: np.ndarray):
        """
        Return the dimensionless TPD value, ILR gradient and Hessian.
        """

        coordinateValues = np.asarray(values, dtype=float)
        coordinateKey = coordinateValues.tobytes()
        fractions = _compositionFromIlr(coordinateValues, basis)
        try:
            if initialEvaluation is not None and coordinateKey == initialKey:
                evidence.observe(model, temperature, fractions, constant, initialEvaluation)
                value, gradient, hessian, density = (
                    _trialDerivativesFromEvaluation(
                        model,
                        temperature,
                        fractions,
                        constant,
                        initialEvaluation,
                    )
                )
            else:
                value, gradient, hessian, density = _trialDerivatives(
                    model,
                    kernel,
                    temperature,
                    pressure,
                    fractions,
                    constant,
                    evidence=evidence,
                )
            projectedGradient, projectedHessian = _ilrProjected(
                gradient,
                hessian,
                fractions,
                basis,
            )
            if (
                math.isfinite(value)
                and np.all(np.isfinite(projectedGradient))
                and np.all(np.isfinite(projectedHessian))
            ):

                # The optimizer returns an evaluated current or qualified trial
                # state. Retaining its scalar density avoids rebuilding the
                # complete thermodynamic state after convergence.
                evaluatedDensities[coordinateKey] = float(density)
                return value, projectedGradient, projectedHessian
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        return None

    result = minimizeScalar(
        objective,
        initialCoordinates,
        np.full(initialCoordinates.size, -_ILR_COORDINATE_BOUND),
        np.full(initialCoordinates.size, _ILR_COORDINATE_BOUND),
        config=config,
    )
    final = _compositionFromIlr(result.state, basis)
    finalKey = result.state.tobytes()
    if finalKey not in evaluatedDensities:
        raise RuntimeError("TPD optimizer returned an unevaluated final state")
    return TPDMinimum(
        composition=tuple(float(value) for value in final),
        density=evaluatedDensities[finalKey],
        value=float(result.value),
        converged=bool(result.converged) and not evidence.densityFailures,
        evaluations=int(result.evaluations),
        iterations=int(result.iterations),
        stationarityNorm=float(result.stationarityNorm),
        minimumHessianEigenvalue=float(result.lambdaMin),
        termination=(
            result.termination + "; unresolved_density_candidates"
            if evidence.densityFailures else result.termination
        ),
        witness=evidence.witness,
        densityFailures=tuple(evidence.densityFailures),
    )


__all__ = ["TPDMinimum", "TPDWitness", "TPDDensityFailure"]
