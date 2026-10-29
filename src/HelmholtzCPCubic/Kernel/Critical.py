r"""
Evaluate critical conditions from Helmholtz component-density derivatives.
Return dimensionless stability and third-order residuals, and their derivatives
with respect to logarithmic temperature and density.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..Common.Model import FreeEnergyModel, TensorLike
from .Thermo import ThermoKernel


@dataclass
class DensityCriticalState:
    """
    dimensionless stability matrix, eigensystem and third-order residual.
    """

    stabilityMatrix: torch.Tensor
    eigenvalues: torch.Tensor
    minEigenvector: torch.Tensor
    qMin: torch.Tensor
    cStar: torch.Tensor
    densities: torch.Tensor


@dataclass(frozen=True)
class DensityStabilityState:
    """
    Stability state before the third-directional CP condition.

    Retained instances are detached from the AD graph so an accepted spinodal
    contour point can reuse its stability matrix and minimum eigendirection
    without keeping an iteration graph alive.
    """

    temperature: torch.Tensor
    totalDensity: torch.Tensor
    stabilityMatrix: torch.Tensor
    eigenvalues: torch.Tensor
    minEigenvector: torch.Tensor
    densities: torch.Tensor

    @property
    def qMin(self) -> torch.Tensor:
        """
        Return the minimum dimensionless stability eigenvalue.
        """

        return self.eigenvalues[0]


@dataclass(frozen=True)
class DensityStabilityEvaluation:
    """
    One log-state stability residual/Jacobian and its retained state.
    """

    logState: torch.Tensor
    stability: DensityStabilityState
    jacobian: torch.Tensor

    @property
    def residual(self) -> torch.Tensor:
        r"""
        Return the one-component $q_{\min}$ contour residual.
        """

        return self.stability.qMin.reshape(1)


@dataclass(frozen=True)
class DensityCriticalEvaluation:
    """
    One log-state critical residual/Jacobian and its detached state.
    """

    logState: torch.Tensor
    critical: DensityCriticalState
    jacobian: torch.Tensor

    @property
    def residual(self) -> torch.Tensor:
        r"""
        Return the two-component $[q^*,C^*]$ critical residual.
        """

        return torch.stack((self.critical.qMin, self.critical.cStar))


class DensityCriticalKernel:
    r"""
    Build the dimensionless $[q^*,C^*]$ ordinary-CP conditions.
    """

    def __init__(self, model: FreeEnergyModel) -> None:
        self.model = model
        self.thermo = ThermoKernel(model)

    def evaluate(
        self,
        T: TensorLike,
        rho: TensorLike,
        composition: TensorLike,
        *,
        validate: bool = True,
        createGraph: bool = False,
    ) -> DensityCriticalState:
        """
        Evaluate dimensionless conditions at one physical state.
        """

        Tt = self.model.asTensor(T)
        rhoT = self.model.asTensor(rho)
        x = self._composition(composition)
        if Tt.ndim != 0 or rhoT.ndim != 0:
            raise ValueError("T and rho must be scalars")
        densities = rhoT * x
        if validate:
            self.model.validateState(Tt, densities)

        stabilityState = self._stabilityState(
            Tt,
            rhoT,
            densities,
        )
        return self.criticalStateFromStability(
            stabilityState,
            createGraph=createGraph,
        )

    def criticalStateFromStability(
        self,
        stability: DensityStabilityState,
        *,
        createGraph: bool = False,
    ) -> DensityCriticalState:
        """
        Complete a stability state with the third CP condition.

        Spinodal correction already computes the stability matrix and
        minimum eigendirection. Completing an accepted state avoids a second
        Hessian/eigendecomposition while retaining the full $C^*$ evaluation.
        """

        third = self._thirdDirectional(
            stability.temperature,
            stability.densities,
            stability.minEigenvector,
            createGraph=createGraph,
        )
        thirdScale = stability.totalDensity**2 / (
            self.model.asTensor(self.model.gasConstant)
            * stability.temperature
        )
        return DensityCriticalState(
            stabilityMatrix=stability.stabilityMatrix,
            eigenvalues=stability.eigenvalues,
            minEigenvector=stability.minEigenvector,
            qMin=stability.qMin,
            cStar=thirdScale * third,
            densities=stability.densities,
        )

    def stabilityResidual(
        self,
        logState: TensorLike,
        composition: TensorLike,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        r"""
        Return only $q_{\min}$ at $[\ln T,\ln\rho]$.

        Spinodal correction needs the stability boundary but not the
        fourth-order derivatives required by the full critical Jacobian.
        """

        state = self.model.asTensor(logState)
        if state.shape != (2,):
            raise ValueError("logState must have shape (2,)")
        Tt, rhoT = torch.exp(state[0]), torch.exp(state[1])
        x = self._composition(composition)
        densities = rhoT * x
        if validate:
            self.model.validateState(Tt, densities)
        _stability, eigenvalues, _eigenvector = self._stabilityComponents(
            Tt,
            rhoT,
            densities,
        )
        return eigenvalues[0]

    def stabilityJacobian(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Return $q_{\min}$ and its $1\times2$ log-state Jacobian.
        """

        evaluation = self.stabilityEvaluation(logState, composition)
        return evaluation.residual, evaluation.jacobian

    def stabilityEvaluation(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> DensityStabilityEvaluation:
        r"""
        Return $q_{\min}$, its Jacobian, and detached stability-state data.
        """

        state = (
            self.model.asTensor(logState)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        Tt, rhoT = torch.exp(state[0]), torch.exp(state[1])
        x = self._composition(composition)
        densities = rhoT * x
        self.model.validateState(Tt, densities)
        stability = self._stabilityState(Tt, rhoT, densities)
        gradient = torch.autograd.grad(stability.qMin, state)[0]
        return DensityStabilityEvaluation(
            logState=state.detach(),
            stability=self._detachedStabilityState(stability),
            jacobian=gradient.detach().reshape(1, 2),
        )

    def residual(
        self,
        logState: TensorLike,
        composition: TensorLike,
        *,
        validate: bool = True,
        createGraph: bool = False,
    ) -> torch.Tensor:
        r"""
        Return $[q^*,C^*]$ at $[\ln T,\ln\rho]$.
        """

        state = self.model.asTensor(logState)
        if state.shape != (2,):
            raise ValueError("logState must have shape (2,)")
        critical = self.evaluate(
            torch.exp(state[0]),
            torch.exp(state[1]),
            composition,
            validate=validate,
            createGraph=createGraph,
        )
        return torch.stack((critical.qMin, critical.cStar))

    def jacobian(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return residual and its 2-by-2 Jacobian by automatic differentiation.
        """

        evaluation = self.criticalEvaluation(logState, composition)
        return evaluation.residual, evaluation.jacobian

    def criticalEvaluation(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalEvaluation:
        """
        Return a critical residual/Jacobian and detached state together.
        """

        state = (
            self.model.asTensor(logState)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        critical = self.evaluate(
            torch.exp(state[0]),
            torch.exp(state[1]),
            composition,
            createGraph=True,
        )
        residual = torch.stack((critical.qMin, critical.cStar))

        # The identity basis requests both residual VJPs in one engine call.
        # Both rows retain the complete higher-order derivative dependencies.
        jacobian = torch.autograd.grad(
            residual,
            state,
            grad_outputs=torch.eye(
                residual.numel(),
                dtype=state.dtype,
                device=state.device,
            ),
            is_grads_batched=True,
        )[0]
        return DensityCriticalEvaluation(
            logState=state.detach(),
            critical=self._detachedCriticalState(critical),
            jacobian=jacobian.detach(),
        )

    def locusResidual(
        self,
        state: TensorLike,
        *,
        validate: bool = True,
        createGraph: bool = False,
    ) -> torch.Tensor:
        r"""
        Return binary $[q^*,C^*]$ at $[\ln T,\ln\rho,x_1]$.
        """

        if self.model.componentCount != 2:
            raise ValueError("critical-locus residual requires a binary model")
        u = self.model.asTensor(state)
        if u.shape != (3,):
            raise ValueError("state must have shape (3,)")
        x = torch.stack((u[2], 1.0 - u[2]))
        return self.residual(
            u[:2],
            x,
            validate=validate,
            createGraph=createGraph,
        )

    def locusJacobian(
        self,
        state: TensorLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return binary residual and its 2-by-3 Jacobian by automatic differentiation.
        """

        u = self.model.asTensor(state).detach().clone().requires_grad_(True)
        residual = self.locusResidual(u, createGraph=True)
        rows = []
        for index in range(2):
            rows.append(
                torch.autograd.grad(
                    residual[index],
                    u,
                    retain_graph=index == 0,
                )[0]
            )
        return residual.detach(), torch.stack(rows).detach()

    def _thirdDirectional(
        self,
        T: torch.Tensor,
        densities: torch.Tensor,
        direction: torch.Tensor,
        *,
        createGraph: bool,
    ) -> torch.Tensor:
        """
        Contract the third density derivative without materializing its tensor.
        """

        step = torch.zeros(
            (),
            dtype=self.model.dtype,
            device=self.model.device,
            requires_grad=True,
        )
        energy = self.model.freeEnergy(T, densities + step * direction)
        first = torch.autograd.grad(energy, step, create_graph=True)[0]
        second = torch.autograd.grad(first, step, create_graph=True)[0]
        return torch.autograd.grad(
            second,
            step,
            create_graph=createGraph,
        )[0]

    def _stabilityComponents(
        self,
        T: torch.Tensor,
        rho: torch.Tensor,
        densities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the dimensionless matrix and its minimum eigendirection.
        """

        stability = self._stabilityMatrix(T, rho, densities)
        eigenvalues, eigenvectors = torch.linalg.eigh(stability)
        minEigenvector = self._canonicalVector(eigenvectors[:, 0])
        return stability, eigenvalues, minEigenvector

    def _stabilityState(
        self,
        T: torch.Tensor,
        rho: torch.Tensor,
        densities: torch.Tensor,
    ) -> DensityStabilityState:
        """
        Return one graph-preserving stability evaluation.
        """

        stability, eigenvalues, minEigenvector = self._stabilityComponents(
            T,
            rho,
            densities,
        )
        return DensityStabilityState(
            temperature=T,
            totalDensity=rho,
            stabilityMatrix=stability,
            eigenvalues=eigenvalues,
            minEigenvector=minEigenvector,
            densities=densities,
        )

    @staticmethod
    def _detachedStabilityState(
        stability: DensityStabilityState,
    ) -> DensityStabilityState:
        """
        Detach retained stability data from one completed AD evaluation.
        """

        return DensityStabilityState(
            temperature=stability.temperature.detach(),
            totalDensity=stability.totalDensity.detach(),
            stabilityMatrix=stability.stabilityMatrix.detach(),
            eigenvalues=stability.eigenvalues.detach(),
            minEigenvector=stability.minEigenvector.detach(),
            densities=stability.densities.detach(),
        )

    @staticmethod
    def _detachedCriticalState(
        critical: DensityCriticalState,
    ) -> DensityCriticalState:
        """
        Detach a completed critical state from its high-order AD graph.
        """

        return DensityCriticalState(
            stabilityMatrix=critical.stabilityMatrix.detach(),
            eigenvalues=critical.eigenvalues.detach(),
            minEigenvector=critical.minEigenvector.detach(),
            qMin=critical.qMin.detach(),
            cStar=critical.cStar.detach(),
            densities=critical.densities.detach(),
        )

    def _stabilityMatrix(
        self,
        T: torch.Tensor,
        rho: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the symmetric dimensionless density stability matrix.
        """

        hessian = self.thermo.hessian(
            T,
            densities,
            validate=False,
        )
        scale = rho / (
            self.model.asTensor(self.model.gasConstant) * T
        )
        stability = scale * hessian
        return 0.5 * (stability + stability.mT)

    def _composition(self, composition: TensorLike) -> torch.Tensor:
        return self.model._normalizedComposition(composition)

    @staticmethod
    def _canonicalVector(vector: torch.Tensor) -> torch.Tensor:
        """
        Choose a deterministic minEigenvector sign for smooth local solving.
        """

        normalized = vector / torch.linalg.vector_norm(vector)
        pivot = torch.argmax(torch.abs(normalized))
        return torch.where(normalized[pivot] < 0.0, -normalized, normalized)


__all__ = [
    "DensityCriticalKernel",
    "DensityCriticalState",
]
