r"""
Evaluate PR/SRK critical conditions and their Jacobian analytically.
Use the same dimensionless residuals as the automatic-differentiation kernel.
Follow Heidemann and Khalil (1980) and Bell and Jäger (2017).
Inputs use K and mol/L; Jacobians use logarithmic temperature and density.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..Common.Composition import normalizeComposition
from ..Common.Model import TensorLike
from ..Transform.CubicBell import CubicBellModel, EOS_PR, EOS_SRK, cubicConstants
from .Critical import DensityCriticalEvaluation, DensityCriticalState


_EIGEN_GAP_TOL = 1.0e-11


class CubicCriticalKernel:
    """
    Calculate analytic PR/SRK critical residuals and log-state Jacobians.
    Support classic CubicBellModel with constant symmetric kij, the implemented
    PR/SRK temperature correlations, CPU execution and float64 arithmetic.
    Other formulations use the automatic-differentiation backend.
    """

    name = "cubic_analytic"

    def __init__(self, model: CubicBellModel) -> None:
        reason = self.unsupportedReason(model)
        if reason is not None:
            raise ValueError(reason)
        self.model = model
        self._Tc = np.asarray(model.pureCriticalTemp, dtype=float)
        self._Pc = np.asarray(model.pureCriticalPres, dtype=float)
        self._omega = np.asarray(model.pureAcentricFactor, dtype=float)
        self._kij = np.asarray(model.cubicKij, dtype=float)
        self._R = float(model.gasConstant)
        self._mode = int(model.mode)
        self._omegaA, self._omegaB, self._delta1, self._delta2 = cubicConstants(
            self._mode
        )
        self._m = self._alphaParameter(self._mode, self._omega)
        self._bi = self._omegaB * self._R * self._Tc / self._Pc

    @staticmethod
    def unsupportedReason(model: object) -> str | None:
        """
        Return why the model is unsupported, or None when analytic evaluation is available.
        """

        if type(model) is not CubicBellModel:
            return (
                f"{type(model).__name__} has no supported analytic critical "
                "backend"
            )
        if model.device.type != "cpu":
            return "cubic analytic critical backend currently requires a CPU model"
        if model.dtype != torch.float64:
            return "cubic analytic critical backend currently requires float64"
        if model.mode not in (EOS_SRK, EOS_PR):
            return "cubic analytic critical backend supports only classic PR and SRK"
        return None

    def evaluate(
        self,
        T: TensorLike,
        rho: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalState:
        """
        Evaluate analytic dimensionless critical conditions at one state.

        Args:
            T: Temperature in K.
            rho: Total molar density in mol/L.
            composition: Positive active-component mole fractions.

        Returns:
            Critical state with dimensionless residuals and eigensystem.
        """

        temp = self._scalar(T, "T")
        density = self._scalar(rho, "rho")
        x = normalizeComposition(
            np.asarray(self.model.asTensor(composition).detach().cpu(), dtype=float),
            self.model.componentCount,
        )
        return self._evaluation(temp, density, x, needJacobian=False)[0]

    def criticalEvaluation(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalEvaluation:
        """
        Return analytic residual, Jacobian and detached critical-state data.
        """

        state = np.asarray(
            self.model.asTensor(logState).detach().cpu(),
            dtype=float,
        )
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError("logState must be a finite vector with shape (2,)")
        T, rho = np.exp(state)
        x = normalizeComposition(
            np.asarray(self.model.asTensor(composition).detach().cpu(), dtype=float),
            self.model.componentCount,
        )
        critical, jacobian = self._evaluation(T, rho, x, needJacobian=True)
        return DensityCriticalEvaluation(
            logState=self.model.asTensor(state).detach(),
            critical=critical,
            jacobian=self.model.asTensor(jacobian).detach(),
        )

    def pressure(self, T: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Return direct cubic pressure in bar without an AD evaluation.
        """

        self.model.validateState(T, densities)
        pressure = self.model.directPres(T, densities)
        self.model.validatePres(pressure)
        return pressure.detach()

    def _evaluation(
        self,
        T: float,
        rho: float,
        x: np.ndarray,
        *,
        needJacobian: bool,
    ) -> tuple[DensityCriticalState, np.ndarray]:
        """
        Assemble one analytic physical-state evaluation.
        """

        if not (math.isfinite(T) and math.isfinite(rho) and T > 0.0 and rho > 0.0):
            raise ValueError("T and rho must be finite and positive")
        densities = rho * x
        if self.model.componentCount != x.size:
            raise ValueError("composition size does not match cubic model")
        self.model.validateState(T, densities)
        context = self._conditions(T, rho, x)

        scale = self._R * T
        qMin = context["q"] / scale
        cStar = context["c"] / scale
        eigenvalues = context["eigenvalues"] / scale
        stability = context["Q"] / scale
        critical = DensityCriticalState(
            stabilityMatrix=self.model.asTensor(stability).detach(),
            eigenvalues=self.model.asTensor(eigenvalues).detach(),
            minEigenvector=self.model.asTensor(context["direction"]).detach(),
            qMin=self.model.asTensor(qMin).detach(),
            cStar=self.model.asTensor(cStar).detach(),
            densities=self.model.asTensor(densities).detach(),
        )
        if not needJacobian:
            return critical, np.empty((0, 0), dtype=float)

        v = 1.0 / rho
        rawJacobian = context["jacobian"]
        jacobian = np.array(
            [
                [
                    (T * rawJacobian[0, 1] - context["q"]) / scale,
                    -v * rawJacobian[0, 0] / scale,
                ],
                [
                    (T * rawJacobian[1, 1] - context["c"]) / scale,
                    -v * rawJacobian[1, 0] / scale,
                ],
            ],
            dtype=float,
        )
        return critical, jacobian

    def _conditions(self, T: float, rho: float, x: np.ndarray) -> dict[str, Any]:
        """
        Return dimensional conditions and derivatives in ``(v, T)``.
        """

        v = 1.0 / rho
        sqrtT = math.sqrt(T)
        sqrtTr = sqrtT / np.sqrt(self._Tc)
        alphaT = (1.0 + self._m * (1.0 - sqrtTr)) ** 2
        aii = self._omegaA * self._R**2 * self._Tc**2 / self._Pc * alphaT
        aij = (1.0 - self._kij) * np.sqrt(np.outer(aii, aii))
        a = float(x @ aij @ x)
        b = float(x @ self._bi)
        K = v / b
        if not (math.isfinite(K) and K > 1.0):
            raise ValueError("cubic analytic state is at or beyond v=b")

        beta = self._bi / b
        alpha = aij @ x / a
        values, derivatives = self._volumeFunctions(K)
        F1, _F2, F3, F4, F5, F6 = values
        F1v, _F2v, F3v, F4v, F5v, F6v = (
            value / b for value in derivatives
        )

        betaPair = beta[:, None] + beta[None, :]
        betaOuter = np.outer(beta, beta)
        alphaBeta = np.outer(alpha, beta) + np.outer(beta, alpha)
        Q = self._R * T * (
            np.diag(1.0 / x) + betaPair * F1 + betaOuter * F1**2
        ) + (a / b) * (
            betaOuter * F3 - (aij / a) * F5 + (betaOuter - alphaBeta) * F6
        )
        Q = 0.5 * (Q + Q.T)
        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        direction = self._canonicalVector(eigenvectors[:, 0])
        eigenvectors[:, 0] = direction

        nBar = float(np.sum(direction))
        bBar = float(direction @ beta)
        aBar = float(direction @ alpha)
        aaBar = float(direction @ (aij / a) @ direction)
        sumDn3x2 = float(np.sum(direction**3 / x**2))
        c = self._R * T * (
            -sumDn3x2
            + 3.0 * nBar * (bBar * F1) ** 2
            + 2.0 * (bBar * F1) ** 3
        ) + (a / b) * (
            3.0 * bBar**2 * (2.0 * aBar - bBar) * (F3 + F6)
            - 2.0 * bBar**3 * F4
            - 3.0 * bBar * aaBar * F6
        )

        dQdvMatrix = self._R * T * F1v * (
            betaPair + 2.0 * betaOuter * F1
        ) + (a / b) * (
            betaOuter * (F3v + F6v)
            - (aij / a) * F5v
            - alphaBeta * F6v
        )

        psi = self._m / np.sqrt(self._Tc) / (
            1.0 + self._m * (1.0 - sqrtTr)
        )
        daijDt = -aij / (2.0 * sqrtT) * (psi[:, None] + psi[None, :])
        sumXdaij = daijDt @ x
        dadt = float(x @ sumXdaij)
        dQdtMatrix = self._R * (
            np.diag(1.0 / x) + betaPair * F1 + betaOuter * F1**2
        ) + (1.0 / b) * (
            betaOuter * (F3 + F6) * dadt
            - F5 * daijDt
            - (np.outer(sumXdaij, beta) + np.outer(beta, sumXdaij)) * F6
        )

        dQdv = float(direction @ dQdvMatrix @ direction)
        dQdt = float(direction @ dQdtMatrix @ direction)
        dDndv = self._directionDerivative(
            eigenvalues,
            eigenvectors,
            direction,
            dQdvMatrix,
        )
        dDndt = self._directionDerivative(
            eigenvalues,
            eigenvectors,
            direction,
            dQdtMatrix,
        )

        dCdvFixed = 6.0 * self._R * T * bBar**2 * F1 * F1v * (
            nBar + bBar * F1
        ) + (a / b) * (
            3.0 * bBar**2 * (2.0 * aBar - bBar) * (F3v + F6v)
            - 2.0 * bBar**3 * F4v
            - 3.0 * bBar * aaBar * F6v
        )

        temperaturePartC = (
            -sumDn3x2
            + 3.0 * nBar * (bBar * F1) ** 2
            + 2.0 * (bBar * F1) ** 3
        )
        attractionPartC = (
            3.0 * bBar**2 * (2.0 * aBar - bBar) * (F3 + F6)
            - 2.0 * bBar**3 * F4
            - 3.0 * bBar * aaBar * F6
        )
        sumNdaij = float(direction @ sumXdaij)
        sumNNdaij = float(direction @ daijDt @ direction)
        daaBardt = -aaBar / a * dadt + sumNNdaij / a
        daBardt = -aBar / a * dadt + sumNdaij / a
        dAttractionCdt = (
            6.0 * bBar**2 * daBardt * (F3 + F6)
            - 3.0 * bBar * daaBardt * F6
        )
        dCdtFixed = (
            self._R * temperaturePartC
            + (dadt / b) * attractionPartC
            + (a / b) * dAttractionCdt
        )

        aiBar = (aij @ direction) / a
        derivativeIdeal = (
            -3.0 * direction**2 / x**2
            + 3.0 * bBar**2 * F1**2
            + 6.0 * nBar * bBar * beta * F1**2
            + 6.0 * bBar**2 * beta * F1**3
        )
        derivativeAttraction = (
            6.0 * bBar * beta * (2.0 * aBar - bBar) * (F3 + F6)
            + 3.0 * bBar**2 * (2.0 * alpha - beta) * (F3 + F6)
            - 6.0 * bBar**2 * beta * F4
            - 3.0 * beta * aaBar * F6
            - 6.0 * bBar * aiBar * F6
        )
        thirdGradient = (
            self._R * T * derivativeIdeal + (a / b) * derivativeAttraction
        ) / 3.0
        dCdv = float(dCdvFixed + 3.0 * thirdGradient @ dDndv)
        dCdt = float(dCdtFixed + 3.0 * thirdGradient @ dDndt)

        return {
            "q": float(eigenvalues[0]),
            "c": float(c),
            "jacobian": np.array([[dQdv, dQdt], [dCdv, dCdt]], dtype=float),
            "Q": Q,
            "eigenvalues": eigenvalues,
            "direction": direction,
        }

    @staticmethod
    def _directionDerivative(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        direction: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        """
        Differentiate a simple minimum eigenvector with an explicit gap check.
        """

        if eigenvalues.size == 1:
            return np.zeros(1, dtype=float)
        gaps = eigenvalues[0] - eigenvalues[1:]
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(np.abs(gaps))) <= _EIGEN_GAP_TOL * scale:
            raise np.linalg.LinAlgError(
                "analytic critical direction is not differentiable at a repeated "
                "minimum eigenvalue"
            )
        other = eigenvectors[:, 1:]
        coefficients = (other.T @ derivative @ direction) / gaps
        return other @ coefficients

    def _volumeFunctions(
        self,
        K: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Return Heidemann--Khalil F1--F6 and derivatives with respect to K.
        """

        delta = self._delta1 - self._delta2
        K1 = K + self._delta1
        K2 = K + self._delta2
        F1 = 1.0 / (K - 1.0)
        F2 = 2.0 / delta * (self._delta1 / K1 - self._delta2 / K2)
        F3 = 1.0 / delta * (
            (self._delta1 / K1) ** 2 - (self._delta2 / K2) ** 2
        )
        F4 = 1.0 / delta * (
            (self._delta1 / K1) ** 3 - (self._delta2 / K2) ** 3
        )
        F5 = 2.0 / delta * math.log(K1 / K2)
        F6 = F2 - F5
        F1p = -F1**2
        F2p = 2.0 / delta * (
            -self._delta1 / K1**2 + self._delta2 / K2**2
        )
        F3p = 2.0 / delta * (
            -self._delta1**2 / K1**3 + self._delta2**2 / K2**3
        )
        F4p = 3.0 / delta * (
            -self._delta1**3 / K1**4 + self._delta2**3 / K2**4
        )
        F5p = -2.0 / (K1 * K2)
        F6p = F2p - F5p
        return (F1, F2, F3, F4, F5, F6), (
            F1p,
            F2p,
            F3p,
            F4p,
            F5p,
            F6p,
        )

    @staticmethod
    def _alphaParameter(mode: int, omega: np.ndarray) -> np.ndarray:
        """
        Return the bundled Soave/PR alpha-function coefficients.
        """

        if mode == EOS_SRK:
            return 0.480 + 1.574 * omega - 0.176 * omega**2
        low = omega <= 0.49
        lowValue = 0.374640 + 1.54226 * omega - 0.26992 * omega**2
        highValue = (
            0.379642
            + 1.48503 * omega
            - 0.164423 * omega**2
            + 0.016666 * omega**3
        )
        return np.where(low, lowValue, highValue)

    @staticmethod
    def _canonicalVector(vector: np.ndarray) -> np.ndarray:
        """
        Choose the same deterministic eigenvector sign as the AD kernel.
        """

        normalized = vector / np.linalg.norm(vector)
        pivot = int(np.argmax(np.abs(normalized)))
        return -normalized if normalized[pivot] < 0.0 else normalized

    def _scalar(self, value: TensorLike, name: str) -> float:
        """
        Convert one model scalar without accepting vector-shaped inputs.
        """

        tensor = self.model.asTensor(value)
        if tensor.ndim != 0:
            raise ValueError(f"{name} must be scalar")
        result = float(tensor.detach().cpu().item())
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result


__all__ = ["CubicCriticalKernel"]
