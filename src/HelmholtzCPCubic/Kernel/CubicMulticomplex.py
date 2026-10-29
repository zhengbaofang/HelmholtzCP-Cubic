r"""
Evaluate PR/SRK critical conditions using the optional multicomplex library.
Differentiate the shared Helmholtz formula through fourth order and include
eigendirection changes in the Jacobian, following Bell and Jäger (2017).
Use CPU float64 inputs in K and mol/L.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import torch

from ..Common.Composition import normalizeComposition
from ..Common.Model import TensorLike
from ..Transform.CubicBell import CubicBellModel, _cubicResidualAlpha
from .Critical import DensityCriticalEvaluation, DensityCriticalState
from .CubicCritical import CubicCriticalKernel


class CubicMulticomplexKernel:
    """
    Evaluate PR/SRK critical residuals and their log-state Jacobian.

    The optional ``multicomplex`` dependency is loaded only when requested.
    A missing dependency, invalid physical state, or repeated minimum
    eigenvalue raises explicitly. No other derivative backend is selected
    after an error. Pressure retains the direct cubic EOS evaluation.
    """

    name = "cubic_pymcx"

    def __init__(self, model: CubicBellModel) -> None:
        reason = CubicCriticalKernel.unsupportedReason(model)
        if reason is not None:
            raise ValueError(f"pymcx requires classic PR/SRK on CPU in float64: {reason}")
        try:
            self._mcx = importlib.import_module("multicomplex")
        except ModuleNotFoundError as error:
            if error.name != "multicomplex":
                raise
            raise ImportError(
                "backend='pymcx' requires the optional multicomplex package"
            ) from error
        self.model = model
        self._R = float(model.gasConstant)

        # Reuse the actual model's constants, alpha coefficients and parameters.
        self._Tc = model._criticalTemp.detach().cpu().numpy()
        self._m = model._m.detach().cpu().numpy()
        self._bi = model._bi.detach().cpu().numpy()
        self._kij = model._cubicKij.detach().cpu().numpy()
        self._a0 = (
            float(model._omegaA) * self._R**2 * self._Tc**2
            / model._criticalPres.detach().cpu().numpy()
        )
        self._delta1, self._delta2 = float(model._delta1), float(model._delta2)

    def evaluate(self, T: TensorLike, rho: TensorLike,
                 composition: TensorLike) -> DensityCriticalState:
        """
        Return the dimensionless critical state at T [K], rho [mol/L].
        """

        state = np.array([self._scalar(T), self._scalar(rho)])
        return self._evaluation(state, composition, needJacobian=False)[0]

    def criticalEvaluation(self, logState: TensorLike,
                           composition: TensorLike) -> DensityCriticalEvaluation:
        """
        Return residual and its 2-by-2 Jacobian in log(T), log(rho).
        """

        state = self.model.asTensor(logState).detach().cpu().numpy()
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError("logState must be a finite vector with shape (2,)")
        critical, jacobian = self._evaluation(
            np.exp(state), composition, needJacobian=True,
        )
        return DensityCriticalEvaluation(
            logState=self.model.asTensor(state).detach(), critical=critical,
            jacobian=self.model.asTensor(jacobian).detach(),
        )

    def pressure(self, T: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Return direct cubic pressure [bar] with the original validation.
        """

        self.model.validateState(T, densities)
        pressure = self.model.directPres(T, densities)
        self.model.validatePres(pressure)
        return pressure.detach()

    def _freeEnergy(self, temperature, densities):
        """
        Extend the model's full Helmholtz density [bar] to multicomplex.

        Scalar loops preserve multicomplex values through every mixing term.
        The residual equation is shared with the Torch model, including the
        same PR/SRK constants and alpha branch selected at construction.
        """

        rho = sum(densities)
        fractions = [value / rho for value in densities]
        pureA = [
            self._a0[i] * (1.0 + self._m[i] * (
                1.0 - np.sqrt(temperature / self._Tc[i])
            ))**2 for i in range(len(densities))
        ]
        attraction = sum(
            fractions[i] * fractions[j] * (1.0 - self._kij[i, j])
            * np.sqrt(pureA[i] * pureA[j])
            for i in range(len(densities)) for j in range(len(densities))
        )
        coVolume = sum(fractions[i] * self._bi[i] for i in range(len(densities)))
        rt = self._R * temperature
        alpha = _cubicResidualAlpha(
            coVolume * rho, attraction, rt, coVolume, self._delta1, self._delta2,
            logarithm=np.log, logOnePlus=lambda value: np.log(1.0 + value),
        )
        ideal = rt * sum(value * (np.log(value) - 1.0) for value in densities)
        return ideal + rho * rt * alpha

    def _derivative(self, T, densities, directions, orders) -> float:
        """
        Differentiate log-state shifts and linear density directions.

        The first two coordinates perturb log(T) and log(rho). Remaining
        coordinates shift component densities along the supplied fixed vectors.
        Only requested contractions are formed, never a full order-4 tensor.
        """

        def energy(offsets):
            """
            Retain logarithmic and component-density shifts in the scalar potential.
            """

            temp = T * np.exp(offsets[0]) if orders[0] else T
            densityScale = np.exp(offsets[1]) if orders[1] else 1.0
            shifted = [
                float(value) * densityScale + sum(
                    offsets[k + 2] * float(direction[i])
                    for k, direction in enumerate(directions)
                ) for i, value in enumerate(densities)
            ]
            return self._freeEnergy(temp, shifted)

        value = float(self._mcx.diff_mcxN(energy, [0.0] * len(orders), list(orders)))
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite multicomplex Helmholtz derivative")
        return value

    def _evaluation(self, state, composition, *, needJacobian):
        """
        Assemble critical conditions and differentiate the simple eigenmode.
        """

        T, rho = (float(value) for value in state)
        if not (math.isfinite(T) and math.isfinite(rho) and T > 0.0 and rho > 0.0):
            raise ValueError("T and rho must be finite and positive")
        x = normalizeComposition(
            self.model.asTensor(composition).detach().cpu().numpy(),
            self.model.componentCount,
        )
        densities = rho * x
        self.model.validateState(T, densities)
        count = len(x)
        axes = np.eye(count)
        hessian = np.empty((count, count))
        hessianDerivatives = np.empty((2, count, count)) if needJacobian else None
        for i in range(count):
            for j in range(i, count):
                directions = (axes[i], axes[j])
                hessian[i, j] = hessian[j, i] = self._derivative(
                    T, densities, directions, (0, 0, 1, 1),
                )
                if needJacobian:
                    for k, order in enumerate(((1, 0, 1, 1), (0, 1, 1, 1))):
                        value = self._derivative(T, densities, directions, order)
                        hessianDerivatives[k, i, j] = hessianDerivatives[k, j, i] = value

        scale = rho / (self._R * T)
        stability = scale * hessian
        eigenvalues, eigenvectors = np.linalg.eigh(stability)
        direction = CubicCriticalKernel._canonicalVector(eigenvectors[:, 0])
        third = self._derivative(T, densities, (direction,), (0, 0, 3))
        thirdScale = rho * scale
        cStar = thirdScale * third
        critical = DensityCriticalState(
            stabilityMatrix=self.model.asTensor(stability),
            eigenvalues=self.model.asTensor(eigenvalues),
            minEigenvector=self.model.asTensor(direction),
            qMin=self.model.asTensor(eigenvalues[0]),
            cStar=self.model.asTensor(cStar),
            densities=self.model.asTensor(densities),
        )
        if not needJacobian:
            return critical, np.empty((0, 0))

        jacobian = np.empty((2, 2))
        for k, scaleDerivative in enumerate((-1.0, 1.0)):
            matrixDerivative = scale * (hessianDerivatives[k] + scaleDerivative * hessian)
            jacobian[0, k] = direction @ matrixDerivative @ direction

            # The mode changes with state. Holding it fixed would omit an
            # essential term in dC*/dlog(T,rho), especially away from a root.
            directionDerivative = CubicCriticalKernel._directionDerivative(
                eigenvalues, eigenvectors, direction, matrixDerivative,
            )
            fixedThird = self._derivative(
                T, densities, (direction,), (1 - k, k, 3),
            )
            modeThird = self._derivative(
                T, densities, (direction, directionDerivative), (0, 0, 2, 1),
            ) if np.any(directionDerivative) else 0.0
            jacobian[1, k] = (-1.0 if k == 0 else 2.0) * cStar + thirdScale * (
                fixedThird + 3.0 * modeThird
            )
        return critical, jacobian

    def _scalar(self, value: TensorLike) -> float:
        """
        Accept only one scalar model value, without retaining an AD graph.
        """

        tensor = self.model.asTensor(value)
        if tensor.ndim != 0:
            raise ValueError("T and rho must be scalars")
        return float(tensor.detach().cpu())
