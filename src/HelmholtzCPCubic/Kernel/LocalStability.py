r"""
Check local fourth-order stability at a corrected critical point.
Include the change of the critical eigendirection along the density path.
Follow Michelsen (1984) and Deiters and Bell (2020).
Use the model's native density and free-energy units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from ..Common.Model import FreeEnergyModel, TensorLike
from .Critical import DensityCriticalKernel


@dataclass(frozen=True)
class LocalStabilityResult:
    r"""
    Fourth-order local classification in the model's density coordinates.

    The direction and its correction use the model density unit as the path
    coordinate. ``eigenvalues`` and ``qMin`` use dimensionless Hessian
    scaling, while the three fourth-directional fields use the fourth density
    derivative unit of ``Psi``. ``dRawDensity`` and ``dStarDensity`` divide
    those contractions by 24 for the corresponding free-energy expansion.

    ``isLocallyStable`` is ``None`` when the corrected fourth-order term is
    numerically indistinguishable from zero and the state is therefore a
    higher-order candidate rather than an ordinary classified point.
    """

    direction: tuple[float, ...]
    directionCorrection: tuple[float, ...]
    eigenvalues: tuple[float, ...]
    eigenvalueGap: float
    qMin: float
    cStar: float
    rawFourthDirectional: float
    correctionContribution: float
    correctedFourthDirectional: float
    dRawDensity: float
    dStarDensity: float
    isLocallyStable: bool | None


class LocalStabilityKernel:
    """
    Evaluate the curved-eigendirection fourth-order stability measure.

    The model must provide a four-times differentiable Helmholtz free-energy
    density at the classified state. Repeated or numerically unresolved
    minimum eigenvalues are outside this simple-eigenvalue implementation.

    For a simple zero eigenvalue, the fourth-order measure is
    S4 = A4[d,d,d,d] - 3*r_perp.T@(H-lambda*I)^+@r_perp, with r = A3[:,d,d].
    This accounts for curvature of the critical eigendirection (Deiters and Bell,
    2020, Eq. 36). S4/24 is the corrected Helmholtz expansion coefficient,
    corresponding to the local stability test of Michelsen (1984).
    Root correction and TPD phase-stability screening are handled separately.
    """

    def __init__(self, model: FreeEnergyModel) -> None:
        self.model = model
        self.criticalKernel = DensityCriticalKernel(model)
        self.thermo = self.criticalKernel.thermo

    def evaluate(
        self,
        T: TensorLike,
        rho: TensorLike,
        composition: TensorLike,
        *,
        direction: Sequence[float] | np.ndarray | None = None,
        criticalResidualTolerance: float = 1.0e-7,
        eigenGapRelativeTolerance: float = 1.0e-8,
        fourthRelativeTolerance: float = 1.0e-8,
    ) -> LocalStabilityResult:
        r"""
        Classify one already corrected ordinary critical point.

        ``T`` is in K and ``rho`` uses the model total-density unit. The three
        tolerances are dimensionless relative controls. A noncritical state,
        unresolved minimum eigenspace, invalid direction, or non-finite
        fourth-order contraction raises ``ValueError``.
        """

        for name, value in (
            ("criticalResidualTolerance", criticalResidualTolerance),
            ("eigenGapRelativeTolerance", eigenGapRelativeTolerance),
            ("fourthRelativeTolerance", fourthRelativeTolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        critical = self.criticalKernel.evaluate(T, rho, composition)
        qMin = float(critical.qMin.detach().cpu().item())
        cStar = float(critical.cStar.detach().cpu().item())
        if math.hypot(qMin, cStar) > criticalResidualTolerance:
            raise ValueError(
                "local stability requires a corrected critical state: "
                f"residual={math.hypot(qMin, cStar):.3e}"
            )

        eigenvalues = critical.eigenvalues.detach().cpu().numpy()
        componentCount = self.model.componentCount
        if componentCount > 1:
            eigenvalueGap = float(eigenvalues[1] - eigenvalues[0])
            eigenvalueScale = max(1.0, float(np.max(np.abs(eigenvalues))))
            if eigenvalueGap <= eigenGapRelativeTolerance * eigenvalueScale:
                raise ValueError(
                    "minimum Hessian eigenspace is unresolved: "
                    f"relative gap={eigenvalueGap / eigenvalueScale:.3e}"
                )
        else:
            eigenvalueGap = math.inf

        criticalDirection = critical.minEigenvector.detach().cpu().numpy()
        if direction is None:
            activeDirection = criticalDirection
        else:
            activeDirection = np.asarray(direction, dtype=float)
            if activeDirection.shape != (componentCount,):
                raise ValueError(f"direction must have shape ({componentCount},)")
            if not np.all(np.isfinite(activeDirection)):
                raise ValueError("direction must be finite")
            directionNorm = float(np.linalg.norm(activeDirection))
            if directionNorm <= np.finfo(float).eps:
                raise ValueError("direction norm must be positive")
            activeDirection = activeDirection / directionNorm
            alignment = abs(float(activeDirection @ criticalDirection))
            if 1.0 - alignment > 10.0 * eigenGapRelativeTolerance:
                raise ValueError("direction must span the minimum Hessian eigenspace")

        # Differentiate along the critical direction to obtain third and fourth terms.
        Tt = self.model.asTensor(T).detach()
        densities = critical.densities.detach().clone().requires_grad_(True)
        directionTensor = self.model.asTensor(activeDirection).detach()
        hessian = self.thermo.hessian(Tt, densities, validate=False)
        secondDirectional = directionTensor @ hessian @ directionTensor
        thirdVector = torch.autograd.grad(
            secondDirectional,
            densities,
            create_graph=True,
        )[0]
        thirdDirectional = torch.dot(thirdVector, directionTensor)
        fourthGradient = torch.autograd.grad(thirdDirectional, densities)[0]
        rawFourth = torch.dot(fourthGradient, directionTensor)

        rawHessian = hessian.detach()
        rawEigenvalues, rawEigenvectors = torch.linalg.eigh(rawHessian)
        perpendicular = thirdVector.detach() - (
            torch.dot(thirdVector.detach(), directionTensor) * directionTensor
        )

        # Account for the rotation of the critical direction into the stable eigenspace.
        directionCorrection = torch.zeros_like(directionTensor)
        for index in range(1, componentCount):
            stableDirection = rawEigenvectors[:, index]
            gap = rawEigenvalues[index] - rawEigenvalues[0]
            directionCorrection -= (
                torch.dot(stableDirection, perpendicular) / gap
            ) * stableDirection

        correctionContribution = 3.0 * torch.dot(
            directionCorrection,
            thirdVector.detach(),
        )
        correctedFourth = rawFourth.detach() + correctionContribution
        values = torch.stack(
            (rawFourth.detach(), correctionContribution, correctedFourth)
        )
        if not bool(torch.all(torch.isfinite(values)).cpu().item()):
            raise ValueError("local fourth-order contraction is not finite")

        rawValue, correctionValue, correctedValue = (
            float(value.cpu().item()) for value in values
        )
        fourthScale = max(
            abs(rawValue),
            abs(correctionValue),
            np.finfo(float).tiny,
        )
        fourthThreshold = fourthRelativeTolerance * fourthScale
        isLocallyStable = (
            None
            if abs(correctedValue) <= fourthThreshold
            else correctedValue > 0.0
        )
        return LocalStabilityResult(
            direction=tuple(float(value) for value in activeDirection),
            directionCorrection=tuple(
                float(value) for value in directionCorrection.cpu().numpy()
            ),
            eigenvalues=tuple(float(value) for value in eigenvalues),
            eigenvalueGap=eigenvalueGap,
            qMin=qMin,
            cStar=cStar,
            rawFourthDirectional=rawValue,
            correctionContribution=correctionValue,
            correctedFourthDirectional=correctedValue,
            dRawDensity=rawValue / 24.0,
            dStarDensity=correctedValue / 24.0,
            isLocallyStable=isLocallyStable,
        )


__all__ = ["LocalStabilityKernel", "LocalStabilityResult"]
