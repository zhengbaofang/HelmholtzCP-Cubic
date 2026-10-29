"""
Explicit backend selection for local ordinary-critical correction.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, cast

import torch

from ..Common.Model import FreeEnergyModel, TensorLike
from ..Kernel.Critical import (
    DensityCriticalEvaluation,
    DensityCriticalKernel,
    DensityCriticalState,
)
from ..Kernel.CubicCritical import CubicCriticalKernel
from ..Kernel.CubicMulticomplex import CubicMulticomplexKernel
from ..Transform.CubicBell import CubicBellModel


class CriticalBackend(str, Enum):
    """
    Requested local critical-condition evaluation backend.
    """

    AD = "ad"
    ANALYTIC = "analytic"
    PYMCX = "pymcx"
    AUTO = "auto"


class CriticalEvaluator(Protocol):
    """
    Residual, Jacobian and pressure interface used by the critical-point solver.
    """

    name: str

    def evaluate(
        self,
        T: TensorLike,
        rho: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalState:
        """
        Evaluate one validated local critical state.
        """

    def criticalEvaluation(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalEvaluation:
        """
        Return the residual, Jacobian and final thermodynamic state.
        """

    def pressure(self, T: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Return pressure in the model unit system.
        """


class ADCriticalEvaluator:
    """
    Adapt the Helmholtz differentiation kernel to the local solver interface.
    """

    name = "helmholtz_ad"

    def __init__(self, kernel: DensityCriticalKernel) -> None:
        self.kernel = kernel

    def evaluate(
        self,
        T: TensorLike,
        rho: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalState:
        """
        Delegate one critical-state evaluation to the AD kernel.
        """

        return self.kernel.evaluate(T, rho, composition)

    def criticalEvaluation(
        self,
        logState: TensorLike,
        composition: TensorLike,
    ) -> DensityCriticalEvaluation:
        """
        Delegate residual and Jacobian generation to the AD kernel.
        """

        return self.kernel.criticalEvaluation(logState, composition)

    def pressure(self, T: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Return Helmholtz pressure generated from the shared free energy.
        """

        return self.kernel.thermo.pressure(T, densities, validate=False)


def selectCriticalBackend(
    model: FreeEnergyModel,
    adEvaluator: ADCriticalEvaluator,
    backend: CriticalBackend | str,
) -> tuple[CriticalBackend, CriticalEvaluator]:
    """
    Resolve one explicit backend without any numerical-failure fallback.

    Args:
        model: Selected Helmholtz free-energy model. It is the sole source of
            EOS identity and parameters.
        adEvaluator: Generic AD evaluator already owned by the solver.
        backend: ``"ad"`` always uses the generic Helmholtz implementation;
            ``"analytic"`` requires a supported analytic implementation;
            ``"pymcx"`` uses multicomplex derivatives for classic PR/SRK;
            ``"auto"`` selects an analytic implementation when the exact model
            formulation is supported and otherwise selects automatic differentiation.

    Returns:
        Normalized request and the evaluator selected from model capability.

    Raises:
        ValueError: If the parameter is unknown or an explicitly requested
            analytic backend is unavailable. Runtime failures never trigger a
            hidden backend switch.
    """

    try:
        requested = (
            backend
            if isinstance(backend, CriticalBackend)
            else CriticalBackend(backend)
        )
    except ValueError as error:
        choices = ", ".join(item.value for item in CriticalBackend)
        raise ValueError(f"backend must be one of: {choices}") from error

    if requested is CriticalBackend.AD:
        return requested, adEvaluator

    if requested is CriticalBackend.PYMCX:
        return requested, CubicMulticomplexKernel(cast(CubicBellModel, model))

    reason = CubicCriticalKernel.unsupportedReason(model)
    if reason is None:
        return requested, CubicCriticalKernel(cast(CubicBellModel, model))
    if requested is CriticalBackend.ANALYTIC:
        raise ValueError(
            f"{reason}; use backend='ad' or backend='auto'"
        )
    return requested, adEvaluator


__all__ = [
    "ADCriticalEvaluator",
    "CriticalBackend",
    "CriticalEvaluator",
    "selectCriticalBackend",
]
