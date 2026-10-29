r"""
Define the Helmholtz free-energy-density interface, Psi(T, rho_i) = A/V.
Models use PyTorch operations so pressure, chemical potentials and higher
component-density derivatives can be calculated by automatic differentiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import math

import torch


TensorLike = torch.Tensor | float | Sequence[float]


class FreeEnergyModel(ABC):
    """
    Base interface for differentiable Helmholtz models.
    Implement freeEnergy(T, densities) with PyTorch operations. Temperature is
    in K; density and pressure units are declared by each model.
    """

    DENSITY_UNIT = "mol/L"
    DENSITY_TO_MOL_L = 1.0
    PRESSURE_UNIT = "bar"
    PRESSURE_TO_BAR = 1.0

    def __init__(
        self,
        *,
        gasConstant: float,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Set the numerical precision, device and gas constant in model-native units.
        The gas constant must be finite and positive.
        """

        if not math.isfinite(gasConstant) or gasConstant <= 0.0:
            raise ValueError("gasConstant must be finite and positive")
        self.dtype = dtype
        self.device = torch.device(device)
        self._gasConstant = float(gasConstant)

    @property
    def gasConstant(self) -> float:
        """
        Gas constant in the unit system used by the model.
        """

        return self._gasConstant

    @property
    @abstractmethod
    def componentCount(self) -> int:
        """
        Number of components accepted by the model.
        """

    @property
    @abstractmethod
    def tempScale(self) -> float:
        """
        Return the positive characteristic temperature [K] for default bounds.
        """

    @property
    @abstractmethod
    def densityScale(self) -> float:
        """
        Return a positive characteristic total density in the model density unit.
        """

    @property
    def densityUnit(self) -> str:
        """
        Return the model-native molar-density unit.
        """

        return self.DENSITY_UNIT

    @property
    def densityScaleMolL(self) -> float:
        """
        Return the characteristic total density in mol/L.
        """

        return self.densityScale * self.DENSITY_TO_MOL_L

    def densityToMolL(self, density: TensorLike) -> torch.Tensor:
        """
        Convert a model-native density to the public ``mol/L`` unit.
        """

        return self.asTensor(density) * self.DENSITY_TO_MOL_L

    def densityFromMolL(self, densityMolL: TensorLike) -> torch.Tensor:
        """
        Convert a public ``mol/L`` density to the model-native unit.
        """

        return self.asTensor(densityMolL) / self.DENSITY_TO_MOL_L

    @property
    def presUnit(self) -> str:
        """
        Return the model-native pressure unit.
        """

        return self.PRESSURE_UNIT

    @property
    def supportsBatchedThermoState(self) -> bool:
        """
        Return whether full thermodynamic states support ``torch.vmap``.

        The default is conservative so third-party models retain scalar state
        evaluation unless they explicitly make batched higher-order automatic
        differentiation explicitly.
        """

        return False

    def presToBar(self, pressure: TensorLike) -> torch.Tensor:
        """
        Convert a model-native pressure to the public ``bar`` unit.
        """

        return self.asTensor(pressure) * self.PRESSURE_TO_BAR

    def presFromBar(self, pressureBar: TensorLike) -> torch.Tensor:
        """
        Convert a public ``bar`` pressure to the model-native unit.
        """

        return self.asTensor(pressureBar) / self.PRESSURE_TO_BAR

    def compositionDensityScale(self, composition: TensorLike) -> float:
        """
        Return a positive pseudo-reducing density for spinodal discovery.

        The default is composition independent and uses ``densityScale``.
        Models with composition-dependent reducing functions should override
        this method. This scale initializes the search; it is not an EOS
        validity bound; domain limits remain owned by the validation hooks.
        """

        self._normalizedComposition(composition)
        scale = float(self.densityScale)
        isFinite = bool(
            torch.isfinite(self.asTensor(scale)).detach().cpu().item()
        )
        if not isFinite or scale <= 0.0:
            raise ValueError("densityScale must be finite and positive")
        return scale

    @property
    def molarMasses(self) -> tuple[float, ...] | None:
        """
        Return component molar masses in kg/mol when model metadata has them.

        Helmholtz pressure and molar caloric properties do not require molar
        masses. Converting the isentropic pressure derivative to a speed of
        sound does, so models may expose the metadata here or callers may
        supply it directly to the thermodynamic kernel.
        """

        return None

    def densityRootUpperBound(
        self,
        temperature: TensorLike,
        composition: TensorLike,
    ) -> float:
        """
        Return a conservative upper density for pressure-root discovery.

        This default is deliberately model-neutral. EOS plugins with a known
        hard-core or packing bound should override it; root finding remains in
        the application layer while the model owns its domain metadata.
        """

        del temperature
        self._normalizedComposition(composition)
        return 1.95 * self.densityScale

    def activeSubmodel(self, componentIndices: Sequence[int]) -> "FreeEnergyModel":
        """
        Return the same EOS restricted to an ordered active-component set.

        Exact zero compositions require the CP application to remove inactive
        density coordinates before evaluating ideal-mixture logarithms. The
        unchanged component set is always supported; mixture EOS models that
        preserve their parameters under reduction should override this hook.
        """

        indices = tuple(int(index) for index in componentIndices)
        if indices == tuple(range(self.componentCount)):
            return self
        raise NotImplementedError(
            f"{type(self).__name__} does not support active-component reduction"
        )

    @abstractmethod
    def freeEnergy(
        self,
        temperature: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Return Helmholtz free energy density $\Psi(T,\boldsymbol\rho)$.
        """

    def fixedCompositionFreeEnergy(
        self, temperature: TensorLike, composition: TensorLike,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Prepare q(rho) = Psi(T, rho*y) for repeated scalar-density AD.

        Temperature is in K; the returned function accepts scalar total
        density and returns energy density in model-native units. The caller
        supplies a validated positive, normalized composition. EOS overrides
        may prepare quantities independent of density once per call, without
        changing ``freeEnergy`` used for component derivatives. The closure
        is local to this temperature/composition, not a model-wide cache.
        """

        temp, fractions = self.stateTensors(temperature, composition)

        def energyAt(density: torch.Tensor) -> torch.Tensor:
            """
            Evaluate energy density along this fixed-temperature composition ray.
            """

            return self.freeEnergy(temp, density * fractions)

        return energyAt

    def asTensor(self, value: TensorLike) -> torch.Tensor:
        """
        Move a value to the model dtype/device without detaching its graph.
        """

        if isinstance(value, torch.Tensor):
            return value.to(dtype=self.dtype, device=self.device)
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _normalizedComposition(self, composition: TensorLike) -> torch.Tensor:
        """
        Normalize finite positive active-component weights to mole fractions.
        """

        fractions = self.asTensor(composition)
        if fractions.shape != (self.componentCount,):
            raise ValueError(
                f"composition must have shape ({self.componentCount},)"
            )
        if not bool(torch.all(torch.isfinite(fractions)).detach().cpu().item()):
            raise ValueError("composition must be finite")
        if bool(torch.any(fractions.detach() <= 0.0).cpu().item()):
            raise ValueError(
                "all active-component composition weights must be positive"
            )
        total = torch.sum(fractions)
        if not bool(torch.isfinite(total).detach().cpu().item()):
            raise ValueError("composition sum must be finite")
        return fractions / total

    def stateTensors(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize a scalar temperature and component-density vector.
        """

        tempTensor = self.asTensor(temperature)
        densityTensor = self.asTensor(densities)
        if tempTensor.ndim != 0:
            raise ValueError("temperature must be a scalar tensor")
        if densityTensor.shape != (self.componentCount,):
            raise ValueError(
                f"densities have shape {tuple(densityTensor.shape)}; "
                f"expected ({self.componentCount},)"
            )
        return tempTensor, densityTensor

    def validateState(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> None:
        """
        Reject non-finite, non-positive or otherwise invalid basic states.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        finite = torch.isfinite(tempTensor) & torch.all(torch.isfinite(densityTensor))
        if not bool(finite.detach().cpu().item()):
            raise ValueError("temperature and densities must be finite")
        if tempTensor.detach().cpu().item() <= 0.0:
            raise ValueError("temperature must be positive")
        if bool(torch.any(densityTensor.detach() <= 0.0).cpu().item()):
            raise ValueError("all active-component densities must be positive")

    def validatePres(self, pressure: TensorLike) -> None:
        """
        Reject a non-finite, non-scalar pressure.

        EOS plugins with a documented pressure envelope may extend this hook.
        The pressure check is separate from :meth:`validateState` because
        pressure is a derivative of the free energy and is only available
        after the thermodynamic kernel has evaluated the state.
        """

        presTensor = self.asTensor(pressure)
        if presTensor.ndim != 0:
            raise ValueError("pressure must be a scalar tensor")
        if not bool(torch.isfinite(presTensor).detach().cpu().item()):
            raise ValueError("pressure must be finite")

    def domainMargin(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> torch.Tensor:
        """
        Return a positive dimensionless margin inside the basic domain.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        tempMargin = tempTensor / self.asTensor(self.tempScale)
        densityMargin = self.asTensor(self.densityScale) / totalDensity
        compositionMargin = torch.min(densityTensor) / torch.mean(densityTensor)
        return torch.minimum(
            torch.minimum(tempMargin, densityMargin),
            compositionMargin,
        )

    def idealChemicalPotentials(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> torch.Tensor:
        r"""
        Return ideal-gas chemical potentials in the model reference.

        The default corresponds to
        ``Psi_id = R*T*sum(rho_i*(log(rho_i) - 1))`` and therefore returns
        ``R*T*log(rho_i)``. A model with additional component-dependent ideal
        reference terms must override this method so residual chemical
        potentials and fugacity coefficients remain reference-consistent.
        """

        tempTensor, densityTensor = self.stateTensors(
            temperature,
            densities,
        )
        return (
            self.asTensor(self.gasConstant)
            * tempTensor
            * torch.log(densityTensor)
        )
