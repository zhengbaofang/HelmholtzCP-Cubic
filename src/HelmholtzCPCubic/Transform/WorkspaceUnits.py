"""
Convert native Helmholtz models to K, bar and mol/L.
Transform the energy and density coordinates consistently, retaining automatic
differentiation and the original EOS parameters.
"""

from collections.abc import Callable, Sequence
import math

import torch

from ..Common.Model import FreeEnergyModel, TensorLike


class WorkspaceUnitModel(FreeEnergyModel):
    """
    Expose a native model's Helmholtz potential in bar and densities in mol/L.

    For native-to-public density and pressure factors d and p,
    Psi_public(T, rho) = p*Psi_native(T, rho/d) and R_public = p*R_native/d.
    This preserves dimensionless critical equations and log-state derivatives.
    Operations remain in the native model differentiation graph.
    """

    def __init__(self, nativeModel: FreeEnergyModel) -> None:
        """
        Retain the EOS and validate its declared positive conversion factors.
        """

        self.nativeModel = nativeModel
        self._densityFactor = float(nativeModel.DENSITY_TO_MOL_L)
        self._pressureFactor = float(nativeModel.PRESSURE_TO_BAR)
        if any(not math.isfinite(v) or v <= 0.0 for v in (
            self._densityFactor, self._pressureFactor,
        )):
            raise ValueError("model unit conversion factors must be positive and finite")
        super().__init__(
            gasConstant=nativeModel.gasConstant * self._pressureFactor / self._densityFactor,
            dtype=nativeModel.dtype,
            device=nativeModel.device,
        )

    @property
    def componentCount(self) -> int:
        """
        Return the native component count.
        """

        return self.nativeModel.componentCount

    @property
    def tempScale(self) -> float:
        """
        Return the unchanged characteristic temperature in K.
        """

        return self.nativeModel.tempScale

    @property
    def densityScale(self) -> float:
        """
        Return the characteristic density in mol/L.
        """

        return self.nativeModel.densityScale * self._densityFactor

    @property
    def supportsBatchedThermoState(self) -> bool:
        """
        Preserve the native model's explicit batching capability.
        """

        return self.nativeModel.supportsBatchedThermoState

    @property
    def molarMasses(self) -> tuple[float, ...] | None:
        """
        Return unchanged native molar-mass metadata in kg/mol.
        """

        return self.nativeModel.molarMasses

    @property
    def pureCriticalTemp(self) -> tuple[float, ...] | None:
        """
        Return optional pure-component critical temperatures in K.
        """

        return getattr(self.nativeModel, "pureCriticalTemp", None)

    @property
    def pureCriticalPres(self) -> tuple[float, ...] | None:
        """
        Convert optional pure-component critical pressures to bar for Wilson starts.
        """

        values = getattr(self.nativeModel, "pureCriticalPres", None)
        return None if values is None else tuple(float(v) * self._pressureFactor for v in values)

    @property
    def pureAcentricFactor(self) -> tuple[float, ...] | None:
        """
        Return optional dimensionless acentric factors.
        """

        return getattr(self.nativeModel, "pureAcentricFactor", None)

    def compositionTempScale(self, composition: TensorLike) -> float:
        """
        Return the native composition-dependent temperature scale in K.
        """

        method = getattr(self.nativeModel, "compositionTempScale", None)
        return float(method(composition)) if callable(method) else self.tempScale

    def compositionDensityScale(self, composition: TensorLike) -> float:
        """
        Convert the native composition-dependent scale to mol/L.
        """

        return self.nativeModel.compositionDensityScale(composition) * self._densityFactor

    def densityRootUpperBound(self, temperature: TensorLike, composition: TensorLike) -> float:
        """
        Convert the model's density-search upper bound to mol/L.
        """

        return self.nativeModel.densityRootUpperBound(temperature, composition) * self._densityFactor

    def activeSubmodel(self, componentIndices: Sequence[int]) -> FreeEnergyModel:
        """
        Adapt a supported native active-component reduction, propagating failures.
        """

        reduced = self.nativeModel.activeSubmodel(componentIndices)
        return self if reduced is self.nativeModel else asWorkspaceModel(reduced)

    def freeEnergy(self, temperature: torch.Tensor, densities: torch.Tensor) -> torch.Tensor:
        """
        Return Psi in bar from component densities in mol/L, retaining autograd.
        """

        return self._pressureFactor * self.nativeModel.freeEnergy(
            temperature, densities / self._densityFactor,
        )

    def fixedCompositionFreeEnergy(
        self, temperature: TensorLike, composition: TensorLike,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Prepare q(rho) in bar for scalar rho in mol/L at fixed K and y.

        The native preparation is reused across density evaluations; its
        density derivatives retain both unit conversion factors.
        """

        nativeEnergy = self.nativeModel.fixedCompositionFreeEnergy(temperature, composition)

        def energyAt(density: torch.Tensor) -> torch.Tensor:
            """
            Map mol/L to native density and return energy density in bar.
            """

            return self._pressureFactor * nativeEnergy(density / self._densityFactor)

        return energyAt

    def validateState(self, temperature: TensorLike, densities: TensorLike) -> None:
        """
        Apply native domain checks after converting component densities.
        """

        self.nativeModel.validateState(temperature, self.asTensor(densities) / self._densityFactor)

    def validatePres(self, pressure: TensorLike) -> None:
        """
        Apply native pressure-domain checks to a public pressure in bar.
        """

        self.nativeModel.validatePres(self.asTensor(pressure) / self._pressureFactor)

    def domainMargin(self, temperature: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Return the unchanged dimensionless native domain margin.
        """

        return self.nativeModel.domainMargin(temperature, self.asTensor(densities) / self._densityFactor)

    def idealChemicalPotentials(self, temperature: TensorLike, densities: TensorLike) -> torch.Tensor:
        """
        Convert ideal chemical potentials to bar L/mol, preserving the EOS reference.
        """

        return (self._pressureFactor / self._densityFactor) * self.nativeModel.idealChemicalPotentials(
            temperature, self.asTensor(densities) / self._densityFactor,
        )


def asWorkspaceModel(model: FreeEnergyModel) -> FreeEnergyModel:
    """
    Return a K/bar/mol/L model; already compatible models retain their identity.

    Preserving the model object keeps the cubic analytic backend available
    and prevents repeated conversion when adapted models cross several APIs.
    Conversion follows declared metadata, never the magnitude of state values.
    """

    if (
        model.DENSITY_TO_MOL_L == 1.0
        and model.PRESSURE_TO_BAR == 1.0
        and model.densityUnit == "mol/L"
        and model.presUnit == "bar"
    ):
        return model
    return WorkspaceUnitModel(model)
