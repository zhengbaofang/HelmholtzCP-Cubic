r"""
Express PR/SRK mixtures as Helmholtz free-energy density Psi(T, rho_i) = A/V.
Use component-density derivatives to obtain pressure and critical conditions.
The formulation follows Bell and Jäger (2016, 2017).
Units: T [K], rho_i [mol/L], Psi and P [bar], R [bar L/(mol K)].
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from ..Common.Model import FreeEnergyModel, TensorLike


R_GAS = 0.083145  # bar L/(mol K), shared by all PR/SRK models.
EOS_SRK = 1
EOS_PR = 2
PA_PER_BAR = 1.0e5
LITERS_PER_CUBIC_METER = 1.0e3
BELL_VOLUME_SLOPE = 2.141  # Pa m^3/(mol K), critical-volume correlation.
BELL_VOLUME_INTERCEPT = 7.773e-6  # m^3/mol.


def normalizeMode(mode: str | int) -> int:
    """
    Normalize a PR/SRK name or integer identifier.
    """

    if isinstance(mode, int):
        if mode in (EOS_SRK, EOS_PR):
            return mode
        raise ValueError(f"unsupported EOS integer: {mode}")
    name = mode.upper()
    if name == "SRK":
        return EOS_SRK
    if name == "PR":
        return EOS_PR
    raise ValueError(f"unsupported EOS name: {mode!r}")


def cubicConstants(mode: str | int) -> tuple[float, float, float, float]:
    """
    Return ``Omega_a, Omega_b, delta1, delta2``.
    """

    normalized = normalizeMode(mode)
    if normalized == EOS_SRK:
        return 0.4274802, 0.08664035, 1.0, 0.0
    rootTwo = math.sqrt(2.0)
    return 0.457235529, 0.077796074, 1.0 + rootTwo, 1.0 - rootTwo


def _cubicResidualAlpha(
    packing, attraction, rt, coVolume, delta1, delta2, *, logarithm, logOnePlus,
):
    """
    Evaluate the shared PR/SRK residual formula with supplied scalar logs.

    Inputs follow bar/L/mol units, so ``packing`` and the returned molar
    residual Helmholtz energy divided by RT are dimensionless. Supplying the
    logarithms lets Torch and multicomplex arithmetic use the same equation.
    State validation remains at their real-valued public entry points.
    """

    repulsive = -logOnePlus(-packing)
    attractive = -attraction / (rt * coVolume * (delta1 - delta2)) * logarithm(
        (1.0 + delta1 * packing) / (1.0 + delta2 * packing)
    )
    return repulsive + attractive


class CubicBellModel(FreeEnergyModel):
    """
    PR/SRK Helmholtz free energy density in bar--L--mol--K units.
    """

    def __init__(
        self,
        criticalTemp: Sequence[float],
        criticalPres: Sequence[float],
        acentricFactor: Sequence[float],
        cubicKij: Sequence[Sequence[float]],
        *,
        mode: str | int = "PR",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Build a PR or SRK mixture from pure-component properties and interactions.
        criticalTemp [K] and criticalPres [bar] follow the component order.
        acentricFactor and symmetric cubicKij are dimensionless. Invalid shapes,
        non-finite parameters or nonpositive critical properties raise ValueError.
        """

        self.mode = normalizeMode(mode)
        super().__init__(gasConstant=R_GAS, dtype=dtype, device=device)
        self._criticalTemp = self.asTensor(criticalTemp)
        self._criticalPres = self.asTensor(criticalPres)
        self._acentricFactor = self.asTensor(acentricFactor)
        self._cubicKij = self.asTensor(cubicKij)
        if self._criticalTemp.ndim != 1 or self._criticalTemp.numel() == 0:
            raise ValueError("criticalTemp must be a nonempty vector")
        count = int(self._criticalTemp.numel())
        if self._criticalPres.shape != (count,):
            raise ValueError("criticalPres must match criticalTemp")
        if self._acentricFactor.shape != (count,):
            raise ValueError("acentricFactor must match criticalTemp")
        if self._cubicKij.shape != (count, count):
            raise ValueError("cubicKij must be a square component matrix")
        if not bool(torch.all(torch.isfinite(self._criticalTemp)).cpu().item()):
            raise ValueError("critical temperatures must be finite")
        if not bool(torch.all(torch.isfinite(self._criticalPres)).cpu().item()):
            raise ValueError("critical pressures must be finite")
        if not bool(torch.all(torch.isfinite(self._acentricFactor)).cpu().item()):
            raise ValueError("acentric factors must be finite")
        if not bool(torch.all(torch.isfinite(self._cubicKij)).cpu().item()):
            raise ValueError("cubicKij must be finite")
        if bool(torch.any(self._criticalTemp <= 0.0).cpu().item()):
            raise ValueError("critical temperatures must be positive")
        if bool(torch.any(self._criticalPres <= 0.0).cpu().item()):
            raise ValueError("critical pressures must be positive")
        if not bool(
            torch.allclose(
                self._cubicKij,
                self._cubicKij.mT,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ):
            raise ValueError("cubicKij must be symmetric")

        omegaA, omegaB, delta1, delta2 = cubicConstants(self.mode)
        self._omegaA = self.asTensor(omegaA)
        self._omegaB = self.asTensor(omegaB)
        self._delta1 = self.asTensor(delta1)
        self._delta2 = self.asTensor(delta2)
        self._m = self._buildMParameter()

        # Pure-component covolumes b_i [L/mol].
        self._bi = (
            self._omegaB
            * self.asTensor(self.gasConstant)
            * self._criticalTemp
            / self._criticalPres
        )

        # The correlation uses Tc [K], Pc [Pa] and volume [m^3/mol]; convert to L/mol.
        self._pseudoCriticalVolumes = LITERS_PER_CUBIC_METER * (
            BELL_VOLUME_SLOPE
            * self._criticalTemp
            / (PA_PER_BAR * self._criticalPres)
            + BELL_VOLUME_INTERCEPT
        )

    @property
    def supportsBatchedThermoState(self) -> bool:
        """
        Return that the closed-form cubic energy supports batched AD.
        """

        return True

    @property
    def componentCount(self) -> int:
        """
        Return the number of component density coordinates.
        """

        return int(self._criticalTemp.numel())

    @property
    def pureCriticalTemp(self) -> tuple[float, ...]:
        """
        Return the pure-component critical temperatures in kelvin.
        """

        return tuple(
            float(value)
            for value in self._criticalTemp.detach().cpu().tolist()
        )

    @property
    def pureCriticalPres(self) -> tuple[float, ...]:
        """
        Return the pure-component critical pressures in bar.
        """

        return tuple(
            float(value)
            for value in self._criticalPres.detach().cpu().tolist()
        )

    @property
    def pureAcentricFactor(self) -> tuple[float, ...]:
        """
        Return the pure-component acentric factors.
        """

        return tuple(
            float(value)
            for value in self._acentricFactor.detach().cpu().tolist()
        )

    @property
    def cubicKij(self) -> tuple[tuple[float, ...], ...]:
        """
        Return the constant symmetric cubic binary-interaction matrix.
        """

        return tuple(
            tuple(float(value) for value in row)
            for row in self._cubicKij.detach().cpu().tolist()
        )

    @property
    def tempScale(self) -> float:
        """
        Return the model temperature scale in K.
        """

        return float(torch.mean(self._criticalTemp).detach().cpu().item())

    @property
    def densityScale(self) -> float:
        """
        Return the model density scale in mol/L.
        """

        return float((1.0 / torch.mean(self._bi)).detach().cpu().item())

    def activeSubmodel(self, componentIndices: Sequence[int]) -> "CubicBellModel":
        """
        Return a PR/SRK model with the selected pure and binary parameters.
        """

        indices = tuple(int(index) for index in componentIndices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("componentIndices must be nonempty and unique")
        if any(index < 0 or index >= self.componentCount for index in indices):
            raise IndexError("active component index is outside the model")
        if indices == tuple(range(self.componentCount)):
            return self
        indexTensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        binary = self._cubicKij.index_select(0, indexTensor).index_select(
            1,
            indexTensor,
        )
        return CubicBellModel(
            self._criticalTemp.index_select(0, indexTensor)
            .detach()
            .cpu()
            .numpy(),
            self._criticalPres.index_select(0, indexTensor)
            .detach()
            .cpu()
            .numpy(),
            self._acentricFactor.index_select(0, indexTensor)
            .detach()
            .cpu()
            .numpy(),
            binary.detach().cpu().numpy(),
            mode=self.mode,
            dtype=self.dtype,
            device=self.device,
        )

    def compositionTempScale(self, composition: TensorLike) -> float:
        """
        Return the composition temperature scale in K.
        """

        fractions = self._normalizedComposition(composition)
        return float(torch.dot(fractions, self._criticalTemp).detach().cpu().item())

    def compositionDensityScale(self, composition: TensorLike) -> float:
        """
        Return a composition-dependent starting density [mol/L].
        Estimate pure critical volumes with Bell and Jäger (2017), Eq. (47), and
        linearly mix them before taking the reciprocal. The published SI volumes
        are converted to L/mol. This reducing scale initializes the search.
        """

        fractions = self._normalizedComposition(composition)
        volume = torch.dot(fractions, self._pseudoCriticalVolumes)
        return float((1.0 / volume).detach().cpu().item())

    def coVolume(self, composition: TensorLike) -> torch.Tensor:
        """
        Return the mixture covolume in L/mol.
        """

        fractions = self._normalizedComposition(composition)
        return torch.dot(fractions, self._bi)

    def densityRootUpperBound(
        self,
        temperature: TensorLike,
        composition: TensorLike,
    ) -> float:
        """
        Return the cubic hard-core bound used by generic cross-checks.
        """

        del temperature
        value = 0.995 / self.coVolume(composition)
        return float(value.detach().cpu().item())

    def mixtureParameters(
        self,
        temperature: TensorLike,
        composition: TensorLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return differentiable one-fluid parameters ``a(T,x)`` and ``b(x)``.
        """

        tempTensor = self.asTensor(temperature)
        fractions = self._normalizedComposition(composition)
        return self._mixtureParametersNormalized(tempTensor, fractions)

    def _mixtureParametersNormalized(
        self,
        temperature: torch.Tensor,
        fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate mixing rules without Python validation inside AD transforms.
        """

        reducedTemp = temperature / self._criticalTemp
        alpha = (1.0 + self._m * (1.0 - torch.sqrt(reducedTemp))) ** 2
        pureA = (
            self._omegaA
            * self.asTensor(self.gasConstant) ** 2
            * self._criticalTemp**2
            / self._criticalPres
            * alpha
        )
        crossA = (1.0 - self._cubicKij) * torch.sqrt(
            torch.outer(pureA, pureA)
        )
        return fractions @ crossA @ fractions, torch.dot(fractions, self._bi)

    def residualAlpha(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> torch.Tensor:
        r"""
        Return molar residual $\alpha^r=A^r/(nRT)$.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        composition = densityTensor / totalDensity
        attraction, coVolume = self._mixtureParametersNormalized(
            tempTensor, composition
        )
        packing = coVolume * totalDensity
        return _cubicResidualAlpha(
            packing, attraction, self.asTensor(self.gasConstant) * tempTensor,
            coVolume, self._delta1, self._delta2,
            logarithm=torch.log, logOnePlus=torch.log1p,
        )

    def freeEnergy(
        self,
        temperature: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Return $\Psi=\Psi^{id}+\Psi^r$ in bar.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        gasConstant = self.asTensor(self.gasConstant)
        ideal = gasConstant * tempTensor * torch.sum(
            densityTensor * (torch.log(densityTensor) - 1.0)
        )
        residual = (
            totalDensity
            * gasConstant
            * tempTensor
            * self.residualAlpha(tempTensor, densityTensor)
        )
        return ideal + residual

    def validateState(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> None:
        """
        Reject nonpositive or nonfinite states and excluded packing fractions.
        """

        super().validateState(temperature, densities)
        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        composition = densityTensor / totalDensity
        _attraction, coVolume = self._mixtureParametersNormalized(
            tempTensor, composition
        )
        hardCoreMargin = 1.0 - coVolume * totalDensity
        if hardCoreMargin.detach().cpu().item() <= 0.0:
            raise ValueError("state is at or beyond the cubic hard-core singularity")

    def domainMargin(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> torch.Tensor:
        """
        Return the dimensionless distance from the cubic packing pole.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        composition = densityTensor / totalDensity
        _attraction, coVolume = self._mixtureParametersNormalized(
            tempTensor, composition
        )
        hardCoreMargin = 1.0 - coVolume * totalDensity
        return torch.minimum(
            super().domainMargin(tempTensor, densityTensor),
            hardCoreMargin,
        )

    def directPres(
        self,
        temperature: TensorLike,
        densities: TensorLike,
    ) -> torch.Tensor:
        """
        Return the closed-form cubic pressure in bar.
        """

        tempTensor, densityTensor = self.stateTensors(temperature, densities)
        totalDensity = torch.sum(densityTensor)
        composition = densityTensor / totalDensity
        attraction, coVolume = self.mixtureParameters(tempTensor, composition)
        molarVolume = 1.0 / totalDensity
        return (
            self.asTensor(self.gasConstant) * tempTensor / (molarVolume - coVolume)
            - attraction
            / (
                (molarVolume + self._delta1 * coVolume)
                * (molarVolume + self._delta2 * coVolume)
            )
        )

    def _buildMParameter(self) -> torch.Tensor:
        omega = self._acentricFactor
        if self.mode == EOS_SRK:
            return 0.480 + 1.574 * omega - 0.176 * omega**2
        low = omega <= 0.49
        lowValue = 0.374640 + 1.54226 * omega - 0.26992 * omega**2
        highValue = (
            0.379642
            + 1.48503 * omega
            - 0.164423 * omega**2
            + 0.016666 * omega**3
        )
        return torch.where(low, lowValue, highValue)
