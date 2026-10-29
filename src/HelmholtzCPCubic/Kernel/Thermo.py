r"""
Calculate pressure, chemical potentials and Hessians from Helmholtz energy.
Reuse lower-order quantities during joint evaluation and check pressure first.
Provide scalar pressure callbacks for fixed-temperature, fixed-composition scans.
All quantities use the model's declared units.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch

from ..Common.Model import FreeEnergyModel, TensorLike


@dataclass
class ThermoState:
    """
    Thermodynamic values and derivatives at one component-density state.
    For a K/bar/mol/L model: freeEnergy and pressure [bar], chemicalPotentials
    [bar L/mol] and hessian [bar L^2/mol^2]. Other models retain their native units.
    """

    freeEnergy: torch.Tensor
    chemicalPotentials: torch.Tensor
    pressure: torch.Tensor
    hessian: torch.Tensor


class ThermoKernel:
    r"""
    Generate $\mu_i$, $P$ and $\partial^2\Psi/\partial\rho_i\partial\rho_j$.
    """

    def __init__(self, model: FreeEnergyModel) -> None:
        """
        Prepare component-density gradient and Hessian functions for this model.
        """

        self.model = model
        self._chemicalPotential = torch.func.grad(model.freeEnergy, argnums=1)

        def energyWithValue(temperature: torch.Tensor, densities: torch.Tensor):
            """
            Retain the potential while computing its component gradient.
            """

            value = model.freeEnergy(temperature, densities)
            return value, value

        chemicalPotentialWithValue = torch.func.grad(
            energyWithValue, argnums=1, has_aux=True,
        )
        self._chemicalPotentialWithValue = chemicalPotentialWithValue

        # Differentiate the chemical-potential gradient to obtain the density Hessian.
        self._hessian = torch.func.jacrev(
            self._chemicalPotential,
            argnums=1,
        )

        def chemicalWithValues(temperature: torch.Tensor, densities: torch.Tensor):
            """
            Carry energy and chemical potentials through the density Jacobian.
            """

            # Capture the transform directly, avoiding a kernel-to-closure-to-
            # kernel reference cycle while sharing the same derivative function.
            chemical, energy = chemicalPotentialWithValue(temperature, densities)
            return chemical, (energy, chemical)

        # Differentiate only mu; auxiliary outputs retain Psi and mu from the
        # same EOS evaluation without detaching component-density dependence.
        self._hessianWithValues = torch.func.jacrev(
            chemicalWithValues, argnums=1, has_aux=True,
        )

    def fixedCompositionPressureFunctions(
        self,
        temperature: TensorLike,
        composition: TensorLike,
    ) -> tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ]:
        r"""
        Prepare pressure-only and joint pressure/slope scalar callbacks.

        Temperature is K and composition is a normalized positive mole-fraction
        vector. Each callback takes scalar total molar density in model units;
        it returns pressure, or (pressure, dP/drho), in the same native units.
        With q(rho)=Psi(T,rho*y), P=rho*q'-q and dP/drho=rho*q''.
        Nested scalar reverse AD retains q and q' as auxiliary outputs, so a
        joint query does not separately rebuild a component gradient or Hessian.

        Invalid temperature/composition raises ValueError. Density-domain and
        final pressure checks remain with the root caller: intermediate scan
        states can have negative pressure. Callbacks support torch.vmap when
        the underlying model does. They are bound to this temperature and
        composition only, with no cache shared between different compositions.
        """

        temp = self.model.asTensor(temperature)
        fractions = self.model.asTensor(composition)
        if temp.ndim != 0 or not bool(torch.isfinite(temp)) or float(temp) <= 0:
            raise ValueError("temperature must be a finite positive scalar")
        if (fractions.shape != (self.model.componentCount,)
                or not bool(torch.all(torch.isfinite(fractions)))
                or not bool(torch.all(fractions > 0))
                or abs(float(fractions.sum()) - 1.0) > 1.0e-12):
            raise ValueError("composition must contain normalized positive mole fractions")

        if self.model.componentCount == 1:

            # Use the same one-variable operation order for pure fluids so
            # rounding does not change the sign of a near-zero mechanical slope.
            def purePressure(density: torch.Tensor):
                """
                Keep the original one-component pressure operation order.
                """

                return self.pressure(temp, density * fractions, validate=False)

            def purePressureAndSlope(density: torch.Tensor):
                """
                Retain pure-fluid tangent-root rounding and slope semantics.
                """

                densities = density * fractions
                return (self.pressure(temp, densities, validate=False),
                        self.pressureDensityDerivative(temp, densities, validate=False))

            return purePressure, purePressureAndSlope

        energyAt = self.model.fixedCompositionFreeEnergy(temp, fractions)

        def energyWithValue(density: torch.Tensor):
            """
            Carry the energy value out of the first reverse derivative.
            """

            value = energyAt(density)
            return value, value

        firstWithValue = torch.func.grad(energyWithValue, has_aux=True)

        def firstWithAux(density: torch.Tensor):
            """
            Keep both lower-order values while differentiating once more.
            """

            first, value = firstWithValue(density)
            return first, (first, value)

        secondWithAux = torch.func.grad(firstWithAux, has_aux=True)

        def pressureAt(density: torch.Tensor) -> torch.Tensor:
            """
            Compute pressure without requesting an unnecessary second derivative.
            """

            first, value = firstWithValue(density)
            return density * first - value

        def pressureAndSlopeAt(density: torch.Tensor):
            """
            Assemble pressure and slope from one nested scalar AD evaluation.
            """

            second, (first, value) = secondWithAux(density)
            return density * first - value, density * second

        return pressureAt, pressureAndSlopeAt

    def chemicalPotentials(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        """
        Return component chemical potentials from the density gradient.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
        return self._chemicalPotential(tempTensor, densityTensor)

    def hessian(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        """
        Return the symmetric component-density Hessian.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
        hessian = self._hessian(tempTensor, densityTensor)
        return 0.5 * (hessian + hessian.mT)

    def chemicalPotentialsAndPressure(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (chemical potentials, pressure) without a full Hessian.

        Inputs and outputs use the model's native units. The potential is
        retained by the first-derivative transform, avoiding a separate EOS
        call for P = densities @ chemicalPotentials - freeEnergy. Validation
        checks the model domain and pressure; callers check derivative
        finiteness. This also supports vmap for models declaring batched AD.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
        chemicalPotentials, freeEnergy = self._chemicalPotentialWithValue(
            tempTensor, densityTensor,
        )
        pressure = torch.dot(densityTensor, chemicalPotentials) - freeEnergy
        if validate:
            self.model.validatePres(pressure)
        return chemicalPotentials, pressure

    def pressure(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        r"""
        Return $P=\boldsymbol\rho^T\boldsymbol\mu-\Psi$.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
        freeEnergy = self.model.freeEnergy(tempTensor, densityTensor)
        chemicalPotentials = self._chemicalPotential(tempTensor, densityTensor)
        pressure = torch.dot(densityTensor, chemicalPotentials) - freeEnergy
        if validate:
            self.model.validatePres(pressure)
        return pressure

    def pressureDensityDerivative(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        r"""
        Return dP/drho along the fixed-composition density direction.

        Component densities and the returned derivative use the model's
        native density and pressure units.  A Hessian--vector product computes
        $dP/d\rho=\rho\,\mathbf y^T\mathcal H\mathbf y$ without materializing
        the complete component-density Hessian.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
        totalDensity = torch.sum(densityTensor)
        fractions = densityTensor / totalDensity
        _chemicalPotentials, hessianTimesDensity = torch.func.jvp(
            lambda localDensities: self._chemicalPotential(
                tempTensor,
                localDensities,
            ),
            (densityTensor,),
            (densityTensor,),
        )
        derivative = torch.dot(fractions, hessianTimesDensity)
        if validate and not bool(torch.isfinite(derivative).detach().cpu().item()):
            raise FloatingPointError("pressure-density derivative must be finite")
        return derivative

    def state(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> ThermoState:
        """
        Return energy, chemical potentials, pressure and density Hessian.

        Inputs and outputs use the model's native units. Without validation,
        all values share one nested AD evaluation. With validation, pressure
        is checked after first derivatives and before requesting the Hessian,
        preserving the original failure order. Model domain/pressure errors
        and derivative failures propagate to the caller.
        """

        tempTensor, densityTensor = self.model.stateTensors(temperature, densities)
        if validate:
            self.model.validateState(tempTensor, densityTensor)
            chemicalPotentials, freeEnergy = self._chemicalPotentialWithValue(
                tempTensor, densityTensor,
            )
            pressure = torch.dot(densityTensor, chemicalPotentials) - freeEnergy
            self.model.validatePres(pressure)
            hessian = self._hessian(tempTensor, densityTensor)
        else:
            hessian, (freeEnergy, chemicalPotentials) = self._hessianWithValues(
                tempTensor, densityTensor,
            )
            pressure = torch.dot(densityTensor, chemicalPotentials) - freeEnergy
        hessian = 0.5 * (hessian + hessian.mT)
        return ThermoState(
            freeEnergy=freeEnergy,
            chemicalPotentials=chemicalPotentials,
            pressure=pressure,
            hessian=hessian,
        )

    def stateBatch(
        self,
        temperature: TensorLike,
        densities: TensorLike,
        *,
        validate: bool = True,
    ) -> ThermoState:
        """
        Evaluate full states for one temperature and many density vectors.

        Args:
            temperature: Shared scalar temperature in the model unit system.
            densities: Component densities with shape
                ``(stateCount, componentCount)`` in the model density unit.
            validate: Whether to validate each input state and output pressure.

        Returns:
            Batched tensors whose leading dimension follows ``densities``.

        Raises:
            NotImplementedError: If the model does not declare batched
                higher-order automatic differentiation support.
            ValueError: If temperature or density shapes are inconsistent, or
                model validation rejects an input or output state.
            RuntimeError: If the declared batched AD operations fail.
        """

        if not self.model.supportsBatchedThermoState:
            raise NotImplementedError(
                f"{type(self.model).__name__} does not declare batched "
                "thermodynamic-state support"
            )
        tempTensor = self.model.asTensor(temperature)
        densityTensor = self.model.asTensor(densities)
        if tempTensor.ndim != 0:
            raise ValueError(
                "batched thermodynamic-state temperature must be scalar"
            )
        if densityTensor.ndim != 2 or densityTensor.shape[1] != (
            self.model.componentCount
        ):
            raise ValueError(
                "batched thermodynamic-state densities must have shape "
                "(stateCount, componentCount)"
            )
        if densityTensor.shape[0] == 0:
            raise ValueError(
                "batched thermodynamic-state densities cannot be empty"
            )
        if validate:
            for localDensities in densityTensor:
                self.model.validateState(tempTensor, localDensities)

        def stateValues(
            localDensities: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Return one state as tensors accepted by ``torch.vmap``.
            """

            hessian, (freeEnergy, chemicalPotentials) = self._hessianWithValues(
                tempTensor,
                localDensities,
            )
            pressure = (
                torch.dot(localDensities, chemicalPotentials) - freeEnergy
            )
            return freeEnergy, chemicalPotentials, pressure, 0.5 * (
                hessian + hessian.mT
            )

        freeEnergy, chemicalPotentials, pressure, hessian = torch.vmap(
            stateValues
        )(densityTensor)
        if validate:
            for localPressure in pressure:
                self.model.validatePres(localPressure)
        return ThermoState(
            freeEnergy=freeEnergy,
            chemicalPotentials=chemicalPotentials,
            pressure=pressure,
            hessian=hessian,
        )
