"""
Check the fourth-order correction for a curved critical eigendirection.
Use exact dimensionless polynomial models and selected PR/SRK fluid states.
Fluid temperatures and densities use K and mol/L.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from HelmholtzCPCubic import CubicBellModel, CriticalPointSolver
from HelmholtzCPCubic.Common import FreeEnergyModel
from HelmholtzCPCubic.Kernel import LocalStabilityKernel, ThermoKernel


class CurvedCriticalModel(FreeEnergyModel):
    r"""
    Polynomial model with an exact ordinary critical point at $(1,1)$.

    With ``p=rho1-1`` and ``q=rho2-1``, the local free energy is

    ``Psi=(T-1)p^2/2 + lambda*q^2/2 + g*p^2*q/2 + raw*p^4/24``.

    The minimum eigendirection is ``e1`` and the exact corrected fourth
    derivative is ``raw - 3*g**2/lambda``. This separates the raw straight
    direction from the curved minimum-minEigenvector path.
    """

    def __init__(self, rawFourth: float, coupling: float = 1.0) -> None:
        super().__init__(gasConstant=1.0)
        self.rawFourth = float(rawFourth)
        self.coupling = float(coupling)
        self.stableEigenvalue = 2.0

    @property
    def componentCount(self) -> int:
        """
        Return the two density coordinates used by the polynomial.
        """

        return 2

    @property
    def tempScale(self) -> float:
        """
        Return the exact critical-temperature scale.
        """

        return 1.0

    @property
    def densityScale(self) -> float:
        """
        Return the exact critical total-density scale.
        """

        return 2.0

    def freeEnergy(
        self,
        temperature: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the analytic local free-energy density.
        """

        p = densities[0] - 1.0
        q = densities[1] - 1.0
        return (
            0.5 * (temperature - 1.0) * p**2
            + 0.5 * self.stableEigenvalue * q**2
            + 0.5 * self.coupling * p**2 * q
            + self.rawFourth * p**4 / 24.0
        )


def test_curved_direction_correction_matches_exact_polynomial() -> None:
    """
    Compare the curved-path fourth-order correction with the exact polynomial.
    """

    model = CurvedCriticalModel(rawFourth=4.0)
    result = LocalStabilityKernel(model).evaluate(1.0, 2.0, (0.5, 0.5))

    expectedCorrection = -3.0 * model.coupling**2 / model.stableEigenvalue
    np.testing.assert_allclose(result.qMin, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.cStar, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.direction, (1.0, 0.0), atol=1.0e-14)
    np.testing.assert_allclose(
        result.directionCorrection,
        (0.0, -model.coupling / model.stableEigenvalue),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(result.rawFourthDirectional, 4.0, atol=1.0e-13)
    np.testing.assert_allclose(
        result.correctionContribution,
        expectedCorrection,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        result.correctedFourthDirectional,
        4.0 + expectedCorrection,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        result.dStarDensity,
        (4.0 + expectedCorrection) / 24.0,
        atol=1.0e-14,
    )
    assert result.isLocallyStable is True


@pytest.mark.parametrize(
    ("rawFourth", "expectedClassification"),
    ((1.0, False), (1.5, None), (4.0, True)),
)
def test_stability_classification_uses_corrected_fourth_order(
    rawFourth: float,
    expectedClassification: bool | None,
) -> None:
    """
    Check that classification uses the corrected fourth-order coefficient.
    """

    result = LocalStabilityKernel(CurvedCriticalModel(rawFourth)).evaluate(
        1.0,
        2.0,
        (0.5, 0.5),
    )
    assert result.isLocallyStable is expectedClassification


def test_fourth_order_classification_is_direction_sign_invariant() -> None:
    """
    Check that reversing the eigenvector leaves the stability result unchanged.
    """

    kernel = LocalStabilityKernel(CurvedCriticalModel(rawFourth=4.0))
    positive = kernel.evaluate(1.0, 2.0, (0.5, 0.5), direction=(1.0, 0.0))
    negative = kernel.evaluate(1.0, 2.0, (0.5, 0.5), direction=(-1.0, 0.0))

    np.testing.assert_allclose(
        negative.correctedFourthDirectional,
        positive.correctedFourthDirectional,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        negative.directionCorrection,
        positive.directionCorrection,
        atol=1.0e-14,
    )
    assert negative.isLocallyStable is positive.isLocallyStable


def test_local_stability_rejects_noncritical_or_repeated_eigenspace() -> None:
    """
    Reject noncritical states and unresolved repeated eigenvalues.
    """

    kernel = LocalStabilityKernel(CurvedCriticalModel(rawFourth=4.0))
    with pytest.raises(ValueError, match="corrected critical state"):
        kernel.evaluate(1.1, 2.0, (0.5, 0.5))

    repeated = CurvedCriticalModel(rawFourth=4.0)
    repeated.stableEigenvalue = 0.0
    with pytest.raises(ValueError, match="eigenspace is unresolved"):
        LocalStabilityKernel(repeated).evaluate(1.0, 2.0, (0.5, 0.5))


def test_srk_binary_root_has_finite_local_stability_measure() -> None:
    """
    Check the fourth-order measure at an SRK binary critical point [K, mol/L].
    """

    model = CubicBellModel(
        criticalTemp=(304.13, 190.56),
        criticalPres=(73.77, 45.99),
        acentricFactor=(0.2249, 0.0115),
        cubicKij=((0.0, 0.09), (0.09, 0.0)),
        mode="SRK",
    )
    composition = np.array([0.5, 0.5])
    solver = CriticalPointSolver(model)
    point = solver.correct(composition, 250.0, 8.0)
    result = LocalStabilityKernel(model).evaluate(point.T, point.rho, composition)

    assert point.converged
    assert np.isfinite(result.correctedFourthDirectional)
    assert result.eigenvalueGap > 0.0
    assert result.isLocallyStable is True

    thermo = ThermoKernel(model)
    baseDensities = point.rho * composition
    referenceDirection = np.asarray(result.direction)

    def minimumDirection(densities: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Return the lowest raw-Hessian direction with a continuous sign.
        """

        hessian = thermo.hessian(point.T, densities).detach().cpu().numpy()
        candidate = np.linalg.eigh(hessian)[1][:, 0]
        return candidate if candidate @ reference >= 0.0 else -candidate

    def eigenvalueAfterPathStep(offset: float) -> float:
        """
        Integrate one Deiters--Bell minEigenvector-path step with RK4.
        """

        k1 = minimumDirection(baseDensities, referenceDirection)
        k2 = minimumDirection(baseDensities + 0.5 * offset * k1, k1)
        k3 = minimumDirection(baseDensities + 0.5 * offset * k2, k2)
        k4 = minimumDirection(baseDensities + offset * k3, k3)
        endpoint = baseDensities + offset * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        hessian = thermo.hessian(point.T, endpoint).detach().cpu().numpy()
        return float(np.linalg.eigvalsh(hessian)[0])

    pathStep = 1.0e-2
    sampled = np.asarray(
        [
            eigenvalueAfterPathStep(-2.0 * pathStep),
            eigenvalueAfterPathStep(-pathStep),
            result.eigenvalues[0]
            * model.gasConstant
            * point.T
            / point.rho,
            eigenvalueAfterPathStep(pathStep),
            eigenvalueAfterPathStep(2.0 * pathStep),
        ]
    )
    pathFiniteDifference = (
        -sampled[4]
        + 16.0 * sampled[3]
        - 30.0 * sampled[2]
        + 16.0 * sampled[1]
        - sampled[0]
    ) / (12.0 * pathStep**2)
    np.testing.assert_allclose(
        result.correctedFourthDirectional,
        pathFiniteDifference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    ("temperature0", "componentDensities0", "expectedStable"),
    (
        (240.21520165269916, (7.186321953929991, 0.6293712312974595), True),
        (240.21803987926600, (7.215126134385907, 0.6377571745643125), False),
    ),
)
def test_r14_butane_branch_changes_stability_at_temperature_cusp(
    temperature0: float,
    componentDensities0: tuple[float, float],
    expectedStable: bool,
) -> None:
    """
    Cross-check Deiters--Bell Figure 5(b) without reading derived CSV data.
    """

    model = CubicBellModel(
        criticalTemp=(227.5, 423.675),
        criticalPres=(37.389, 36.348),
        acentricFactor=(0.18884, 0.19290),
        cubicKij=((0.0, 0.21901), (0.21901, 0.0)),
        mode="PR",
    )
    densities0 = np.asarray(componentDensities0)
    density0 = float(np.sum(densities0))
    composition = densities0 / density0
    point = CriticalPointSolver(model).correct(
        composition,
        temperature0,
        density0,
    )
    stability = LocalStabilityKernel(model).evaluate(
        point.T,
        point.rho,
        composition,
    )

    assert point.converged
    assert stability.isLocallyStable is expectedStable
    assert (stability.correctedFourthDirectional > 0.0) is expectedStable
