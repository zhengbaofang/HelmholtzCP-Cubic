"""
Check the shared PR/SRK gas constant in all 22 example models.
Compare direct and Helmholtz-derived pressures for two methane--ethane states.
Units: temperature [K], density [mol/L], pressure [bar].
"""

from importlib import import_module

import numpy as np
import pytest

from HelmholtzCPCubic.Kernel.Thermo import ThermoKernel
from HelmholtzCPCubic.Transform.CubicBell import R_GAS
from examples import CASE_NAMES


@pytest.mark.parametrize("name", CASE_NAMES)
def test_paper_cases_use_shared_gas_constant(name):
    """
    Ensure no example overrides R with a historical per-case value.
    """

    model, _composition, _config = import_module(f"examples.{name}").makeCase()
    assert model.gasConstant == R_GAS == 0.083145


@pytest.mark.parametrize("name", ("C1C2_PR", "C1C2_SRK"))
def test_shared_gas_constant_preserves_pressure_consistency(name):
    """
    Compare pressure routes at T=250 K and total density 5 mol/L.
    """

    model, composition, _config = import_module(f"examples.{name}").makeCase()
    temperature = 250.0
    density = 5.0
    thermo = ThermoKernel(model)
    helmholtzPressure = float(thermo.pressure(temperature, density * composition))
    cubicPressure = float(model.directPres(temperature, density * composition))
    np.testing.assert_allclose(helmholtzPressure, cubicPressure,
                               rtol=1.0e-12, atol=1.0e-10)
