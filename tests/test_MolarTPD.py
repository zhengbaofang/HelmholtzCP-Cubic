"""
Check pressure acceptance and starting-root selection in joint TPD searches.
Use saved SRK starts and a PR state with two stable density roots.
Units: temperature [K], pressure [bar], density [mol/L], TPD [-].
"""


from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from HelmholtzCPCubic import CubicBellModel
from HelmholtzCPCubic.Equilibrium import findDensityRoots
from HelmholtzCPCubic.Equilibrium.DensityRoots import _rootEvaluationAt
from HelmholtzCPCubic.Equilibrium.MolarTPD import _molarDerivatives, findMolarTpdMinima
from HelmholtzCPCubic.Equilibrium.MolarTPD import _minimizeMolarPrepared
from HelmholtzCPCubic.Equilibrium.TPD import _referenceState
from HelmholtzCPCubic.Equilibrium.TPDMinimization import _TPDEvidence, _helmertBasis, _ilrCoordinates
from HelmholtzCPCubic.Kernel import ThermoKernel
from HelmholtzCPCubic.Numerics.ScalarTrustRegion import ScalarTrustRegionConfig


@pytest.mark.parametrize("diagonalScaling", [False, True])
@pytest.mark.parametrize("firstFraction,density", [
    (9.835002932585126e-7, 5.804988632820405),
    (0.0032370110959738267, 5.8198477370669544),
    (0.05, 6.043442612312318),
])
def test_saved_srk_starts_complete_pressure_qualification(firstFraction, density, diagonalScaling):
    """
    Check three CO2/n-octane SRK starts with diagonal scaling on and off.
    Use the same pressure, composition-gradient, curvature and TPD requirements.
    Inputs use K, bar and mol/L; density-root discovery is not repeated.
    """

    model = CubicBellModel(
        criticalTemp=(304.13, 568.74), criticalPres=(73.77, 24.84),
        acentricFactor=(0.225, 0.3975), cubicKij=((0.0, 0.0), (0.0, 0.0)), mode="SRK",
    )
    temperature, pressure = 141.41916779600544, 13.49502417618396
    feed = np.array([0.9147, 0.0853])
    start = np.array([firstFraction, 1.0 - firstFraction])
    referenceLnPhi, _, _, _ = _referenceState(
        model, temperature, pressure, feed, referenceDensityMolL=21.572370586542664,
    )
    constant = np.log(pressure / (model.gasConstant * temperature)) + np.log(feed) + referenceLnPhi
    basis, kernel = _helmertBasis(2), ThermoKernel(model)
    initial = _rootEvaluationAt(
        model, kernel, model.asTensor(temperature), temperature, pressure,
        start, density, withHessian=False,
    )
    result = _minimizeMolarPrepared(
        model, kernel, temperature, pressure, start, initial, constant, basis,
        ScalarTrustRegionConfig(diagonalScaling=diagonalScaling), _TPDEvidence(stopBelow=-1e-10),
    )
    assert result.converged, result.termination
    assert result.stationarityNorm <= 1e-8
    assert result.value >= -1e-10

    # Independently re-evaluate the returned pressure root in physical units.
    final = _rootEvaluationAt(
        model, kernel, model.asTensor(temperature), temperature, pressure,
        np.asarray(result.composition), result.density, withHessian=False,
    )
    assert final.root.mechanicalStable
    assert abs(final.root.pressureResidual) <= 1e-8


@pytest.mark.parametrize("qualifiedReference", (False, True))
def test_all_stable_density_roots_enter_the_same_local_solver(monkeypatch, qualifiedReference):
    """
    Retain the other stable density root when the reference root is skipped.
    This test checks start selection; qualifiedReference is supplied only to
    exercise that code path, not to classify this fixture as a critical point.
    """

    model = CubicBellModel(
        criticalTemp=(190.56, 305.32), criticalPres=(45.99, 48.72),
        acentricFactor=(0.0115, 0.098), cubicKij=((0.0, 0.0), (0.0, 0.0)), mode="PR",
    )
    feed = np.array((0.5, 0.5))
    roots = findDensityRoots(model, 220.0, 20.0, feed)
    stable = tuple(root for root in roots if root.mechanicalStable)
    assert len(stable) == 2
    module = importlib.import_module("HelmholtzCPCubic.Equilibrium.MolarTPD")
    original = module._minimizeMolarPrepared
    visited = []

    def observe(*args, **kwargs):
        visited.append(args[5].root.density)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_minimizeMolarPrepared", observe)
    reference = min(stable, key=lambda root: root.gibbsLike)
    result = findMolarTpdMinima(
        model, feed, 220.0, 20.0, starts=(feed,),
        referenceDensityMolL=reference.density,
        config=ScalarTrustRegionConfig(maxEval=1),
        qualifiedReference=qualifiedReference,
    )
    expected = [root.density for root in stable if not qualifiedReference or root is not reference]
    np.testing.assert_allclose(visited, expected, atol=0.0, rtol=0.0)
    assert result.attemptedStarts == 2
    assert not result.completed  # maxEval=1 deliberately prevents a complete search.

