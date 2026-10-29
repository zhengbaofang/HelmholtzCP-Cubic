"""
OilG0.4 oil/gas fluid: fixed-composition PR critical-point calculation.
Source: Xu (2023), Development of Robust Trust-Region Algorithms for Multiphase Equilibrium
Calculations, University of Alberta, Table 4-A1.
Run from the repository root: python examples/OilG04.py
"""

from pathlib import Path

import numpy as np
import torch

from HelmholtzCPCubic import CubicBellModel, CriticalPointSolver, AdaptiveBoundaryBellFinder
from HelmholtzCPCubic.Application import (
    CriticalSolverConfig, CriticalDiscoveryConfig, LinearForecastStepConfig,
    TPDStabilityClassifier,
)


# Import the shared plotter for direct execution and package imports.
if __package__:
    from .PlotResults import plotResults
else:
    from PlotResults import plotResults


# Fluid inputs: Tc [K], Pc [bar], acentric factor [-].
eos = 'PR'

# HC labels denote characterized fractions, not assigned carbon-number cuts.
componentNames = ('CO2', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6')
fluidProperties = np.array([
    [304.2, 73.76, 0.225],  # CO2
    [174.44, 46.0, 0.008],  # HC1
    [347.26, 44.69, 0.1331],  # HC2
    [459.74, 34.18, 0.2358],  # HC3
    [595.14, 21.87, 0.5977],  # HC4
    [729.98, 16.04, 0.9118],  # HC5
    [910.18, 15.21, 1.2444],  # HC6
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.41014, 0.10512, 0.13464, 0.10038, 0.14532, 0.07296, 0.03144
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, 0.085, 0.085, 0.085, 0.104, 0.104, 0.104],
    [0.085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.085, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=float)

# Search controls. None retains the composition-dependent default bounds.
solverConfig = CriticalSolverConfig(
    temperatureBoundsK=None, densityBoundsMolL=None,
    resTol=1.0e-8, stepTol=1.0e-10,
)

# Dimensionless continuation steps in logarithmic temperature/density coordinates.
stepConfig = LinearForecastStepConfig(
    conservativeMaximumStep=0.04, nominalMaximumStep=0.12,
)

# Initial and minimum tracing steps; the maximum follows the nominal step cap.
traceConfig = CriticalDiscoveryConfig(
    initialStep=0.03, minimumStep=0.0025,
    maximumStep=stepConfig.nominalMaximumStep,
)


def makeCase():
    """
    Build this EOS, normalized composition and solver controls in K/bar/mol/L.
    """

    criticalTemp, criticalPres, acentricFactor = fluidProperties.T
    model = CubicBellModel(criticalTemp, criticalPres, acentricFactor,
                           binaryInteractions, mode=eos)
    return model, composition / composition.sum(), solverConfig


def main(saveFigure=True):
    """
    Find and screen critical points, print results and optionally save a PNG.
    Set saveFigure=False for numerical output only. Return 0 on completion.
    """

    torch.set_num_threads(1)
    model, fractions, config = makeCase()
    solver = CriticalPointSolver(model, config)
    finder = AdaptiveBoundaryBellFinder(solver, traceConfig, stepConfig=stepConfig)
    print('Running OilG04...', flush=True)
    discovery = finder.findBoundaryContours(fractions)

    classifier = TPDStabilityClassifier(model)
    assessments = []
    completed = discovery.completed
    print('T [K]          P [bar]        rho [mol/L]    TPD status   lowest TPD')
    for point in discovery.roots:
        assessment = classifier.evaluate(point)
        assessments.append(assessment)
        print(f'{point.T:14.8f} {point.P:14.8f} {point.rho:14.8f} '
              f'{assessment.status.value:12s} {assessment.lowestTpdValue}', flush=True)
        if assessment.failureReason:
            print(assessment.failureReason)
        completed = completed and assessment.status.value != 'failed'
    for failure in discovery.boundaryFailures:
        print(failure)
    print(f'Candidates: {len(discovery.roots)}; completed: {completed}')

    # Plot the existing paths and screening results without repeating the search.
    if saveFigure:
        outputPath = Path(__file__).resolve().parents[1] / "output" / f"{Path(__file__).stem}.png"
        figurePath = plotResults(model, fractions, discovery, assessments, outputPath)
        print(f"Figure: {figurePath}")
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
