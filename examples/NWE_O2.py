"""
NWE oil/injection-gas blend NWE_O2: fixed-composition PR critical-point calculation.
Source: Khan, Pope and Sepehrnoori (1992), Fluid Characterization of Three-Phase CO2/Oil
Mixtures, SPE-24130-MS.
Run from the repository root: python examples/NWE_O2.py
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
componentNames = ('CO2', 'C1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6')
fluidProperties = np.array([
    [304.2, 73.7646020094375, 0.225],  # CO2
    [190.6, 46.00155176456252, 0.008],  # C1
    [343.63722222222225, 45.04848256916896, 0.1301],  # HC2
    [466.40999999999997, 33.50431464290596, 0.2436],  # HC3
    [603.0722222222222, 24.237485050580315, 0.6],  # HC4
    [733.7866666666666, 18.030762482444036, 0.903],  # HC5
    [923.1989444444445, 17.257956716480663, 1.229],  # HC6
], dtype=float)

# Mole fractions [-], in the same component order.

# Oil and injection-gas mole fractions from the same source.
oilComposition = np.array([
    0.0077, 0.2025, 0.118, 0.1484, 0.2863, 0.149, 0.0881
], dtype=float)
gasComposition = np.array([
    0.95, 0.05, 0, 0, 0, 0, 0
], dtype=float)

# Prescribed injection-gas fraction from Supplementary Table S2.
mixingFraction = 0.5716123511768751
composition = ((1.0 - mixingFraction) * oilComposition / oilComposition.sum()
               + mixingFraction * gasComposition / gasComposition.sum())

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, 0.12, 0.12, 0.12, 0.09, 0.09, 0.09],
    [0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    print('Running NWE_O2...', flush=True)
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
