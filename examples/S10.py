"""
S10 Stenby reservoir oil: fixed-composition PR critical-point calculation.
Source: Nichita, Broseta and Leibovici (2007), Tables 1 and 2; Stenby scheme D.
Run from the repository root: python examples/S10.py
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
componentNames = ('N2', 'CO2', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7', 'HC8')
fluidProperties = np.array([
    [126.2, 33.6, 0.04],  # N2
    [304.2, 72.9, 0.23],  # CO2
    [190.6, 45.4, 0.01],  # HC1
    [305.4, 48.2, 0.1],  # HC2
    [369.8, 41.9, 0.15],  # HC3
    [507.4, 29.3, 0.3],  # HC4
    [416.3, 36.72, 0.1848],  # HC5
    [463.31, 33.371, 0.23627],  # HC6
    [644.841, 21.36, 0.68836],  # HC7
    [886.863, 14.453, 1.19528],  # HC8
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.004030040300403004, 0.01000010000100001, 0.45396453964539646, 0.04202042020420205,
    0.008870088700887008, 0.01011010110101101, 0.01079010790107901, 0.00941009410094101,
    0.3344833448334484, 0.11632116321163213
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, 0.0, 0.02, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.0, 0.0, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    [0.02, 0.12, 0.0, 0.0, 0.0, 0.0298, 0.0, 0.0, 0.045522, 0.062802],
    [0.06, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.0298, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.045522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.15, 0.062802, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    print('Running S10...', flush=True)
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
