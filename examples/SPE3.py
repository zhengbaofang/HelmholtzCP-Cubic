"""
SPE3 gas condensate: fixed-composition PR critical-point calculation.
Source: Kenyon and Behie (1987), SPE3 Table 17, ARCO characterization.
Run from the repository root: python examples/SPE3.py
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
componentNames = ('CO2', 'N2', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6', 'HC7')
fluidProperties = np.array([
    [304.2, 73.865925, 0.225],  # CO2
    [126.2, 33.943875, 0.04],  # N2
    [186.6, 46.2042, 0.013],  # HC1
    [305.4, 48.83865, 0.098],  # HC2
    [369.9, 42.5565, 0.152],  # HC3
    [396.2, 34.349174999999995, 0.234],  # HC4
    [572.5, 25.9392, 0.332],  # HC5
    [630.2, 16.921274999999998, 0.495],  # HC6
    [862.6, 8.612625, 0.833],  # HC7
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.0121, 0.0194, 0.6599, 0.0869, 0.0591, 0.0967, 0.0472, 0.0153, 0.0034
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, -0.02, 0.1, 0.13, 0.135, 0.1277, 0.1, 0.1, 0.1],
    [-0.02, 0.0, 0.036, 0.05, 0.08, 0.1002, 0.1, 0.1, 0.1],
    [0.1, 0.036, 0.0, 0.0, 0.0, 0.09281, 0.0, 0.0, 0.1392],
    [0.13, 0.05, 0.0, 0.0, 0.0, 0.0, 0.00385, 0.0063, 0.006],
    [0.135, 0.08, 0.0, 0.0, 0.0, 0.0, 0.00385, 0.0063, 0.006],
    [0.1277, 0.1002, 0.09281, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.0, 0.00385, 0.00385, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.0, 0.0063, 0.0063, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1392, 0.006, 0.006, 0.0, 0.0, 0.0, 0.0],
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
    print('Running SPE3...', flush=True)
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
