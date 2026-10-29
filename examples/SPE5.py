"""
SPE5 reservoir oil: fixed-composition PR critical-point calculation.
Source: Killough and Kossack (1987), SPE5.
Run from the repository root: python examples/SPE5.py
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
componentNames = ('HC1', 'HC2', 'HC3', 'HC4', 'HC5', 'HC6')
fluidProperties = np.array([
    [190.56, 46.0522, 0.013],  # HC1
    [369.83, 42.5058, 0.1524],  # HC2
    [507.44, 30.1341, 0.3007],  # HC3
    [617.67, 20.9641, 0.4885],  # HC4
    [705.56, 13.7903, 0.65],  # HC5
    [766.67, 11.1721, 0.85],  # HC6
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.5, 0.03, 0.07, 0.2, 0.15, 0.05
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, 0.0, 0.0, 0.05, 0.05, 0.0],
    [0.0, 0.0, 0.0, 0.005, 0.005, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.005, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.005, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    print('Running SPE5...', flush=True)
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
