"""
Mix-2: fixed-composition SRK critical-point calculation.
Source: Venkatarathnam (2014), Mix-2 composition; CoolProp 8.0.0 pure-component properties;
Saber (2011) binary interaction parameters.
Run from the repository root: python examples/Mix2.py
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
eos = 'SRK'
componentNames = ('Nitrogen', 'Methane', 'Ethane', 'Propane', 'n-Butane', 'n-Pentane')
fluidProperties = np.array([
    [126.19199999958556, 33.95800444647145, 0.0372],  # Nitrogen
    [190.56400265128698, 45.99200474282439, 0.01142],  # Methane
    [305.3220000000155, 48.72199977781725, 0.099],  # Ethane
    [369.8900089509634, 42.51165328013042, 0.1521],  # Propane
    [425.125000000008, 37.9600001741581, 0.200810094644],  # n-Butane
    [469.69999987111663, 33.67518983606875, 0.251031912680427],  # n-Pentane
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.304, 0.548, 0.071, 0.037, 0.02, 0.02
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, 0.02, 0.06, 0.08, 0.08, 0.08],
    [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    print('Running Mix2...', flush=True)
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
