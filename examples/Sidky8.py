"""
Sidky eight-component case: fixed-composition SRK critical-point calculation.
Source: Sidky et al. (2016).
Run from the repository root: python examples/Sidky8.py
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

# HC labels denote characterized fractions, not assigned carbon-number cuts.
componentNames = ('C1', 'C2', 'C3', 'nC4', 'nC5', 'nC6', 'HC1', 'HC2')
fluidProperties = np.array([
    [190.555, 45.98837, 0.01131],  # C1
    [305.4, 48.839, 0.098],  # C2
    [369.8, 42.455, 0.152],  # C3
    [425.2, 37.997, 0.193],  # nC4
    [469.6, 33.741, 0.251],  # nC5
    [507.4, 29.688, 0.296],  # nC6
    [606.28, 25.756815, 0.4019],  # HC1
    [825.67, 14.5806675, 0.7987],  # HC2
], dtype=float)

# Mole fractions [-], in the same component order.
composition = np.array([
    0.6883688368836883, 0.0914091409140914, 0.046004600460046, 0.0333033303330333,
    0.013901390139013899, 0.0152015201520152, 0.08960896089608959, 0.0222022202220222
], dtype=float)

# Symmetric binary interaction coefficients kij [-].
binaryInteractions = np.array([
    [0.0, -0.0078, 0.009, 0.0056, 0.019, 0.0422, 0.05, 0.09],
    [-0.0078, 0.0, -0.0022, 0.0067, 0.0056, 0.0, 0.04, 0.055],
    [0.009, -0.0022, 0.0, 0.0, 0.023, 0.0, 0.01, 0.01],
    [0.0056, 0.0067, 0.0, 0.0, 0.0204, 0.0, 0.0, 0.0],
    [0.019, 0.0056, 0.023, 0.0204, 0.0, 0.0, 0.0, 0.0],
    [0.0422, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.09, 0.055, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    print('Running Sidky8...', flush=True)
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
