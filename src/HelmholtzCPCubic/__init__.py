"""
Independent implementation of the paper critical-point workflows.
Public state values use K, bar and mol/L; EOS internals retain declared units.
"""

from .Application.AdaptiveDiscovery import AdaptiveBoundaryBellFinder
from .Application.CriticalPoint import CriticalPointSolver
from .Application.Discovery import BellCriticalFinder
from .Application.PredictiveDiscovery import PredictiveBellCriticalFinder
from .Transform.CubicBell import CubicBellModel
__version__ = "0.1.0"
