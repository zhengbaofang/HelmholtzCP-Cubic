"""
Shared active-composition validation and model scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .Model import FreeEnergyModel


def normalizeComposition(
    composition: Sequence[float] | np.ndarray,
    componentCount: int,
) -> np.ndarray:
    """
    Return a finite, positive mole-fraction vector of the requested size.
    """

    fractions = np.asarray(composition, dtype=float)
    if fractions.shape != (componentCount,):
        raise ValueError(f"composition must have shape ({componentCount},)")
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("all active-component mole fractions must be finite and positive")
    return fractions / np.sum(fractions)


def normalizeFullComposition(
    composition: Sequence[float] | np.ndarray,
    componentCount: int,
) -> np.ndarray:
    """
    Return a normalized full composition that may contain exact zeros.

    The full vector is used only by application-level active-component
    reduction. Thermodynamic kernels continue to receive strictly positive
    compositions whose dimension equals the active EOS submodel.
    """

    fractions = np.asarray(composition, dtype=float)
    if fractions.shape != (componentCount,):
        raise ValueError(f"composition must have shape ({componentCount},)")
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("component mole fractions must be finite and nonnegative")
    total = float(np.sum(fractions))
    if total <= 0.0:
        raise ValueError("composition must contain at least one positive component")
    return fractions / total


def compositionTemperatureScale(
    model: FreeEnergyModel,
    composition: np.ndarray,
) -> float:
    """
    Return the model-provided composition scale or its scalar fallback.
    """

    method = getattr(model, "compositionTempScale", None)
    if callable(method):
        return float(method(composition))
    return float(model.tempScale)


def compositionDensityScale(
    model: FreeEnergyModel,
    composition: np.ndarray,
) -> float:
    """
    Return the model-provided composition density scale or its fallback.
    """

    method = getattr(model, "compositionDensityScale", None)
    scale = (
        float(method(composition))
        if callable(method)
        else float(model.densityScale)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("composition density scale must be finite and positive")
    return scale
