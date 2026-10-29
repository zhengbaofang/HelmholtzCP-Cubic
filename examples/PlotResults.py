"""
Plot fixed-composition spinodals and screened critical-point candidates.
The two panels use density–temperature and temperature–pressure coordinates.
Curves retain their tracing order. Pressure uses the cubic EOS at saved states.
The function uses existing TPD assessments and writes one PNG without a window.
"""

from pathlib import Path

import numpy as np


def plotResults(model, composition, discovery, assessments, outputPath):
    """
    Save spinodals and candidate markers in a two-panel PNG and return its path.
    model is a PR/SRK model and composition is the normalized mole-fraction vector.
    discovery contains the saved traces and corrected candidates in K/bar/mol/L.
    assessments must follow discovery.roots in the same order, one per candidate.
    outputPath is the PNG destination. Its parent directory is created as needed.
    Mismatched assessments or a non-PNG destination raise ValueError.
    Matplotlib is required only when this plotting function is called.
    """

    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(assessments) != len(discovery.roots):
        raise ValueError("provide one TPD assessment per critical-point candidate")
    destination = Path(outputPath)
    if destination.suffix.lower() != ".png":
        raise ValueError("outputPath must end in .png")
    fractions = np.asarray(composition, dtype=float)
    markerStyles = {
        "stable": ("#009E73", "o", "TPD stable"),
        "unstable": ("#D55E00", "^", "TPD unstable"),
        "failed": ("#777777", "x", "Screening failed"),
    }

    # Keep formatting local to this figure and consistent across both panels.
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.linewidth": 0.5,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.grid": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8.0,
    }
    with mpl.rc_context(style):
        figure = Figure(figsize=(180.0 / 25.4, 78.0 / 25.4), dpi=300)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.82, wspace=0.36)
        legendEntries = {}

        # Draw each trace separately without sorting or joining distinct paths.
        for trace in discovery.contours:
            if not trace.contour:
                continue
            temperatures = [point.T for point in trace.contour]
            densities = [point.rho for point in trace.contour]
            pressures = [float(model.directPres(point.T, point.rho * fractions))
                         for point in trace.contour]
            line, = axes[0].plot(densities, temperatures, color="#0072B2", linewidth=1.1)
            axes[1].plot(temperatures, pressures, color="#0072B2", linewidth=1.1)
            legendEntries.setdefault("Spinodal", line)

        # Reuse accepted candidate pressures and the completed TPD classifications.
        for point, assessment in zip(discovery.roots, assessments, strict=True):
            color, marker, label = markerStyles[assessment.status.value]
            artist = axes[0].scatter(point.rho, point.T, color=color, marker=marker,
                                     s=30, linewidths=0.8, zorder=3)
            axes[1].scatter(point.T, point.P, color=color, marker=marker,
                            s=30, linewidths=0.8, zorder=3)
            legendEntries.setdefault(label, artist)

        axes[0].set_xlabel(r"$\rho$ (mol L$^{-1}$)")
        axes[0].set_ylabel(r"$T$ (K)")
        axes[1].set_xlabel(r"$T$ (K)")
        axes[1].set_ylabel(r"$P$ (bar)")
        for letter, axis in zip(("a", "b"), axes):
            axis.tick_params(direction="in", length=3.0)
            axis.margins(0.05)
            axis.text(-0.18, 1.03, f"({letter})", transform=axis.transAxes,
                      fontsize=9.0, fontweight="bold", va="bottom")
        if legendEntries:
            figure.legend(list(legendEntries.values()), list(legendEntries),
                          loc="upper center", bbox_to_anchor=(0.54, 0.99),
                          ncol=len(legendEntries))
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, facecolor="white")
    return destination
