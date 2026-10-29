# Sources and third-party notices

This package implements the fixed-composition critical-point discovery and stability workflows described in the accompanying paper. It includes PR/SRK thermodynamic models, numerical solvers and 22 example cases.

Literature citations accompany the relevant source modules. Core thermodynamic references are listed in the [README](../README.md#references), with additional numerical references in [Method and API](../docs/Method.md#6-references).

Some basic pure-component properties were transcribed from CoolProp 8.0.0, as identified in the individual example scripts. The corresponding MIT license is retained in `CoolProp-LICENSE.txt`. CoolProp is not a runtime dependency. The optional multicomplex backend uses the separately installed `multicomplex` package and does not bundle its source or binaries.

Each example script specifies its composition, interaction parameters, units and parameter sources. The gas constant and EOS formulas are defined in the algorithm package.

The CoolProp license applies to the corresponding third-party material only. A public license for the authors' original source has not yet been selected for this prepared release.
