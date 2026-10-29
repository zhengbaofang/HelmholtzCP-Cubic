# Example cases

[Home](../README.md) · [中文主页](../README.zh-CN.md) · [Method](Method.md) · **Examples** · [Validation](Development.md)

![Examples: 22 fixed-composition cases](https://img.shields.io/badge/Examples-22_fixed--composition_cases-00897B)

One Python script per case. Open the file, inspect or edit its inputs, and run it directly.

**On this page:** [Catalogue](#case-catalogue) · [Run examples](#run-examples) · [Inputs and settings](#inputs-and-settings) · [Results](#results-and-units) · [Plotting](#plotting) · [Batch export](#optional-batch-export)

---

## Case catalogue

The 22 scripts comprise 16 base cases and six NWE composition variants. Every script contains its fluid inputs and numerical settings. The algorithm package supplies the gas constant and EOS formulas.

### Base cases

| Case name | EOS | Components | Mixture |
| :--- | :---: | ---: | :--- |
| [C1C2_SRK.py](../examples/C1C2_SRK.py) | SRK | 2 | Methane–ethane |
| [C1C2_PR.py](../examples/C1C2_PR.py) | PR | 2 | Methane–ethane |
| [C1H2S_SRK.py](../examples/C1H2S_SRK.py) | SRK | 2 | Methane–hydrogen sulfide |
| [D2014.py](../examples/D2014.py) | SRK | 3 | Dimitrakopoulos case 10 |
| [Mix2.py](../examples/Mix2.py) | SRK | 6 | Mix-2 |
| [Sidky8.py](../examples/Sidky8.py) | SRK | 8 | Sidky eight-component case |
| [SourGas3.py](../examples/SourGas3.py) | SRK | 3 | Methane/CO₂/H₂S sour gas |
| [NaturalGas6.py](../examples/NaturalGas6.py) | PR | 6 | Six-component synthetic natural gas |
| [NaturalGas7.py](../examples/NaturalGas7.py) | SRK | 7 | Seven-component natural gas |
| [SPE3.py](../examples/SPE3.py) | PR | 9 | SPE3 gas condensate |
| [LF12.py](../examples/LF12.py) | PR | 12 | LF12 Lille Frigg condensate |
| [S10.py](../examples/S10.py) | PR | 10 | S10 Stenby reservoir oil |
| [SPE5.py](../examples/SPE5.py) | PR | 6 | SPE5 reservoir oil |
| [NWE.py](../examples/NWE.py) | PR | 7 | NWE0.7 CO₂-injected oil |
| [OilG04.py](../examples/OilG04.py) | PR | 7 | OilG0.4 oil/gas fluid |
| [Glaso095.py](../examples/Glaso095.py) | PR | 11 | Glaso0.95 |

### NWE composition variants

Each of these six seven-component PR scripts contains the NWE properties, oil and injection-gas compositions, and its prescribed mixing fraction. The blend is calculated explicitly in that script.

| Case name | Mixing fraction |
| :--- | ---: |
| [NWE_O1.py](../examples/NWE_O1.py) | 0.25671627785900253 |
| [NWE_O2.py](../examples/NWE_O2.py) | 0.57161235117687514 |
| [NWE_O3.py](../examples/NWE_O3.py) | 0.71135187281560019 |
| [NWE_G1.py](../examples/NWE_G1.py) | 0.99306655645985731 |
| [NWE_G2.py](../examples/NWE_G2.py) | 0.99682912243412747 |
| [NWE_G3.py](../examples/NWE_G3.py) | 0.99932864424750179 |

## Run examples

Install the package with its plotting extra, then run a case from the repository root:

```bash
python -m pip install ".[plot]"
python examples/C1C2_PR.py
python examples/C1H2S_SRK.py
python examples/NWE_O1.py
```

Each script shows the same short calculation sequence: construct the EOS, discover critical points, screen each candidate with TPD, print the results, and save the figure. Importing a script does not start a calculation.

## Inputs and settings

The inputs appear above `makeCase()` and `main()` in each script:

| Name | Meaning |
| :--- | :--- |
| `eos` | `"PR"` or `"SRK"` |
| `componentNames` | Component order used throughout the script |
| `fluidProperties` | One row per component: critical temperature [K], critical pressure [bar], acentric factor [-] |
| `composition` | Mole fractions, normalized when constructing the case |
| `binaryInteractions` | Symmetric dimensionless interaction matrix in the same component order |
| `solverConfig` | Search bounds and critical-point tolerances |
| `stepConfig` | Conservative and nominal continuation step caps, 0.04 and 0.12 |
| `traceConfig` | Initial and minimum continuation steps, 0.03 and 0.0025 |

Step sizes are dimensionless in logarithmic temperature/density coordinates. `temperatureBoundsK=None` and `densityBoundsMolL=None` retain the original composition-dependent search ranges. The maximum tracing step follows `stepConfig.nominalMaximumStep`.

For the NWE variants, edit `mixingFraction` to blend the explicitly listed `oilComposition` and `gasComposition`. Parameter sources are recorded in each script's opening comment. `NaturalGas6` is a synthetic regression input rather than an independent literature dataset.

To define another mixture, copy a script and replace its properties, composition and interaction matrix. Keep their component order consistent. The gas constant, cubic coefficients and mixing formulas are defined in the algorithm package and are not case inputs.

## Results and units

A directly executed script prints candidate temperatures [K], pressures [bar], molar densities [mol/L], TPD status and the lowest TPD found. It also reports the candidate count, completion status and any returned failure messages. An incomplete search or failed screening gives a nonzero exit status. Direct execution also saves `output/<case>.png`; repeating the case updates that image.

For programmatic access, `makeCase()` returns the model, normalized composition and solver configuration. The explicit `main()` body shows how to obtain the discovery and screening objects.

## Plotting

The shared [plotting function](../examples/PlotResults.py) draws density–temperature and temperature–pressure panels. Each trace retains its original point order and is drawn separately. Green circles indicate `stable` candidates, orange triangles indicate `unstable` candidates, and grey crosses indicate failed screening. Only categories present in the result appear in the legend.

The function reuses the traced temperatures, densities and candidate assessments. It evaluates pressure at the saved curve states using the cubic EOS, without repeating critical-point discovery or TPD screening. Figures have a white background, 0.5 pt axes and ticks, and a size of 180 × 78 mm at 300 dpi. They are saved as PNG without opening a window.

The last part of each case calls `plotResults(model, fractions, discovery, assessments, outputPath)`. The assessments follow `discovery.roots` in the same order. To run a case without plotting or requiring Matplotlib:

```python
from examples.C1C2_PR import main

main(saveFigure=False)
```

## Optional batch export

The batch runner reads the same case scripts. It can save the full spinodal, candidate and screening records when needed:

```bash
python -m examples.RunPaperCases --list
python -m examples.RunPaperCases C1C2_PR NaturalGas7
python -m examples.RunPaperCases --all --output output/AllCases
```

It processes cases sequentially with one Torch thread and writes one result JSON per case. These are outputs, not fluid input files. Existing outputs are preserved; use a different `--output` directory to repeat a batch.

| Output field | Contents |
| :--- | :--- |
| `discovery` | Starting points, contours, candidates and discovery completion information |
| `screening` | Local fourth-order and TPD assessments |
| `completed` | Overall completion status |
| `discoverySeconds` | Model preparation and discovery time, excluding writing |
| `screeningSeconds` | Candidate classification time, excluding writing |
