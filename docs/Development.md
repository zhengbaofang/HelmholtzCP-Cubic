# Validation record

[Home](../README.md) · [中文主页](../README.zh-CN.md) · [Method](Method.md) · [Examples](Examples.md) · **Validation**

![Validation record: 2026-10-28](https://img.shields.io/badge/Validation_record-2026--10--28-526D82)

Recorded validation of the standalone PR/SRK package, including the comparison baseline, numerical results and an unresolved test discrepancy.

**On this page:** [Environment](#environment-and-numerical-configuration) · [Input checks](#input-and-numerical-equivalence) · [Representative results](#representative-calculations) · [Known test issue](#retained-checks-and-known-baseline-failure)

---

## 2026-10-28: standalone PR/SRK validation

The comparison baseline was the paper implementation before standalone packaging. Numerical expressions, search controls and acceptance conditions were preserved. The checks covered all 22 input definitions and two complete representative calculations.

### Environment and numerical configuration

| Item | Recorded value |
| :--- | :--- |
| Operating system | Windows |
| Python | 3.12.7 |
| NumPy | 2.2.5 |
| SciPy | 1.14.1 |
| PyTorch | 2.7.0+cu118 |
| pytest | 9.1.1 |
| Execution | CPU, float64, one Torch thread for the representative runs |

The numerical configuration followed the paper implementation:

| Control | Setting |
| :--- | :--- |
| Intermediate cubic-EOS trial pressure | May cross zero |
| Trust-region parameter `Keasy` | 0.1 |
| Roundoff-sized objective changes | Gradient-progress rule |
| Optional diagonal scaling | Off |

### Input and numerical equivalence

| Check | Scope | Outcome |
| :--- | :--- | :--- |
| Input equivalence | All 22 case definitions | Exact agreement in compositions, pure-component parameters, binary interactions, derived cubic parameters, gas constants and search configurations |
| Source comparison | Numerical modules, with package names normalized | Numerical expressions and executable syntax preserved |
| Standalone packaging | Wheel and source distribution | Built successfully; the extracted wheel was used for the representative runs |

The wheel contained the calculation modules and all **17 JSON files**: 16 fluid input tables and one constants file. External dependencies are declared in [`pyproject.toml`](../pyproject.toml).

### Representative calculations

Complete discovery and screening were rerun for **C1C2_PR** and **C1H2S_SRK**. They returned one and two candidates, respectively.

| Case / candidate | $T$ / K | $P$ / bar | $\rho$ / mol L$^{-1}$ | TPD status |
| :--- | ---: | ---: | ---: | :--- |
| `C1C2_PR` / 1 | 265.7288796195616 | 68.30523456369365 | 8.616194743675429 | `stable` |
| `C1H2S_SRK` / 1 | 260.27083779911146 | 148.99882101202718 | 18.21305989210115 | `stable` |
| `C1H2S_SRK` / 2 | 270.02004364778037 | 146.06699863685148 | 16.884988566543036 | `stable` |

**Acceptance criteria:** `rtol=1e-12` and `atol=1e-12` for numerical comparisons, with identical completion and classification outcomes. These criteria were set before comparison.

Both discovery runs and candidate screenings completed. Candidate coordinates and critical residuals had zero numerical difference from the baseline. TPD values and completion/status fields also matched.

The other 20 full examples were not rerun in this validation. No performance benchmark was conducted. The meaning of `stable` is defined under [TPD screening](Method.md#4-tpd-screening).

### Retained checks and known baseline failure

| Test run | Recorded outcome |
| :--- | :--- |
| Initial selected suite | 54 passed, 1 failed, 1 full-search test deselected |
| Baseline check of the failed test | Same discrepancy reproduced |
| Targeted rerun after applying the expected-failure marker | 1 passed, 1 expected failure |

The unresolved test is `test_adaptive_scanner_reports_exhausted_resolution` in [`test_AdaptiveBoundary.py`](../tests/test_AdaptiveBoundary.py). For the artificial function $\sin(95x)$ on $[-1,1]$ with `maxDepth=1`, it expects an unresolved edge. The scanner instead reports a resolved edge with sampled roots. The same result occurred in the comparison baseline.

The cause remains unresolved. The assertion is retained with `pytest.mark.xfail(strict=True)`, which also flags an unexpected pass for review. No algorithm or tolerance was changed to accommodate this test.

The representative PR/SRK comparisons above assess numerical equivalence. They do not establish complete branch coverage for arbitrary oscillatory functions.

## 2026-10-29: shared gas constant

PR and SRK now use a single algorithm constant, $R=0.083145\ \mathrm{bar\,L\,mol^{-1}\,K^{-1}}$, rounded to six decimal places. The cubic model no longer accepts a gas-constant input, and all 22 examples use this value. The per-case constants table was removed. The previous values, 0.0831, 0.08314472 and 0.0831446261815324, remain relevant only to the historical comparisons above.

On Windows with float64 arithmetic, the [targeted checks](../tests/test_GasConstant.py) constructed all 22 case models and confirmed the common value. At $T=250$ K and $\rho=5$ mol/L, the methane–ethane PR and SRK cases also gave matching Helmholtz-derived and direct cubic pressures, within `rtol=1e-12` and `atol=1e-10` bar. All 24 checks passed. Full critical-point searches and TPD screenings were not rerun with the new constant.

## 2026-10-29: independent example scripts

Each of the 22 cases now defines its fluid properties, composition, interaction matrix and numerical controls in a standalone [example script](Examples.md#case-catalogue). Before disconnecting the former data loader, all 22 new models were compared with that loader using the shared gas constant. Pure-component properties, interaction matrices and normalized compositions matched exactly, including the six NWE blends. Solver, tracing, predictive-step and initialization configurations also matched exactly. The calculation modules were unchanged by this input migration.

All 24 gas-constant and pressure checks passed after changing the test imports. Direct execution of `examples/C1C2_PR.py` completed discovery and TPD screening with one stable candidate: $T=265.72887962$ K, $P=68.30523456$ bar and $\rho=8.61615601$ mol/L. The other 21 cases were checked for input and configuration equivalence without rerunning their searches. The optional batch exporter uses the same case scripts.
