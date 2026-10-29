# Method and API

[Home](../README.md) · [中文主页](../README.zh-CN.md) · **Method** · [Examples](Examples.md) · [Validation](Development.md)

![Formulation: Helmholtz free energy](https://img.shields.io/badge/Formulation-Helmholtz_free_energy-005F73)

The thermodynamic formulation, numerical controls and Python interface for fixed-composition critical-point calculations.

**On this page:** [1. Workflow](#1-workflow-and-architecture) · [2. Equations](#2-equations-and-units) · [3. Settings](#3-numerical-settings) · [4. TPD](#4-tpd-screening) · [5. API](#5-python-api) · [6. References](#6-references)

---

## 1. Workflow and architecture

The workflow follows the Helmholtz criticality formulation of Bell and Jäger (2017), with adaptive initialization and predictive step control used in the accompanying paper:

1. Construct a PR or SRK Helmholtz model from the input parameters.
2. Locate spinodal intersections with the edges of the search region.
3. Trace the spinodals and correct critical-point candidates.
4. Assess each candidate using local fourth-order analysis and TPD.

### 1.1 Source structure

| Layer | Responsibility |
| :--- | :--- |
| `Common` | Composition and free-energy model contracts |
| `Transform/CubicBell.py` | PR/SRK Helmholtz free energy and model parameters |
| `Kernel` | Thermodynamic derivatives, criticality and local stability |
| `Numerics` | Bracketed roots, trust-region solvers and continuation correctors |
| `Equilibrium` | Density roots and TPD minimization |
| `Application` | Initialization, spinodal discovery, candidate correction and classification |
| `examples/<case>.py` | Fluid inputs, composition, search controls and a runnable calculation |
| `examples/RunPaperCases.py` | Optional batch execution and result writing |

The calculation modules operate on model parameters and compositions. The example runner handles input selection and result writing.

## 2. Equations and units

**Public units:** temperature in K, pressure in bar, molar density in mol/L and dimensionless mole fractions.

The Helmholtz free-energy density is $\Psi(T,\boldsymbol\rho)$. Chemical potentials, pressure and the component-density Hessian follow Bell and Jäger (2017):

$$
\begin{aligned}
\mu_i &= \frac{\partial\Psi}{\partial\rho_i}, \\
P &= \boldsymbol\rho^{\mathsf T}\boldsymbol\mu-\Psi, \\
H_{ij} &= \frac{\partial^2\Psi}{\partial\rho_i\partial\rho_j}.
\end{aligned}
$$

For an ordinary critical point, the implementation solves its dimensionless minimum-Hessian eigenvalue and third-directional-derivative conditions, $q^*=0$ and $C^*=0$, in $(\ln T,\ln\rho)$, following Heidemann and Khalil (1980) and Bell and Jäger (2017). Local fourth-order classification follows Michelsen (1984).

### 2.1 Cubic EOS constants

| Model | $\Omega_a$ | $\Omega_b$ |
| :--- | ---: | ---: |
| PR | 0.457235529 | 0.077796074 |
| SRK | 0.4274802 | 0.08664035 |

The SRK coefficients follow the shared Code_Po convention as of 2026-10-29.
This coefficient alignment received static checks only, as requested; existing
numerical results retain their original conventions and have not been recalculated.

The cubic model defines the temperature correlations and the quadratic attraction and linear covolume mixing rules. PR, SRK and all 22 examples use $R=0.083145\ \mathrm{bar\,L\,mol^{-1}\,K^{-1}}$, defined once as `R_GAS` in the cubic model. Individual example scripts supply fluid properties, composition and numerical controls.

## 3. Numerical settings

| Operation | Default setting |
| :--- | :--- |
| Temperature search interval | $[0.3T_s,5T_s]$, $T_s=\sum_i z_iT_{c,i}$ |
| Density search interval | $[0.005\rho_u,0.95\rho_u]$, using the model's density upper bound $\rho_u$ |
| Critical residual tolerance | $10^{-8}$ |
| Critical correction step tolerance | $10^{-10}$ |
| Adaptive edge root residual / coordinate tolerances | $10^{-8}$ / $10^{-10}$ |
| Adaptive edge subdivision depth | 9 |
| Predictive conservative / nominal step caps | 0.04 / 0.12 |
| TPD projected composition-gradient tolerance | $10^{-8}$ |
| Independent pressure qualification | $10^{-8}$ bar |
| Instability criterion | Accepted original TPD $<-10^{-10}$ |
| Optional scalar trust-region diagonal scaling | Off |

The model's domain checks and search-edge inset also apply. Search intervals specify the numerical search region, not an EOS validity range. Full controls are exposed through `CriticalSolverConfig`, `AdaptiveBoundaryConfig`, `CriticalDiscoveryConfig`, `LinearForecastStepConfig` and `ScalarTrustRegionConfig`.

### 3.1 Solver responsibilities

| Numerical task | Solver |
| :--- | :--- |
| Critical and spinodal correction | Bounded least-squares trust-region solver |
| TPD minimization | Scalar Moré–Sorensen trust-region solver (Moré and Sorensen, 1983) |
| One-dimensional bracketed roots | Chandrupatla method (Chandrupatla, 1997) |

An unbracketed scalar basin may be located with bounded scalar minimization, followed by a check against the original residual.

## 4. TPD screening

### 4.1 Objective and TPD relation

At fixed reference temperature and pressure $P_0$, the classifier jointly optimizes trial composition $\mathbf y$ and total density $r$. Let $b_i=\ln(P_0/(RT))+\ln z_i+\ln\hat\phi_i^0$. The extended molar objective is

$$
J=\frac{\Psi(T,r\mathbf y)+P_0}{RT\,r}-\mathbf y^{\mathsf T}\mathbf b.
$$

Its relationship to the usual dimensionless TPD follows from the Helmholtz pressure relation:

$$
\mathrm{TPD}=\mathbf y^{\mathsf T}\left(\frac{\boldsymbol\mu}{RT}-\mathbf b\right)=J+\frac{P-P_0}{RT\,r}.
$$

The reference criterion is Michelsen's (1982) tangent-plane test. The use of density/volume variables follows the approach discussed by Nichita (2018), while the above normalization retains the paper's original TPD threshold.

### 4.2 Search and acceptance

Each initial composition supplies all qualified mechanically stable density roots to the same local solver. Iteration uses ILR composition coordinates and logarithmic total density without a nested density search at each step.

Intermediate PR/SRK trials may cross zero pressure or mechanical slope. Final acceptance requires the target pressure, the projected-gradient tolerance and mechanical/compositional curvature conditions. The negative threshold is applied to the pressure-qualified TPD.

The classifier returns `stable`, `unstable` or `failed`. `stable` denotes completion of the prescribed search without an accepted negative TPD. Boundary-root discovery and finite multistart screening do not guarantee global branch coverage or global minimization. Failure and incomplete-search information remain in the returned results.

## 5. Python API

### 5.1 Run a packaged case

```python
from HelmholtzCPCubic import AdaptiveBoundaryBellFinder, CriticalPointSolver
from HelmholtzCPCubic.Application import TPDStabilityClassifier
from examples.C1H2S_SRK import makeCase, traceConfig, stepConfig

model, composition, config = makeCase()
solver = CriticalPointSolver(model, config)
finder = AdaptiveBoundaryBellFinder(solver, traceConfig, stepConfig=stepConfig)
discovery = finder.findBoundaryContours(composition)
classifier = TPDStabilityClassifier(model)
for point in discovery.roots:
    assessment = classifier.evaluate(point)
    print(point.T, point.P, point.rho, assessment.status.value)
print("Search completed:", discovery.completed)
```

### 5.2 Supply a mixture

Construct `CubicBellModel` with critical temperatures in K, critical pressures in bar, acentric factors, a dimensionless interaction matrix and `mode="PR"` or `mode="SRK"`. Pass its mole fractions to the same discovery entry.

### 5.3 Select a derivative backend

| `backend` | Local critical-condition derivatives |
| :--- | :--- |
| `"ad"` | Automatic differentiation, the default |
| `"analytic"` | Analytic derivatives for supported PR/SRK models |
| `"pymcx"` | Optional multicomplex derivatives for supported PR/SRK models |
| `"auto"` | Analytic derivatives when supported, otherwise automatic differentiation |

Set the option on `CriticalPointSolver`, for example `CriticalPointSolver(model, config, backend="analytic")`. Backend selection leaves the search strategy, TPD criterion and default tolerances unchanged.

## 6. References

1. Heidemann, R. A., & Khalil, A. M. (1980). The calculation of critical points. *AIChE Journal*, **26**(5), 769–779. [DOI: 10.1002/aic.690260510](https://doi.org/10.1002/aic.690260510).

2. Michelsen, M. L. (1982). The isothermal flash problem. Part I. Stability. *Fluid Phase Equilibria*, **9**(1), 1–19. [DOI: 10.1016/0378-3812(82)85001-2](https://doi.org/10.1016/0378-3812%2882%2985001-2).

3. Michelsen, M. L. (1984). Calculation of critical points and phase boundaries in the critical region. *Fluid Phase Equilibria*, **16**(1), 57–76. [DOI: 10.1016/0378-3812(84)85021-9](https://doi.org/10.1016/0378-3812%2884%2985021-9).

4. Bell, I. H., & Jäger, A. (2017). Calculation of critical points from Helmholtz-energy-explicit mixture models. *Fluid Phase Equilibria*, **433**, 159–173. [DOI: 10.1016/j.fluid.2016.10.030](https://doi.org/10.1016/j.fluid.2016.10.030).

5. Moré, J. J., & Sorensen, D. C. (1983). Computing a trust region step. *SIAM Journal on Scientific and Statistical Computing*, **4**(3), 553–572. [DOI: 10.1137/0904038](https://doi.org/10.1137/0904038).

6. Chandrupatla, T. R. (1997). A new hybrid quadratic/bisection algorithm for finding the zero of a nonlinear function without using derivatives. *Advances in Engineering Software*, **28**(3), 145–149. [DOI: 10.1016/S0965-9978(96)00051-8](https://doi.org/10.1016/S0965-9978%2896%2900051-8).

7. Nichita, D. V. (2018). Density-based phase envelope construction. *Fluid Phase Equilibria*, **478**, 100–113. [DOI: 10.1016/j.fluid.2018.09.007](https://doi.org/10.1016/j.fluid.2018.09.007).

Individual [example scripts](Examples.md#case-catalogue) specify their parameter sources and units. See [Sources and notices](../references/NOTICE.md) for third-party attribution.
