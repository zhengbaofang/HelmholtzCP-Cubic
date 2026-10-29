<div align="center">

<h1>HelmholtzCP Cubic</h1>

<p><strong>基于 Helmholtz 自由能的混合物临界点追踪与计算</strong></p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

<p>
  <a href="#快速开始"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11 及以上"></a>
  <a href="#快速开始"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.0 及以上"></a>
  <a href="#其他-helmholtz-模型"><img src="https://img.shields.io/badge/Included_EOS-PR_%2F_SRK-00897B" alt="内置状态方程：PR 和 SRK"></a>
  <a href="#运行算例"><img src="https://img.shields.io/badge/Example_cases-22-B7791F" alt="22 组算例"></a>
</p>

<p>
  <a href="#基础理论">基础理论</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#运行算例">运行算例</a> &middot;
  <a href="#代表性结果">代表性结果</a> &middot;
  <a href="#其他-helmholtz-模型">其他热力学模型</a> &middot;
  <a href="#文档导航">文档导航</a>
</p>

</div>

## 项目概览

**HelmholtzCP Cubic** 用于搜索固定组成混合物的临界点候选，并评估其热力学稳定性。计算流程包括自适应初值搜索、预测步长控制下的旋节线续接、临界点校正和切平面距离（TPD）筛查。

本方法基于 Helmholtz 自由能密度，通过模型接口也可用于其他可微的 Helmholtz 型状态方程。**当前公开包提供 Peng–Robinson（PR）和 Soave–Redlich–Kwong（SRK）模型**，并附带论文中的 22 组可运行算例。其他模型的接入要求见[其他 Helmholtz 模型](#其他-helmholtz-模型)。

| 阶段 | 主要功能 |
| :--- | :--- |
| **初始化** | 搜索旋节线与指定搜索区域各条边的交点，生成续接起点。 |
| **续接** | 沿旋节线追踪，并通过预测步长控制识别临界点候选。 |
| **校正** | 使用信赖域方法求解临界条件。 |
| **筛查** | 对获得的候选点进行局部四阶分析和 TPD 稳定性筛查。 |

## 基础理论

模型以 Helmholtz 自由能密度 $\Psi=A/V$ 为热力学势，各组分密度为 $\boldsymbol\rho=\rho\mathbf z$，其中 $\rho$ 为总摩尔密度，$\mathbf z$ 为给定的摩尔分数组成。PR/SRK 的 Helmholtz 表达采用 Bell 和 Jäger（2016）的形式，化学势和压力由其导数得到。

在固定温度 $T$ 下，定义 $H_{ij}=\partial^2\Psi/\partial\rho_i\partial\rho_j$ 和 $M=\rho H/(RT)$，其中 $R$ 为气体常数。当 $M$ 具有简单零特征值时，普通临界点候选满足 Heidemann 和 Khalil（1980）及 Bell 和 Jäger（2017）的临界条件：

$$
\begin{aligned}
q^* &= \lambda_{\min}(M)=0, \\
C^* &= \frac{\rho^2}{RT}\left.\frac{\mathrm d^3\Psi(T,\boldsymbol\rho+s\mathbf u)}{\mathrm ds^3}\right|_{s=0}=0,
\end{aligned}
$$

其中 $\mathbf u$ 为 $\lambda_{\min}(M)$ 对应的单位特征向量，$s$ 为沿该方向的标量密度扰动。算法沿旋节线 $q^*=0$ 追踪，并定位 $C^*$ 的零点。局部四阶分析用于表征临界结构（Michelsen，1984），TPD 筛查用于评估热力学稳定性（Michelsen，1982）。热力学关系、数值设置和参考文献见[方法文档](docs/Method.md#2-equations-and-units)。

## 快速开始

需要 **Python 3.11 及以上版本**。NumPy、SciPy 和 PyTorch 已列入项目依赖，`pip` 会按需安装。

```bash
git clone https://github.com/zhengbaofang/HelmholtzCP-Cubic.git
cd HelmholtzCP-Cubic
python -m pip install ".[plot]"
python examples/C1C2_PR.py
```

上述命令运行甲烷–乙烷 PR 算例，打印临界点及其 TPD 状态，并将旋节线图保存至 `output/C1C2_PR.png`。

## 运行算例

**22 组固定组成算例**涵盖二元混合物、酸性气、天然气、凝析气和地层原油，组分数为 **2–12**。每组算例对应一个独立的 Python 脚本，直接列出流体属性、组成、交互参数、搜索设置、单位和来源。

安装后，在仓库根目录执行以下命令：

| 操作 | 命令 |
| :--- | :--- |
| 甲烷–乙烷 PR 算例 | `python examples/C1C2_PR.py` |
| 甲烷–硫化氢 SRK 算例 | `python examples/C1H2S_SRK.py` |
| NWE 组成变体 | `python examples/NWE_O1.py` |
| 可选：批量运行并导出结果 | `python -m examples.RunPaperCases C1C2_PR NaturalGas7` |

打开对应脚本即可编辑输入参数和步长设置，然后直接运行该文件。脚本清楚展示计算流程，打印临界点的温度、压力、摩尔密度和 TPD 状态（单位为 **K、bar 和 mol/L**），并将双面板图保存至 `output/<case>.png`。无需 JSON 输入文件或集中式流体数据加载器。可选批量入口用于导出完整结果，支持 `--all` 和 `--output`，详见[算例说明](docs/Examples.md#optional-batch-export)。

混合物定义和输出字段见[完整算例目录](docs/Examples.md)。如需输入自己的混合物，可参考 [Python API](docs/Method.md#5-python-api)。

## 代表性结果

### 两个相邻的临界点候选

甲烷–硫化氢 SRK 算例展示了预测步长控制及两个相邻候选点的定位，对应算例为 `C1H2S_SRK`。

<p align="center">
  <img src="docs/figures/C1H2S_SRK.png" width="860" alt="甲烷–硫化氢体系的旋节线续接及两个临界点候选的定位">
</p>

### 多组分酸性气

甲烷–二氧化碳–硫化氢 SRK 算例展示了自动初始化、旋节线续接和临界点校正，对应算例为 `SourGas3`。

<p align="center">
  <img src="docs/figures/SourGas3_SRK.png" width="860" alt="三组分酸性气的初始化、旋节线续接和临界点候选">
</p>

以上图片来自配套论文。算例脚本负责数值计算；论文图的排版和图中单独的步长对比由其他绘图与计算流程完成。

### 更多混合物与状态方程

附属图 S1 和 S2 分别在密度–温度和压力–温度坐标中展示同样的九组固定组成计算。子图 **(a)–(b) 采用 SRK**，**(c)–(h) 采用 PR**。子图 **(i) 采用四位点 PC-SAFT**，计算水–二氧化碳–甲烷体系，展示该方法在缔合混合物中的应用。**PC-SAFT 模型实现未包含在当前公开的 PR/SRK 包中。**

**附属图 S1。** 密度–温度坐标下的旋节线。红色圆点表示临界点候选。

<p align="center">
  <a href="docs/figures/FigS1.png"><img src="docs/figures/FigS1.png" width="860" alt="九组密度–温度旋节线：两组 SRK、六组 PR 及一组四位点 PC-SAFT 算例"></a>
</p>

**附属图 S2。** 同一组旋节线在压力–温度坐标中的投影，子图顺序与 S1 一致。红色圆点表示临界点候选，虚线表示负压延伸段。

<p align="center">
  <a href="docs/figures/FigS2.png"><img src="docs/figures/FigS2.png" width="860" alt="九组旋节线的压力–温度投影，其中子图 i 为 PC-SAFT 结果"></a>
</p>

## 其他 Helmholtz 模型

接入其他状态方程时，需要实现 [`FreeEnergyModel`](src/HelmholtzCPCubic/Common/Model.py) 接口，并提供：

1. **可微自由能。** 在 PyTorch 中实现包含理想项和残余项的总 Helmholtz 自由能密度，保留其对温度和各组分密度的可微依赖，以支持最高四阶导数计算。
2. **模型参数与尺度。** 定义组分数、气体常数、特征温度、特征密度及单位换算。
3. **热力学定义域。** 提供适当的状态检查、密度边界，以及与自由能参考态一致的理想气体化学势。

通用的 `backend="ad"` 可据此计算临界条件。其他状态方程的实现和参数数据需要另行提供，当前公开包内置的模型为 PR 和 SRK。

## 文档导航

| 文档 | 内容 |
| :--- | :--- |
| [方法与 Python API](docs/Method.md) | 公式、数值设置、TPD 目标函数、代码结构及 API 用法。 |
| [算例目录](docs/Examples.md) | 全部 22 组算例、输入约定、运行命令及输出字段。 |
| [验证记录](docs/Development.md) | 输入一致性、代表性数值对比及已标记的预期失败测试。 |
| [来源与第三方声明](references/NOTICE.md) | 数据来源及第三方归属说明。 |
| [参考文献](#参考文献) | 临界条件、稳定性及 Helmholtz 形式的基础文献。 |

## 测试与可选导数后端

### 运行已有测试

```bash
python -m pip install ".[test]"
python -m pytest -q
```

默认测试集不运行标记为 `slow` 的完整搜索测试。[验证记录](docs/Development.md#retained-checks-and-known-baseline-failure)说明了保留的预期失败测试，以及已核验的代表性计算。

### 选择导数后端

临界点校正默认采用自动微分。PR/SRK 也支持解析导数和多复数导数，可分别通过 `CriticalPointSolver(..., backend="analytic")` 或 `backend="pymcx"` 选择。多复数后端的可选依赖安装方式为：

```bash
python -m pip install ".[pymcx]"
```

后端选择及适用范围见 [Python API 文档](docs/Method.md#5-python-api)。

## 参考文献

1. Heidemann, R. A., & Khalil, A. M. (1980). The calculation of critical points. *AIChE Journal*, **26**(5), 769–779. [DOI: 10.1002/aic.690260510](https://doi.org/10.1002/aic.690260510).

2. Michelsen, M. L. (1982). The isothermal flash problem. Part I. Stability. *Fluid Phase Equilibria*, **9**(1), 1–19. [DOI: 10.1016/0378-3812(82)85001-2](https://doi.org/10.1016/0378-3812%2882%2985001-2).

3. Michelsen, M. L. (1984). Calculation of critical points and phase boundaries in the critical region. *Fluid Phase Equilibria*, **16**(1), 57–76. [DOI: 10.1016/0378-3812(84)85021-9](https://doi.org/10.1016/0378-3812%2884%2985021-9).

4. Bell, I. H., & Jäger, A. (2016). Helmholtz energy transformations of common cubic equations of state for use with pure fluids and mixtures. *Journal of Research of the National Institute of Standards and Technology*, **121**, 238–263. [DOI: 10.6028/jres.121.011](https://doi.org/10.6028/jres.121.011).

5. Bell, I. H., & Jäger, A. (2017). Calculation of critical points from Helmholtz-energy-explicit mixture models. *Fluid Phase Equilibria*, **433**, 159–173. [DOI: 10.1016/j.fluid.2016.10.030](https://doi.org/10.1016/j.fluid.2016.10.030).
