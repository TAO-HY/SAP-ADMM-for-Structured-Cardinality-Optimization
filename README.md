# SAP-ADMM: signal and image denoising

This repository uses **SAP-ADMM to solve signal denoising and image denoising problems** from *A Safeguarded Accelerated Proximal ADMM Algorithm for Solving Structured Cardinality Penalized Optimization Problems*, by Wei Bian, Hongyuan Tao, and Fan Wu. The experiments compare capped, convex $\ell_1$, $\ell_{1/2}$ and $\ell_0$ penalty models.

All solvers, metrics and plots run in Python; MATLAB and MATLAB Engine are not required. SDCAM and pADMM are independent implementations of the mathematical algorithms in [1,2], rather than redistributed copies of those authors' software.

## Models and algorithms

Let $\hat b$ denote the noisy observation, $D$ the first-order difference operator, $n$ the number of signal entries or image pixels, and $s$ the number of rows of $D$. Define

```math
\Phi_\nu(y)=\sum_{i=1}^{s}\min\{1,|y_i|/\nu\},\qquad
R_{1/2}(y)=\sum_{i=1}^{s}\sqrt{|y_i|}.
```

The half penalty is a sum of square roots, not the square root of an $\ell_1$ norm. The implemented models are

```math
\begin{aligned}
\text{Capped:}\quad&\min_{x,y}\ \frac{\|x-\hat b\|_1}{n}
+\lambda_p\|Dx-y\|_2+\lambda_0\Phi_\nu(y),\\
\text{Convex:}\quad&\min_{x,y}\ \frac{\|x-\hat b\|_1}{n}
+\lambda_p\|Dx-y\|_2+\lambda_{\ell_1}\|y\|_1,\\
\text{SDCAM, }\ell_1\text{ loss:}\quad&\min_x\ \frac{\|x-\hat b\|_1}{n}
+\lambda_{\ell_{1/2}}R_{1/2}(Dx),\\
\text{SDCAM, }\ell_2\text{ loss:}\quad&\min_x\ \frac12\|x-\hat b\|_2^2
+\lambda_{\ell_{1/2}}R_{1/2}(Dx),\\
\text{Signal pADMM:}\quad&\min_{x,y}\ \frac12\|x-\hat b\|_2^2
+\lambda_{\ell_0}\|y\|_0,\qquad Dx=y.
\end{aligned}
```

The squared $\ell_2$ losses are **not divided by $n$**. The capped and convex models use the **Euclidean mismatch norm**; pADMM imposes $Dx=y$.

| Display name | Result key | Model | Signal | MNIST | Source |
| --- | --- | --- | --- | --- | --- |
| SAP-ADMM | `sap_admm` | $\ell_1$–capped-$\ell_1$ | Yes | Yes | Associated manuscript |
| SAP-ADMM-<sup>H</sup> | `sap_admm_halpern` | Same capped model | Yes | — | SAP-ADMM with $\alpha=2,t=1$, one restart |
| SAP-ADMM-<sup>1</sup> | `sap_admm_l1` | $\ell_1$–$\ell_1$ | Yes | Yes | Convex SAP-ADMM specialization |
| SDCAM<sup>1</sup> | `sdcam_l1` | $\ell_1$–$\ell_{1/2}$ | Yes | Yes | Liu, Pong and Takeda [1] |
| SDCAM<sup>2</sup> | `sdcam_l2` | $\ell_2$–$\ell_{1/2}$ | Yes | Yes | Liu, Pong and Takeda [1] |
| pADMM | `padmm_l0` | $\ell_2$–$\ell_0$ | Yes | — | Boţ and Nguyen [2] |

Each method receives the **same noisy observation** within a trial. SAP-ADMM-<sup>H</sup> is a parameter setting of SAP-ADMM, with one restart after effective update $1000$, before update $1001$; the global continuation counter does not reset.

## Data and operators

**Signals:** $n=1000$, $D_{i,i}=-1$, $D_{i,i+1}=1$ and $D\in\mathbb R^{(n-1)\times n}$. Segment lengths are integer draws from $[50,150]$ and levels from $[-5,10]$, with successive levels differing by more than $2$. Gaussian standard deviation is $0.5$. Independent additive impulse perturbations are uniform on $[-5,5]$. There are $50$ trials for each $\pi\in\{0,0.05,0.10,0.15,0.20\}$.

The generator uses NumPy `RandomState`, seed `trial_index + 1 + (probability_index + 1)*100`. Illustrative recovery inputs use seed `20261001`; capped-model recovery examples use $N_{\max}=2000$, whereas statistical runs use $N_{\max}=500$.

**MNIST:** ten fixed $28\times28$ normalized images, one per digit, with $n=784$ and columnwise vectorization. Each digit has $20$ noisy observations. A probability-$0.10$ mask replaces selected pixels by uniform values on $[0,1]$, then Gaussian noise with standard deviation $0.2$ is added. Noisy observations and images evaluated for quality are clipped to $[0,1]$.

```math
D=\begin{bmatrix}D_h\\D_v\end{bmatrix},\qquad
D_h=L_{28}\otimes I_{28},\qquad D_v=I_{28}\otimes L_{28}.
```

The first $27$ rows of $L_{28}$ implement forward differences and its last row is zero, giving $D$ exactly $2n$ rows without boundary wraparound. Noise seeds are `20260602 + 1000*(digit_index + 1) + trial_index + 1`.

**General matrix:** the additional capped-model experiment solves

```math
\min_{x,y}\ \|Ax-\hat b\|_1/m+\lambda_p\|Dx-y\|_2+\lambda_0\Phi_\nu(y).
```

Here $m=800$, $n=1000$, and a Gaussian matrix is normalized to $\|A\|_2=1$. Gaussian standard deviation is $0.2$ and additive impulse probability is $0.05$. Each of the $20$ instances is reused for $N_{\max}\in\{200,300,400,500,600\}$.

## SAP-ADMM parameters and continuation

The capped-model parameters are listed below.

| Parameter | Signal | General $A$ | MNIST |
| --- | --- | --- | --- |
| $L_f$ | $1/\sqrt n$ | $1/\sqrt m$ | $1/\sqrt n$ |
| $\lambda_p$ | $500L_f$ | $500L_f$ | $2L_f$ |
| $\lambda_0$ | $0.016$ | $0.0018$ | $8\times10^{-5}$ |
| $\rho$ | $2/n$ | $0.6/n$ | $0.5/n$ |
| $\beta$ | $6\rho+10^{-8}$ | $6\rho+10^{-8}$ | $10\rho+10^{-8}$ |
| $(\alpha,t)$ | $(15,1.5)$ | $(15,1.5)$ | $(15,1.5)$ |
| $\nu_0$ | $2$ | $2$ | $0.1$ |
| Stop tolerance | $2\times10^{-4}$ | $2\times10^{-4}$ | $4\times10^{-4}$ |
| Effective-update limit | $20000$ | $20000$ | $20000$ |

```math
\nu_{\min}=\begin{cases}
0.99\lambda_0/(3\lambda_p),&\text{signal, including general }A,\\
0.999\lambda_0/(3\lambda_p),&\text{MNIST},
\end{cases}\qquad
\nu_k=\max\{\gamma_\nu(k)\nu_{k-1},\nu_{\min}\}.
```

With a one-based effective-output counter $k$,

```math
\gamma_\nu(k)=\begin{cases}
0.999,&\text{identity-matrix signal},\ 1\le k\le1500,\\
0.95,&\text{identity-matrix signal},\ k\ge1501,\\
0.9995,&\text{general }A,\ 1\le k\le4000,\\
0.95,&\text{general }A,\ k\ge4001,\\
0.99,&\text{MNIST}.
\end{cases}
```

The first **4000** general-$A$ updates retain $0.9995$; the internal zero-based condition is `bar_count < 4000`. Rejected acceleration trials do not advance the effective counter or commit their proposed $\nu$.

Identity-matrix signal and image initial points are

```math
x^0=\hat b,\quad y^0=0,\quad p^0=x^0,\quad q^0=Dx^0-y^0,
\quad\eta^0=0,\quad\mu^0=0.
```

For general $A$, use $x^0=A^\top\hat b$ and $p^0=Ax^0$. The six-block stopping residual is

```math
\Delta_k=\max_{v\in\{x,y,p,q,\eta,\mu\}}\|\bar v^k-v^k\|_2.
```

Let $k_\nu$ be the first committed output at $\nu_{\min}$. Capped-model tolerance termination requires

```math
\Delta_k<\varepsilon_{\mathrm{stop}},\qquad\nu_k=\nu_{\min},\qquad k-k_\nu\ge50.
```

The first floor output is excluded from those $50$ additional updates. The independent $20000$-update cap remains active. Returned signal variables are proximal outputs $\bar x,\bar y$.

## Comparison algorithm parameters

### SAP-ADMM-<sup>1</sup>: convex penalty

| Parameter | Signal | MNIST |
| --- | --- | --- |
| $\lambda_{\ell_1}$ | $0.02$ | $7\times10^{-4}$ |
| $\lambda_p$ | $500/\sqrt n$ | $2/\sqrt n$ |
| $\rho$ | $2/n$ | $0.5/n$ |
| $\beta$ | $6\rho+10^{-8}$ | $10\rho+10^{-8}$ |
| $(\alpha,t)$ | $(15,1.5)$ | $(15,1.5)$ |
| Stop criterion | $\Delta_k<2\times10^{-4}$ | $\Delta_k<4\times10^{-4}$ |
| Maximum outputs | $20000$ | $20000$ |

Initialization is the same as for the corresponding capped model. The $y$ proximal output uses soft thresholding at $\lambda_{\ell_1}/\beta$. Let $w=(x,y,p,q,\eta,\mu)$ collect the six variable blocks. The accelerated iteration is

```math
\begin{aligned}
\widehat w^{k+1}&=(1-t)w^k+t\bar w^k,\\
w^{k+1}&=w^k+\frac{\alpha}{2(k+\alpha)}(\widehat w^{k+1}-w^k)
+\frac{k}{k+\alpha}(\widehat w^{k+1}-\widehat w^k),
\end{aligned}
```

with $\widehat w^0=w^0$ and $k=0$ initially. This model has no $\nu$, branch safeguard, restart or post-floor wait.

The image stopping check uses all six variable blocks and permits at most $20000$ proximal outputs.

### pADMM: signal $\ell_2$–$\ell_0$ model

This implementation specializes the proximal ADMM scheme of Boţ and Nguyen [2], with

```math
\lambda_{\ell_0}=200,\quad r=100,\quad\rho_{\mathrm{pADMM}}=1.9,
\quad M_2^k=0,\quad M_1^k=(4r+10^{-8})I-rD^\top D.
```

Initialize $x^0=\hat b$, $y^0=0$, $z^0=0$, where $z$ is the unscaled dual variable. With $\beta_0=4r+10^{-8}$,

```math
\begin{aligned}
y^{k+1}&=\mathrm{hard}_{\sqrt{2\lambda_{\ell_0}/r}}(Dx^k+z^k/r),\\
x^{k+1}&=\frac{\hat b+\beta_0x^k-D^\top[z^k+r(Dx^k-y^{k+1})]}{1+\beta_0},\\
z^{k+1}&=z^k+\rho_{\mathrm{pADMM}}r(Dx^{k+1}-y^{k+1}).
\end{aligned}
```

Hard thresholding returns zero at and below the threshold. Terminate when

```math
\max\{\|x^{k+1}-x^k\|_2,\|y^{k+1}-y^k\|_2,\|z^{k+1}-z^k\|_2\}<2\times10^{-4},
```

or after $20000$ iterations. Support is extracted from $y$. pADMM is used in the signal experiment.

### SDCAM<sup>1</sup> and SDCAM<sup>2</sup>

Both use SDCAM and its NPG majorization subsolver from Liu, Pong and Takeda [1], with the parameters listed below.

| Parameter | Signal | MNIST |
| --- | --- | --- |
| $\lambda_{\ell_{1/2}}$, SDCAM<sup>1</sup> | $0.05$ | $8\times10^{-4}$ |
| $\lambda_{\ell_{1/2}}$, SDCAM<sup>2</sup> | $5$ | $0.15$ |
| Initial point / fixed feasible reference | $\hat b$ | $\hat b$ |
| Initial smoothing $\lambda_0^{\mathrm{smooth}}$ | $0.1$ | $1$ |
| Smoothing update | $\lambda_{t+1}=\lambda_t/10$ | Same |
| Outer termination | Next smoothing $<10^{-8}$ | Same |
| NPG memory $M$ | $4$ (five objective values) | Same |
| Initial inverse stepsize per stage | $1$ | Same |
| $L_{\min},L_{\max}$ | $10^{-8},10^8$ | Same |
| Backtracking multiplier $\tau$ | $2$ | Same |
| Descent coefficient $c$ | $10^{-4}$ | Same |
| Initial inner tolerance $\epsilon_0$ | $10^{-5}$ | Same |
| Tolerance update | $\epsilon_{t+1}=\max\{\epsilon_t/1.5,10^{-6}\}$ | Same |
| Relative objective tolerance | $10^{-12}$ | Same |
| Maximum inner updates per stage | $10000$ | Same |

The Moreau smoothing parameter $\lambda_t$ is distinct from penalty coefficients and the capped-model $\nu$. Write $a=\lambda_{\ell_{1/2}}$ and $\mu=\lambda_t$ temporarily:

```math
F_\mu(x)=\ell(x)+\min_y\left\{aR_{1/2}(y)+\frac{\|Dx-y\|_2^2}{2\mu}\right\},
\qquad y_\mu(x)\in\mathrm{prox}_{\mu aR_{1/2}}(Dx).
```

The half-penalty proximal mapping uses a global scalar minimizer per component, selecting zero at the tie $|v|=\frac32(\mu a)^{2/3}$. For inverse stepsize $L$, let $g_\mu(x)=D^\top(Dx-y_\mu(x))/\mu$. Candidate updates are

```math
u=\begin{cases}
\hat b+\mathrm{soft}_{1/(nL)}(x-g_\mu(x)/L-\hat b),&\ell_1\text{ loss},\\
x-[x-\hat b+g_\mu(x)]/L,&\ell_2\text{ loss}.
\end{cases}
```

The inverse Barzilai–Borwein stepsize uses the gradient difference of the convex smooth DC part $h_\mu(x)=\|Dx\|_2^2/(2\mu)$, with $\frac12\|x-\hat b\|_2^2$ added for $\ell_2$ loss. Backtracking accepts only when

```math
F_\mu(u)\le\max_{\max\{0,j-M\}\le i\le j}F_\mu(x^i)-\frac c2\|u-x^j\|_2^2.
```

Let $\bar L_j$ denote the inverse stepsize accepted by the line search at inner update $j$. An inner solve terminates at its cap or when either

```math
\frac{\|x^{j+1}-x^j\|_2}{\max\{1,\|x^{j+1}\|_2\}}<\frac{\epsilon_t}{\bar L_j},
\qquad\text{or}\qquad
\frac{|F_\mu(x^{j+1})-F_\mu(x^j)|}{\max\{1,|F_\mu(x^{j+1})|\}}<10^{-12}.
```

Each stage compares its warm start against $x^{\mathrm{feas}}=\hat b$ **using the current smoothing value for both**, as required by Algorithm 1 of [1]. The next stage starts from the last accepted output or the selected feasible reference. Failed line-search steps are never committed.

The stated schedule solves eight signal stages and nine MNIST stages, including the $10^{-8}$ stage; a roundoff guard preserves this equality. `Iterations` is the sum of accepted inner updates and may exceed $20000$. A $120$-attempt backtracking guard raises an error on failure; it is an implementation guard, not an extra paper parameter. `solver_diagnostics.json` records each stage's smoothing, tolerance, iterations, backtracks, reference reset and stop reason. `inner_cap_reached` identifies capped stages and does not certify convergence.

## Metrics and timing

```math
\mathrm{MSE}=\|b-\bar x\|_2^2/n,\qquad
F_1=\frac{2|S\cap\widehat S|}{|S|+|\widehat S|},\qquad
\mathrm{PSNR}=10\log_{10}\frac n{\|b-\bar x\|_2^2}.
```

Here $b$ is the clean reference, $S$ is its jump support, and $\widehat S$ is the estimated jump support. Both empty supports give $F_1=0$ by the implemented convention. The support threshold is $10^{-6}$: SAP-ADMM variants use $\bar y$, pADMM uses $y$, and SDCAM uses $Dx$. Exact recovery means $F_1=1$. Efficiency is mean $F_1$ divided by mean measured time.

Image quality is evaluated using PSNR, SSIM [3] and GMSD [4], with clipping to $[0,1]$ and replicated boundaries. SSIM uses an $11\times11$ normalized Gaussian window of standard deviation $1.5$:

```math
G_{u,v}=\frac{\exp[-(u^2+v^2)/4.5]}{\sum_{p,q=-5}^{5}\exp[-(p^2+q^2)/4.5]},
\quad\mu_z=G*z,\quad\sigma_z^2=G*(z^2)-\mu_z^2,
\quad\sigma_{b\bar x}=G*(b\bar x)-\mu_b\mu_{\bar x}.
```

```math
\mathrm{SSIM}=\frac1n\sum_i
\frac{(2\mu_{b,i}\mu_{\bar x,i}+10^{-4})(2\sigma_{b\bar x,i}+9\times10^{-4})}
{(\mu_{b,i}^2+\mu_{\bar x,i}^2+10^{-4})(\sigma_{b,i}^2+\sigma_{\bar x,i}^2+9\times10^{-4})}.
```

GMSD uses Prewitt kernels and the **sample** standard deviation:

```math
P_x=\frac13\begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix},\quad P_y=P_x^\top,
\quad m_z=\sqrt{(P_x*z)^2+(P_y*z)^2},\quad T=170/255^2,
```

```math
g_i=\frac{2m_{b,i}m_{\bar x,i}+T}{m_{b,i}^2+m_{\bar x,i}^2+T},\quad
\bar g=\frac1n\sum_i g_i,\quad
\mathrm{GMSD}=\sqrt{\frac1{n-1}\sum_i(g_i-\bar g)^2}.
```

Every method uses the same metric implementation. `Time_s` is solver **elapsed wall time** measured with `time.perf_counter`, including backtracking and rejected trials, excluding data generation, metrics, saving and plotting. It is not process CPU time. Hardware and numerical-library information are recorded in `metadata.json`.

## Run directly in Spyder

Use Python 3.10 or newer. **Extract the entire ZIP**, keeping the outer entry scripts beside the inner `SAP_ADMM/` folder. Open an outer script and press **F5**:

| Entry script | Experiment |
| --- | --- |
| `run_signal.py` | All six signal methods |
| `run_mnist.py` | All four MNIST methods, digits 0–9 |
| `run_general_a.py` | General-$A$, all five safeguard limits |
| `run_all.py` | All three experiments |

The scripts locate source and inputs automatically. No package installation or manual working-directory change is necessary. Restart the Spyder console before running the scripts. If a dependency is missing, run `install_dependencies.py` with F5, then restart the console. Dependencies are NumPy, SciPy, Matplotlib, Pillow and threadpoolctl. Installing them needs internet access; experiment inputs are included.

Editable settings:

```python
QUICK = False
MODE = "all"
OUTPUT_ROOT = None
```

- `QUICK=False` uses the paper trial counts; `True` uses one trial per setting with unchanged dimensions and solver settings.
- `MODE="all"` computes and plots. `"compute"` saves data; `"plot"` redraws saved data.
- `OUTPUT_ROOT=None` saves under `SAP_ADMM/results/`, or `SAP_ADMM/quick_results/` in quick mode. Relative custom paths are resolved inside `SAP_ADMM/`.

SDCAM solves multiple inner problems per input, so full comparisons take longer than SAP-ADMM alone. Recomputing replaces outputs at the chosen location; use a new folder when changing settings.

## Configuration and outputs

Paths below are relative to the inner `SAP_ADMM/` folder.

| File / field | Meaning |
| --- | --- |
| `experiments/config/signal.json` | Signal data, SAP-ADMM, nested `sdcam` and `padmm` parameters |
| `experiments/config/mnist.json` | Image data, nested `admm` and `sdcam` parameters |
| `experiments/config/general_a.json` | General-$A$ data and solver settings |
| `src/sap_admm/general_a.py`: `update_nu_by_bar_count` | General-$A$ continuation factors and the 4000-update boundary |
| `lambda2_capped`, `lambda2_l1` | $\lambda_0$, $\lambda_{\ell_1}$ |
| `sdcam.lambda_half_l1`, `sdcam.lambda_half_l2` | Half-penalty coefficients for the two losses |
| `sdcam.lambda_init`, `sdcam.lambda_min` | Smoothing start and threshold |
| `sdcam.max_inner` | Update cap **per stage** |
| `padmm.lambda_l0`, `padmm.r`, `padmm.relaxation` | $\lambda_{\ell_0}$, $r$, $\rho_{\mathrm{pADMM}}$ |

Each experiment saves `metadata.json`, `results.npz`, `trial_metrics.csv` and `summary.csv`. Signal and MNIST also save `solver_diagnostics.json` and compact `paper_mse_time_table.csv` / `paper_quality_time_table.csv`. Signal illustrative inputs and outputs are in `recovery_example.npz`.

Signal metric arrays have shape `(trials, probabilities, 6)`; MNIST arrays have shape `(trials, digits, 4)`. The saved `methods` array specifies the axis order. Noisy inputs are stored once per trial and setting. CSV files include `Method` and `StopReason`.

For convex SAP-ADMM, SDCAM and pADMM, `nu_applicable=False`. Capped-continuation floats are `NaN` in NPZ and blank in CSV; zero counters and false post-floor flags indicate an inapplicable rule. SDCAM's distinct Moreau continuation is recorded in its diagnostics.

Signal and MNIST figures use Times New Roman with STIXGeneral as the fallback font, bold labels, distinct algorithm colors and line markers. The signal metric figure compares all six algorithms in three panels. The distribution figure shows SAP-ADMM, SDCAM<sup>1</sup>, SDCAM<sup>2</sup> and pADMM in four horizontal panels; bar heights are trial counts, with percentages annotated inside the bars and a $10^{-6}$ category-boundary tolerance. Exact-recovery statistics use $F_1=1$. The signal recovery figure uses a $4\times3$ grid, with one noisy panel and five algorithm panels per noise setting. SAP-ADMM-<sup>H</sup> is included in the statistical comparison. MNIST has six rows (Original, Noisy, SAP-ADMM, SDCAM<sup>1</sup>, SAP-ADMM-<sup>1</sup> and SDCAM<sup>2</sup>), vertical row labels, square image tiles and no digit titles.

Signal and MNIST plots export as 600-dpi PNG and PDF. Their figure basenames are `F1_Time_Efficiency`, `F1_Distribution`, `Figure2_Recovery_Comparison_FiveModels_acc_capl1_python` and `mnist_four_model_restoration_python`. The general-$A$ figure uses `general_A_sensitivity` and exports as 300-dpi PNG and vector PDF. Figures are saved inside each experiment's `figures/` directory.

## Running and verification

From the outer repository folder:

```bash
python -m pip install -r SAP_ADMM/requirements.txt
python run_all.py
```

For module usage and tests:

```bash
cd SAP_ADMM
python -m pip install -e ".[test]"
python -m experiments.run_all --quick --out quick_results
python -m pytest
```

Tests cover scalar half-proximal global minimizers, the Moreau derivative, pADMM against small exact segmentation, continuation boundaries, iteration caps, shared input storage, figures and direct Spyder imports. Numerical fixtures check solver updates on deterministic inputs.

SDCAM stages that reach the inner iteration limit are reported explicitly in the solver diagnostics.

## Citation and licenses

Cite the associated manuscript using `CITATION.bib` or `CITATION.cff`. When reporting comparisons, also cite the relevant algorithm and metric papers below. Citations do not imply that the original authors supplied or endorsed this software. BibTeX entries are included in `REFERENCES.bib`.

1. Tianxiang Liu, Ting Kei Pong and Akiko Takeda. **A successive difference-of-convex approximation method for a class of nonconvex nonsmooth optimization problems.** *Mathematical Programming* **176**, 339–367 (2019). [DOI](https://doi.org/10.1007/s10107-018-1327-8). SDCAM, its NPG majorization subsolver and referenced numerical settings.
2. Radu Ioan Boţ and Dang-Khoa Nguyen. **The proximal alternating direction method of multipliers in the nonconvex setting: convergence analysis and rates.** *Mathematics of Operations Research* **45**(2), 682–712 (2020). [DOI](https://doi.org/10.1287/moor.2019.1008). pADMM.
3. Zhou Wang, Alan C. Bovik, Hamid R. Sheikh and Eero P. Simoncelli. **Image quality assessment: from error visibility to structural similarity.** *IEEE Transactions on Image Processing* **13**(4), 600–612 (2004). [DOI](https://doi.org/10.1109/TIP.2003.819861). SSIM.
4. Wufeng Xue, Lei Zhang, Xuanqin Mou and Alan C. Bovik. **Gradient magnitude similarity deviation: a highly efficient perceptual image quality index.** *IEEE Transactions on Image Processing* **23**(2), 684–695 (2014). [DOI](https://doi.org/10.1109/TIP.2013.2293423). GMSD.
5. Yann LeCun, Corinna Cortes and Christopher J. C. Burges. **The MNIST Database of Handwritten Digits.** [Dataset homepage](https://yann.lecun.com/exdb/mnist/).

Python source is distributed under the **MIT License** in `LICENSE`. Dependencies retain their own licenses and are not vendored. Original SDCAM/pADMM author software, publisher PDFs and publisher figures are not bundled.

The MNIST extract and image derivatives are **outside the MIT source-code license**. They retain attribution to Yann LeCun and Corinna Cortes and the dataset's [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/), as described by the [Keras MNIST documentation](https://keras.io/api/datasets/mnist/). Selected images were normalized to floating-point values and repackaged into NPZ. Upstream sample indices were not supplied; the array hash identifies the exact inputs. Preserve attribution and the dataset license when redistributing the data or image derivatives. See `THIRD_PARTY_NOTICES.md` and `experiments/inputs/README.md`.
