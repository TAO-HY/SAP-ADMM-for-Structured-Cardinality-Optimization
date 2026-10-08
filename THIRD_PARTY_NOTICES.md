# Third-party notices

## Python dependencies

NumPy, SciPy, Matplotlib, Pillow, and threadpoolctl are runtime dependencies. They are not copied into this repository and remain subject to their own licenses.

## MNIST sample

`experiments/inputs/mnist_clean_images.npz` contains ten clean 28-by-28 images selected from the MNIST database for reproducibility. Dataset copyright holders: Yann LeCun and Corinna Cortes. The data and image derivatives are outside the MIT source-code license and retain the dataset's Creative Commons Attribution-ShareAlike 3.0 license:

- Dataset: https://yann.lecun.com/exdb/mnist/
- License: https://creativecommons.org/licenses/by-sa/3.0/
- License documentation: https://keras.io/api/datasets/mnist/

Changes: selection of ten images, normalization to float64 in [0,1], and NPZ repackaging. The upstream sample indices were not provided. Noise generation and restoration produce image derivatives. Keep attribution and the CC BY-SA 3.0 license with redistributed input/noisy/restored image arrays or plots. Cite:

Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, *The MNIST Database of Handwritten Digits*.

## Comparison algorithms

`src/sap_admm/comparisons.py` independently implements the mathematical SDCAM and pADMM schemes; no original author implementation or repository was copied or vendored. Cite:

1. T. Liu, T. K. Pong and A. Takeda, *A successive difference-of-convex approximation method for a class of nonconvex nonsmooth optimization problems*, Mathematical Programming 176, 339–367 (2019), https://doi.org/10.1007/s10107-018-1327-8.
2. R. I. Boţ and D.-K. Nguyen, *The proximal alternating direction method of multipliers in the nonconvex setting: convergence analysis and rates*, Mathematics of Operations Research 45(2), 682–712 (2020), https://doi.org/10.1287/moor.2019.1008.

Experimental coefficients are listed in README. The independent implementation is MIT-licensed research software; attribution to an algorithm publication does not grant a license to its publisher PDF, figures or third-party software. Those materials are not distributed here. The original authors have not endorsed this implementation.

SSIM and GMSD use the formulas described in README; the publications of Wang et al. (2004) and Xue et al. (2014) are cited in `REFERENCES.bib` and README. Their original software is not bundled.

Numeric regression fixtures contain reference outputs for deterministic test inputs. The companion JSON files record their parameters, seeds and source hashes.

## Source-code license scope

The MIT license applies to this repository's source code. It does not relicense MNIST images, their derivatives, dependencies, or the referenced publications. The release contains no publisher PDFs or manuscript figure reproductions. Generated signal data and numerical statistics are research outputs; MNIST image arrays retain the separate dataset license above.
