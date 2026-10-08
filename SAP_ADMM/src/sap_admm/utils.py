"""Numerical operators, metrics, diagnostics, and result storage."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse

LABELS = {"sap_admm": "SAP-ADMM", "sap_admm_halpern": r"SAP-ADMM$^{H}$",
          "sap_admm_l1": r"SAP-ADMM$^{1}$", "sdcam_l1": r"SDCAM$^{1}$",
          "sdcam_l2": r"SDCAM$^{2}$", "padmm_l0": "pADMM"}


def vec(x):
    """Column-major vectorization, always float64."""
    return np.asarray(x, dtype=np.float64).reshape(-1, order="F")


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def difference_matrix(n):
    return sparse.diags((-np.ones(n - 1), np.ones(n - 1)), (0, 1),
                        shape=(n - 1, n), format="csr")


def image_operators(rows, cols):
    """D=[D_h;D_v], zero terminal differences, column-major order."""
    def forward(x):
        a = vec(x).reshape(rows, cols, order="F")
        h = np.zeros_like(a)
        v = np.zeros_like(a)
        h[:, :-1] = np.diff(a, axis=1)
        v[:-1, :] = np.diff(a, axis=0)
        return np.concatenate((vec(h), vec(v)))

    def adjoint(y):
        y = vec(y)
        h = y[:rows * cols].reshape(rows, cols, order="F")
        v = y[rows * cols:].reshape(rows, cols, order="F")
        ht, vt = np.zeros_like(h), np.zeros_like(v)
        # Compute the horizontal and vertical adjoints separately, then add.
        if cols > 1:
            ht[:, 0] = -h[:, 0]
            ht[:, 1:-1] = -np.diff(h[:, :-1], axis=1)
            ht[:, -1] = h[:, -2]
        if rows > 1:
            vt[0, :] = -v[0, :]
            vt[1:-1, :] = -np.diff(v[:-1, :], axis=0)
            vt[-1, :] = v[-2, :]
        return vec(ht) + vec(vt)
    return forward, adjoint


def random_y(N, n1, n2, y1, y2, rng):
    blocks, last, total = [], None, 0
    while total < N:
        length = min(int(rng.randint(n1, n2 + 1)), N - total)
        value = int(rng.randint(y1, y2 + 1))
        while last is not None and abs(value - last) <= 2:
            value = int(rng.randint(y1, y2 + 1))
        blocks.append(np.full(length, value, dtype=np.float64))
        last, total = value, total + length
    return np.concatenate(blocks)


def compute_f1_from_support(truth, estimate):
    truth, estimate = np.unique(truth), np.unique(estimate)
    tp = np.intersect1d(truth, estimate).size
    den = truth.size + estimate.size
    # The signal/image convention assigns 0 to two empty support sets.
    return float(2 * tp / den) if den else 0.0


def psnr(img, ref):
    mse = np.mean((np.clip(img, 0, 1) - np.clip(ref, 0, 1)) ** 2)
    return float(10 * np.log10(1 / mse)) if mse > 0 else float("inf")


def compute_ssim_value(img, ref, mode="local"):
    """SSIM with local Gaussian statistics or optional global statistics."""
    a, b = np.clip(np.asarray(img, float), 0, 1), np.clip(np.asarray(ref, float), 0, 1)
    if mode == "global":
        ma, mb = a.mean(), b.mean()
        va, vb = np.var(a), np.var(b)
        cov = np.mean((a - ma) * (b - mb))
    elif mode == "local":
        u, v = np.mgrid[-5:6, -5:6]
        kernel = np.exp(-(u ** 2 + v ** 2) / 4.5)
        kernel /= kernel.sum()
        conv = lambda z: ndimage.convolve(z, kernel, mode="nearest")
        ma, mb = conv(a), conv(b)
        va, vb = conv(a * a) - ma * ma, conv(b * b) - mb * mb
        cov = conv(a * b) - ma * mb
    else:
        raise ValueError("ssim_mode must be local or global")
    return float(np.mean(((2 * ma * mb + 1e-4) * (2 * cov + 9e-4)) /
                         ((ma * ma + mb * mb + 1e-4) * (va + vb + 9e-4))))


def compute_gmsd_value(img, ref, boundary="replicate"):
    hx = np.array([[1, 0, -1]] * 3, dtype=float) / 3
    if boundary not in ("replicate", "zero"):
        raise ValueError("gmsd_boundary must be replicate or zero")
    mode = "nearest" if boundary == "replicate" else "constant"
    def magnitude(z):
        z = np.clip(np.asarray(z, float), 0, 1)
        return np.hypot(ndimage.convolve(z, hx, mode=mode, cval=0),
                        ndimage.convolve(z, hx.T, mode=mode, cval=0))
    a, b = magnitude(img), magnitude(ref)
    T = 170 / 255 ** 2
    quality = (2 * a * b + T) / (a * a + b * b + T)
    return float(np.std(quality, ddof=1))


def sample_std(x, axis=0):
    # Sample standard deviation; a singleton has standard deviation zero.
    x = np.asarray(x)
    return np.std(x, axis=axis, ddof=1 if x.shape[axis] > 1 else 0)


def save_npz(path, **arrays):
    """Atomic write; a plotter never sees a partially written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NuMonitor:
    """Track committed updates and require at least 50 updates after the floor."""
    def __init__(self, initial, floor, minimum=50):
        if not np.isfinite(initial) or not np.isfinite(floor) or initial <= 0 or floor <= 0:
            raise ValueError("nu0 and nu floor must be positive and finite")
        self.final, self.floor = float(initial), float(floor)
        self.minimum = max(50, int(minimum))
        self.first_iteration = 0
        self.count = 0
        self.reached = initial <= floor

    def record(self, value, count):
        self.final, self.count = float(value), int(count)
        if not self.reached and self.final <= self.floor:
            self.first_iteration = int(count)
            self.reached = True

    @property
    def after_floor(self):
        return self.count - self.first_iteration if self.reached else 0

    def can_stop(self):
        return self.reached and self.after_floor >= self.minimum

    def result(self):
        return {"nu_applicable": True, "nu_final": self.final, "nu_floor": self.floor,
                "nu_reached_floor": self.reached,
                "nu_floor_iteration": self.first_iteration,
                "nu_iterations_after_floor": self.after_floor,
                "nu_min_iterations_after_floor": self.minimum,
                "nu_post_floor_satisfied": self.can_stop()}
