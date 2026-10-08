"""Independent Python implementations of the manuscript's comparison models.

SDCAM: Liu, Pong and Takeda, Mathematical Programming 176 (2019), 339--367,
doi:10.1007/s10107-018-1327-8, Algorithms 1--2 and Section 5.1.
pADMM: Bot and Nguyen, Mathematics of Operations Research 45 (2020), 682--712,
doi:10.1287/moor.2019.1008, proximal ADMM scheme.
The experimental coefficients and stopping rules follow ex_supplement.tex.
These routines are not copies of the cited authors' software.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import numpy as np

from .utils import soft_threshold, vec


def half_threshold(v, weight):
    """A global minimizer of .5*(u-v)**2 + weight*sqrt(abs(u)), entrywise.

    The threshold comes from comparing the positive cubic root's objective
    with the objective at zero. At a tie the zero minimizer is selected.
    """
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    v = np.asarray(v, dtype=float)
    if weight == 0:
        return v.copy()
    magnitude = np.abs(v)
    out = np.zeros_like(v)
    active = magnitude > 1.5 * weight ** (2 / 3)
    a = magnitude[active]
    if a.size:
        angle = np.arccos(np.clip(weight / 4 * (3 / a) ** 1.5, -1, 1))
        root = 2 * a / 3 * (1 + np.cos(2 * np.pi / 3 - 2 * angle / 3))
        out[active] = np.sign(v[active]) * root
    return out


def moreau_objective(x, observation, D, penalty, smoothing, loss):
    """Return F_mu, its loss, Dx and the half-penalty proximal minimizer."""
    dx = D(x)
    y = half_threshold(dx, smoothing * penalty)
    residual = x - observation
    fidelity = np.abs(residual).sum() / observation.size if loss == "l1" else 0.5 * np.dot(residual, residual)
    value = fidelity + np.dot(dx-y, dx-y) / (2*smoothing) + penalty * np.sqrt(np.abs(y)).sum()
    return float(value), float(fidelity), dx, y


def sdcam(observation, operators, parameters=None, *, loss="l1"):
    """SDCAM with NPG majorization for L1/n or unnormalized squared L2 loss.

    Return (x, Dx, info). Smoothing mu is distinct from the capped-model nu.
    The feasible reference and all warm-start comparisons use the SAME mu.
    """
    prm = parameters or {}
    b = vec(observation)
    if not b.size or not np.all(np.isfinite(b)) or loss not in {"l1", "l2"}:
        raise ValueError("Require finite nonempty input and loss 'l1' or 'l2'")
    D, DT = operators
    penalty = float(prm.get("lambda_half", 0.05 if loss == "l1" else 5.0))
    mu = float(prm.get("lambda_init", 0.1))
    mu_min = float(prm.get("lambda_min", 1e-8))
    reduction = float(prm.get("lambda_reduction", 0.1))
    eps = float(prm.get("epsilon_init", 1e-5))
    eps_min = float(prm.get("epsilon_min", 1e-6))
    eps_divisor = float(prm.get("epsilon_divisor", 1.5))
    max_inner = int(prm.get("max_inner", 10000))
    memory = int(prm.get("memory", 4))
    Lmin, Lmax = float(prm.get("Lmin", 1e-8)), float(prm.get("Lmax", 1e8))
    growth, c = float(prm.get("backtrack_factor", 2.0)), float(prm.get("descent_c", 1e-4))
    f_tol = float(prm.get("relative_objective_tol", 1e-12))
    max_backtracks = int(prm.get("max_backtracks", 120))
    if not all(np.isfinite(a) and a > 0 for a in (penalty, mu, mu_min, eps, eps_min, Lmin, Lmax, c, f_tol)):
        raise ValueError("Coefficients, smoothing values and tolerances must be positive and finite")
    if mu_min > mu or not 0 < reduction < 1 or eps_divisor <= 1 or growth <= 1 or Lmin > Lmax:
        raise ValueError("Invalid continuation, line-search or curvature bounds")
    if max_inner < 1 or memory < 0 or max_backtracks < 1:
        raise ValueError("Invalid iteration, memory or backtracking limit")
    x = vec(prm.get("x0", b)).copy()
    reference = x.copy()
    if x.shape != b.shape or not np.all(np.isfinite(x)):
        raise ValueError("x0 must match the observation and be finite")
    stages, total = [], 0
    # Numerical tolerance keeps the mathematically equal mu_min stage included.
    while mu >= mu_min * (1 - 1e-12):
        value, _, dx, prox = moreau_objective(x, b, D, penalty, mu, loss)
        ref_value = moreau_objective(reference, b, D, penalty, mu, loss)[0]
        reset = value > ref_value
        if reset:
            x = reference.copy()
            value, _, dx, prox = moreau_objective(x, b, D, penalty, mu, loss)
        history = [value]
        hgrad = DT(dx) / mu + (x-b if loss == "l2" else 0)
        curvature = float(np.clip(1.0, Lmin, Lmax))
        stage_backtracks = 0
        reason = "max_inner"
        for inner in range(1, max_inner+1):
            direction = hgrad - DT(prox) / mu
            L = curvature
            for ls in range(max_backtracks):
                trial = x - direction / L
                if loss == "l1":
                    trial = b + soft_threshold(trial-b, 1 / (b.size*L))
                next_value, _, next_dx, next_prox = moreau_objective(trial, b, D, penalty, mu, loss)
                step = trial-x
                sq_step = float(np.dot(step, step))
                if np.isfinite(next_value) and next_value <= max(history[-(memory+1):]) - c*sq_step/2:
                    break
                L *= growth
            else:
                raise RuntimeError(f"SDCAM line search failed at mu={mu:g}, inner={inner}; no step was committed")
            stage_backtracks += ls
            rel_step = np.sqrt(sq_step) / max(1.0, np.linalg.norm(trial))
            rel_value = abs(next_value-value) / max(1.0, abs(next_value))
            next_hgrad = DT(next_dx) / mu + (trial-b if loss == "l2" else 0)
            if sq_step:
                curvature = float(np.clip(np.dot(step, next_hgrad-hgrad)/sq_step, Lmin, Lmax))
            else:
                curvature = float(np.clip(1.0, Lmin, Lmax))
            x, value, dx, prox, hgrad = trial, next_value, next_dx, next_prox, next_hgrad
            history.append(value)
            if len(history) > memory+1:
                del history[0]
            if rel_step < eps/L:
                reason = "relative_step"
                break
            if rel_value < f_tol:
                reason = "relative_objective"
                break
        total += inner
        stages.append({"smoothing": mu, "epsilon": eps, "iterations": inner,
                       "stop_reason": reason, "reference_reset": bool(reset),
                       "backtracks": stage_backtracks, "objective": value,
                       "relative_step": float(rel_step), "accepted_curvature": L})
        mu *= reduction
        eps = max(eps/eps_divisor, eps_min)
    exact_loss = np.abs(x-b).sum()/b.size if loss == "l1" else 0.5*np.dot(x-b,x-b)
    jump = D(x)
    info = {"iter": total, "stop_reason": "smoothing_threshold", "nu_applicable": False,
            "method": "sdcam_l1" if loss == "l1" else "sdcam_l2",
            "loss": loss, "lambda_half": penalty, "smoothing_final": stages[-1]["smoothing"],
            "smoothing_next": mu, "outer_iterations": len(stages), "stages": stages,
            "inner_cap_reached": any(s["stop_reason"] == "max_inner" for s in stages),
            "objective": float(exact_loss + penalty*np.sqrt(np.abs(jump)).sum())}
    return x, jump, info


def padmm_l0(observation, D, parameters=None):
    """pADMM for .5*||x-b_hat||^2 + lambda_l0*||y||_0, subject to Dx=y.

    M1=(4*r+epsilon)I-r*D.T*D, M2=0. y is the split primal
    variable and z is the unscaled dual variable, as in the supplement.
    """
    prm = parameters or {}
    b = vec(observation)
    weight, r = float(prm.get("lambda_l0", 200)), float(prm.get("r", 100))
    relaxation = float(prm.get("relaxation", 1.9))
    beta = 4*r + float(prm.get("beta_eps", 1e-8))
    tolerance, budget = float(prm.get("stop_tol", 2e-4)), int(prm.get("max_iter", 20000))
    if not b.size or not np.all(np.isfinite(b)) or D.shape[1] != b.size:
        raise ValueError("Input must be finite and compatible with D")
    if not all(np.isfinite(a) and a > 0 for a in (weight,r,beta,tolerance)) or not 0 < relaxation < 2 or budget < 1:
        raise ValueError("Invalid penalty, relaxation, tolerance or budget")
    x = b.copy()
    y, z = np.zeros(D.shape[0]), np.zeros(D.shape[0])
    for iteration in range(1,budget+1):
        argument = D@x + z/r
        next_y = np.where(np.abs(argument) > np.sqrt(2*weight/r), argument, 0.0)
        next_x = (b + beta*x - D.T@(z + r*(D@x-next_y))) / (1+beta)
        next_z = z + relaxation*r*(D@next_x-next_y)
        change = max(np.linalg.norm(next_x-x), np.linalg.norm(next_y-y), np.linalg.norm(next_z-z))
        x,y,z = next_x,next_y,next_z
        if change < tolerance:
            break
    info = {"iter": iteration, "stop_reason": "converged" if change < tolerance else "max_iter",
            "nu_applicable": False, "method": "padmm_l0", "final_change": float(change),
            "lambda_l0": weight, "r": r, "relaxation": relaxation, "beta": beta,
            "constraint_residual": float(np.linalg.norm(D@x-y)),
            "objective": float(0.5*np.dot(x-b,x-b) + weight*np.count_nonzero(y))}
    return x,y,info
