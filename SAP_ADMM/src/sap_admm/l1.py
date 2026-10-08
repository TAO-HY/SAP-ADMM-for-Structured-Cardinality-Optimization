"""SAP-ADMM^1 for the convex l1-loss / l1-penalty model.

The proximal update and accelerated iteration follow the supplied comparison
code. There is no capped-penalty branch selection or continuation parameter.
State order is (x, y, p, q, lam, mu), with lam representing eta in the paper.
"""
from __future__ import annotations

import numpy as np

from .solver import _accelerate, _change, _initial
from .utils import image_operators, soft_threshold, vec


def _bar_step_l1(state, observation, D, DT, rho, beta, lambda_p, lambda_l1):
    """One proximal output for ||x-b_hat||_1/n + lambda_p||Dx-y||_2 + lambda_l1||y||_1."""
    x, y, p, q, lam, mu = state
    v1 = x + lam / rho
    pbar = observation + soft_threshold(v1 - observation, 1 / (rho * observation.size))
    lambar = lam + rho * (x - pbar)
    B2u = D(x) - y
    v2 = B2u + mu / rho
    nv = np.linalg.norm(v2)
    qbar = v2 * max(1 - (lambda_p / rho) / nv, 0) if nv > 0 else v2 * 0
    mubar = mu + rho * (B2u - qbar)
    term = rho * (B2u - qbar) + mubar
    grad = (rho * (x - pbar) + lambar) + DT(term)
    xbar = x - (1 / beta) * grad
    vy = y + (1 / beta) * term
    ybar = soft_threshold(vy, lambda_l1 / beta)
    return (xbar, ybar, pbar, qbar, lambar, mubar)


def _solve_l1(observation, D, DT, *, image, parameters):
    observation = vec(observation)
    if observation.size == 0 or not np.all(np.isfinite(observation)):
        raise ValueError("observation must be nonempty and finite")
    prm = parameters
    n = observation.size
    rho = prm.get("rho_scale", 0.5 if image else 2.0) / n
    beta = rho * prm.get("beta_factor", 10 if image else 6) + prm.get("beta_eps", 1e-8)
    lambda_p = (1 / np.sqrt(n)) * prm.get("lambda1_factor", 2 if image else 500)
    lambda_l1 = prm.get("lambda2_l1", 7e-4 if image else 0.020)
    alpha = prm.get("acc_alpha", prm.get("alpha", 15.0))
    t = prm.get("acc_t", prm.get("t", 1.5))
    tol = prm.get("stop_tol", prm.get("tol_stop", 4e-4 if image else 2e-4))
    max_iter = int(prm.get("max_iter", 20000))
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not all(np.isfinite(value) and value > 0 for value in (rho, beta, lambda_p, lambda_l1, tol)):
        raise ValueError("rho, beta, penalty coefficients and tolerance must be positive and finite")
    if not np.isfinite(alpha) or alpha < 2 or not np.isfinite(t) or not 0 < t <= 2:
        raise ValueError("alpha must be at least 2 and t must lie in (0, 2]")

    state = _initial(observation, D)
    hat, bar = state, state
    profile = prm.get("stopping_profile", "supplement")
    if profile not in {"supplement", "legacy_image"}:
        raise ValueError("Unknown l1 stopping_profile")
    legacy_image = image and profile == "legacy_image"
    budget = max_iter + 1 if legacy_image else max_iter
    trace = [] if prm.get("record_trace", False) else None
    for k in range(budget):
        bar = _bar_step_l1(state, observation, D, DT, rho, beta, lambda_p, lambda_l1)
        count = k + 1
        change = _change(bar, state, full=not legacy_image)
        if trace is not None:
            trace.append((count, change))
        if change < tol or count >= budget:
            break
        state, hat = _accelerate(state, bar, hat, k, alpha, t)

    info = {
        "iter": count, "bar_update_count": count,
        "max_change": change, "final_change": change, "tol_stop": tol,
        "requested_max_iter": max_iter, "effective_max_iter": budget,
        "stop_reason": "converged" if change < tol else "max_iter",
        "stopping_blocks": "x" if legacy_image else "x,y,p,q,eta,mu",
        "stopping_profile": profile,
        "alpha": alpha, "t": t, "rho": rho, "beta": beta,
        "lambda_p": lambda_p, "lambda_l1": lambda_l1,
        "nu_applicable": False, "restarted": False,
        "penalty": "l1", "method": "sap_admm_l1",
    }
    if trace is not None:
        info["trace"] = trace
    return bar[0], bar[1], info


def sap_admm_l1(p_hat, D, opts=None):
    """Return (x_bar, y_bar, info) for the paper's signal SAP-ADMM^1 model.

    lambda2_l1 is lambda_{ell_1}; defaults are alpha=15, t=1.5,
    rho=2/n, beta=6*rho+1e-8, lambda_p=500/sqrt(n), lambda_l1=0.020.
    Continuation, post-floor and restart settings are inapplicable.
    """
    DT = D.T
    return _solve_l1(p_hat, lambda x: D @ x, lambda y: DT @ y,
                     image=False, parameters=opts or {})


def sap_admm_l1_image(u_obs, ops=None, parameters=None):
    """Return (clipped image, update count, info) for image SAP-ADMM^1.

    The supplement profile uses lambda_l1=7e-4, rho=0.5/n, beta=10*rho+1e-8,
    lambda_p=2/sqrt(n), and six-block stopping. Images use column-major vectors.
    stopping_profile='legacy_image' is only for historical source verification.
    """
    observation = np.asarray(u_obs)
    if observation.ndim != 2:
        raise ValueError("u_obs must be a two-dimensional image")
    D, DT = ops or image_operators(*observation.shape)
    x, _, info = _solve_l1(observation, D, DT, image=True, parameters=parameters or {})
    restored = np.clip(x, 0, 1).reshape(observation.shape, order="F")
    return restored, info["iter"], info
