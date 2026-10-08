"""Safeguarded accelerated proximal ADMM for a rectangular matrix A."""
from __future__ import annotations

import numpy as np
from .utils import NuMonitor, soft_threshold, vec
from .solver import _accelerate, _average, _branch, _change


def update_nu_by_bar_count(nu_old, bar_count, floor_value):
    """Decay before the next update: first 4000 use 0.9995, then 0.95."""
    gamma = 0.9995 if bar_count < 4000 else 0.95
    return max(nu_old * gamma, floor_value)


def sap_admm_generalA(b_hat, A, D, n, opts=None):
    opts = opts or {}
    def get(name, default):
        value = opts.get(name)
        return default if value is None or np.size(value) == 0 else value
    mA, nA = A.shape
    if nA != n or D.shape[1] != n or np.size(b_hat) != mA:
        raise ValueError("A/D column dimensions or b_hat length do not match")
    b_hat = vec(b_hat)
    mD = D.shape[0]
    rho = get("rho", 2 / n)
    beta = get("beta", 6 * rho + 1e-8)
    Lf = 1 / np.sqrt(mA)
    lambda_p, lambda_0 = get("lambda_p", 500 * Lf), get("lambda_0", 0.016)
    alpha, t = get("alpha", 15), get("t", 1.5)
    Nmax = get("Nmax", 500)
    max_iter, tol = int(get("max_iter", 20000)), get("tol_stop", 2e-4)
    nu = get("nu0", 2)
    if alpha < 2 or not 0 < t <= 2 or Nmax < 1 or Nmax != int(Nmax):
        raise ValueError("Require alpha>=2, 0<t<=2, and a positive integer Nmax")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    floor_value = 0.99 * lambda_0 / (3 * lambda_p)
    monitor = NuMonitor(nu, floor_value, get("min_iterations_after_floor", 50))
    x, y = vec(get("x0", A.T @ b_hat)), vec(get("y0", np.zeros(mD)))
    if x.size != n or y.size != mD:
        raise ValueError("x0/y0 dimensions do not match")
    state = (x, y, A @ x, D @ x - y, np.zeros(mA), np.zeros(mD))
    anchor, hat, bar = state, state, state
    index = _branch(y, nu)
    accepted = fallback = count = outer = 0
    enabled = not (alpha == 2 and t == 1)
    need_bar = True
    trace = [] if get("record_trace", False) else None

    def compute_bar_step(current, branch, current_nu):
        xi, yi, pi, qi, eta, mu = current
        Ax = A @ xi
        v1 = Ax + eta / rho
        pb = b_hat + soft_threshold(v1 - b_hat, 1 / (rho * mA))
        etab = eta + rho * (Ax - pb)
        B2u = D @ xi - yi
        v2 = B2u + mu / rho
        nv = np.linalg.norm(v2)
        qb = v2 * max(1 - (lambda_p / rho) / nv, 0) if nv > 0 else np.zeros_like(v2)
        mub = mu + rho * (B2u - qb)
        zeta = rho * (B2u - qb) + mub
        grad = A.T @ (rho * (Ax - pb) + etab) + D.T @ zeta
        xb = xi - grad / beta
        tau = yi + zeta / beta
        tau_bar = tau + branch * lambda_0 / (beta * current_nu)
        yb = soft_threshold(tau_bar, lambda_0 / (beta * current_nu))
        return (xb, yb, pb, qb, etab, mub), _branch(yb, current_nu)

    change = float("inf")
    while count < max_iter:
        outer += 1
        if need_bar:
            nu = update_nu_by_bar_count(nu, count, floor_value)
            count += 1
            monitor.record(nu, count)
            bar, index_next = compute_bar_step(state, index, nu)
        k = count - 1
        change = _change(bar, state)
        if (change < tol and monitor.can_stop()) or count >= max_iter:
            break
        stable = False
        if enabled:
            trial, hat_next = _accelerate(state, bar, hat, k, alpha, t)
            nu_trial = update_nu_by_bar_count(nu, count, floor_value)
            trial_bar, trial_index = compute_bar_step(trial, index_next, nu_trial)
            stable = np.array_equal(trial_index, index_next)
        if enabled and stable:
            state, hat, bar, nu = trial, hat_next, trial_bar, nu_trial
            count += 1
            monitor.record(nu, count)
            accepted += 1
            index, index_next = index_next, trial_index
            need_bar = False
            change = _change(bar, state)
            if trace is not None:
                trace.append((count, nu, "accepted", change))
            if (change < tol and monitor.can_stop()) or count >= max_iter:
                break
        else:
            state, hat = _average(anchor, bar, k), bar
            if enabled:
                fallback += 1
                if fallback >= Nmax:
                    enabled = False
            index = index_next
            need_bar = True
            if trace is not None:
                trace.append((count, nu, "fallback", change))
    info = {"iter": count, "bar_update_count": count, "outer_count": outer,
            "accepted_count": accepted, "fallback_count": fallback, "Nmax": Nmax,
            "acc_enabled": enabled, "acc_disabled": not enabled, "alpha": alpha, "t": t,
            "tol_stop": tol, "max_change": change, "nu_final": nu, "rho": rho, "beta": beta,
            "lambda_p": lambda_p, "lambda_0": lambda_0, "Lf": Lf}
    info["requested_max_iter"] = max_iter
    info["effective_max_iter"] = max_iter
    info.update(monitor.result())
    info["stop_reason"] = "converged" if change < tol and monitor.can_stop() else "max_iter"
    if trace is not None:
        info["trace"] = trace
    return bar[0], bar[1], info
