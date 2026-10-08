"""SAP-ADMM solvers for capped-l1 structured sparsity.

State order is (x,y,p,q,lam,mu). Effective bar updates, not rejected trial
computations, control the continuation schedule and accelerated iteration.
"""
from __future__ import annotations

import numpy as np

from .utils import NuMonitor, image_operators, soft_threshold, vec


def _initial(observation, D):
    x = vec(observation).copy()
    dx = D(x)
    y = np.zeros_like(dx)
    return (x, y, x.copy(), dx.copy(), np.zeros_like(x), np.zeros_like(dx))


def _branch(y, nu):
    index = np.zeros(y.shape, dtype=np.int8)
    index[y <= -nu] = -1
    index[y >= nu] = 1
    return index


def _bar_step(state, index, nu, observation, D, DT, rho, beta, lambda1, lambda2):
    x, y, p, q, lam, mu = state
    v1 = x + lam / rho
    pbar = observation + soft_threshold(v1 - observation, 1 / (rho * observation.size))
    lambar = lam + rho * (x - pbar)
    B2u = D(x) - y
    v2 = B2u + mu / rho
    nv = np.linalg.norm(v2)
    qbar = v2 * max(1 - (lambda1 / rho) / nv, 0) if nv > 0 else v2 * 0
    mubar = mu + rho * (B2u - qbar)
    term = rho * (B2u - qbar) + mubar
    grad = (rho * (x - pbar) + lambar) + DT(term)
    xbar = x - (1 / beta) * grad
    vy = y + (1 / beta) * term
    ybar = soft_threshold(vy + index * lambda2 / (beta * nu), lambda2 / (beta * nu))
    next_index = _branch(ybar, nu)
    return (xbar, ybar, pbar, qbar, lambar, mubar), next_index


def _change(bar, state, full=True):
    if full:
        return max(float(np.linalg.norm(a - b)) for a, b in zip(bar, state))
    return float(np.linalg.norm(bar[0] - state[0]))


def _average(anchor, bar, k):
    ak, bk = 1 / (k + 2), (k + 1) / (k + 2)
    return tuple(ak * a + bk * b for a, b in zip(anchor, bar))


def _accelerate(state, bar, hat, k, alpha, t):
    hat_next = tuple((1 - t) * a + t * b for a, b in zip(state, bar))
    c1, c2 = alpha / (2 * (k + alpha)), k / (k + alpha)
    trial = tuple(a + c1 * (h - a) + c2 * (h - prev)
                  for a, h, prev in zip(state, hat_next, hat))
    return trial, hat_next


def _solve(observation, D, DT, *, accelerated, image, parameters):
    observation = vec(observation)
    N = observation.size
    prm = parameters
    rho = prm.get("rho_scale", 0.5 if image else 2.0) / N
    beta = rho * prm.get("beta_factor", 10 if image else 6) + prm.get("beta_eps", 1e-8)
    lambda1 = (1 / np.sqrt(N)) * prm.get("lambda1_factor", 2 if image else 500)
    lambda2 = prm.get("lambda2_capped", 8e-5 if image else 0.016)
    ell = lambda2 / (3 * lambda1)
    nu = prm.get("nu0", 0.1 if image else 2.0)
    monitor = NuMonitor(
        nu,
        (0.999 if image else 0.99) * ell,
        prm.get("min_iterations_after_floor", 50),
    )
    max_iter = int(prm.get("max_iter", 20000))
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    tol = prm.get("stop_tol", prm.get("tol_stop", 4e-4 if image else 2e-4))
    alpha = prm.get("acc_alpha", prm.get("alpha", 15.0)) if accelerated else 2.0
    t = prm.get("acc_t", prm.get("t", 1.5)) if accelerated else 1.0
    max_fail = prm.get("max_fail", 500)
    state = _initial(observation, D)
    anchor, hat, bar = state, state, state
    index = _branch(state[1], nu)
    count, accepted, fallback, outer, local_k = 0, 0, 0, 0, 0
    restarted = False
    trace = [] if prm.get("record_trace", False) else None

    def decay(old, effective_count):
        if image:
            return max(old * prm.get("nu_decay", 0.99), 0.999 * ell)
        gamma = (prm.get("nu_decay_early", 0.999)
                 if effective_count < prm.get("nu_switch_updates", 1500)
                 else prm.get("nu_decay_late", 0.95))
        return max(old * gamma, 0.99 * ell)

    if not all(0 < prm.get(key, default) < 1 for key, default in
               (("nu_decay", .99), ("nu_decay_early", .999),
                ("nu_decay_late", .95))):
        raise ValueError("nu decay factors must be between zero and one")

    def step(current, branch, current_nu):
        return _bar_step(current, branch, current_nu, observation, D, DT,
                         rho, beta, lambda1, lambda2)

    if accelerated:
        enabled = not (max_fail <= 0 or (alpha == 2 and t == 1))
        need_bar = True
        change = float("inf")
        while count < max_iter:
            outer += 1
            if need_bar:
                if not image or count > 0 or prm.get("decay_first_image_bar", True):
                    nu = decay(nu, count)
                count += 1
                monitor.record(nu, count)
                bar, index_next = step(state, index, nu)
            k = count - 1
            change = _change(bar, state)
            if (change < tol and monitor.can_stop()) or count >= max_iter:
                break
            stable = False
            if enabled:
                trial, hat_next = _accelerate(state, bar, hat, k, alpha, t)
                nu_trial = decay(nu, count)
                trial_bar, trial_index = step(trial, index_next, nu_trial)
                stable = np.array_equal(trial_index, index_next)
            if enabled and stable:
                state, hat, bar = trial, hat_next, trial_bar
                nu = nu_trial
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
                state = _average(anchor, bar, k)
                hat = bar
                if enabled:
                    fallback += 1
                    if fallback >= max_fail:
                        enabled = False
                index = index_next
                need_bar = True
                if trace is not None:
                    trace.append((count, nu, "fallback", change))
        info = {"iter": count, "bar_update_count": count, "outer_count": outer,
                "accepted_count": accepted, "fallback_count": fallback,
                "max_fail": max_fail, "acc_enabled": enabled, "acc_disabled": not enabled,
                "nu_final": nu, "max_change": change, "final_change": change,
                "alpha": alpha, "t": t}
    else:
        budget = max_iter
        change = float("inf")
        restart_iter = int(prm.get("restart_iter", 1000))
        for k in range(budget):
            if not image and not restarted and k == restart_iter:
                anchor = state
                local_k = 0
                restarted = True
            nu = decay(nu, k)
            bar, index = step(state, index, nu)
            count = k + 1
            monitor.record(nu, count)
            change = _change(bar, state, full=not image)
            if trace is not None:
                trace.append((count, nu, "bar", change))
            if change < tol and monitor.can_stop():
                break
            state = _average(anchor, bar, local_k if not image else k)
            local_k += 1
        info = {"iter": count, "max_change": change, "final_change": change,
                "max_iter": max_iter, "tol_stop": tol, "alpha": alpha, "t": t,
                "restarted": restarted, "restart_iter": restart_iter,
                "halpern_k_final": local_k, "nu_final": nu}
    info["requested_max_iter"] = int(prm.get("max_iter", 20000))
    info["effective_max_iter"] = max_iter
    info.update(monitor.result())
    info["stop_reason"] = "converged" if change < tol and monitor.can_stop() else "max_iter"
    if trace is not None:
        info["trace"] = trace
    info["rho"], info["beta"] = rho, beta
    info["lambda_p"], info["lambda_0"] = lambda1, lambda2
    return bar[0], bar[1], info


def sap_admm_halpern(p_hat, D, opts=None):
    """Run SAP-ADMM with alpha=2, t=1, and one signal restart."""
    DT = D.T
    return _solve(p_hat, lambda x: D @ x, lambda y: DT @ y,
                  accelerated=False, image=False, parameters=opts or {})


def sap_admm(p_hat, D, opts=None):
    """Run accelerated SAP-ADMM for a one-dimensional signal."""
    DT = D.T
    return _solve(p_hat, lambda x: D @ x, lambda y: DT @ y,
                  accelerated=True, image=False, parameters=opts or {})


def sap_admm_image(u_obs, ops=None, parameters=None):
    """Run accelerated SAP-ADMM for a two-dimensional image."""
    D, DT = ops or image_operators(*np.asarray(u_obs).shape)
    x, y, info = _solve(u_obs, D, DT, accelerated=True,
                        image=True, parameters=parameters or {})
    return np.clip(x, 0, 1).reshape(np.asarray(u_obs).shape, order="F"), info["iter"], info
