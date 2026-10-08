"""Run the one-dimensional SAP-ADMM experiment and save its data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Support running this file directly in Spyder or with python /path/to/file.py.
# Resolve imports from this checkout independently of the working directory.
if __package__ in (None, ""):
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    __package__ = "experiments"

import numpy as np

from sap_admm import sap_admm, sap_admm_halpern, sap_admm_l1, padmm_l0, sdcam
from sap_admm.utils import compute_f1_from_support, difference_matrix, random_y

from ._shared import REPOSITORY_ROOT, load_config, save_metadata, save_npz, write_csv, save_diagnostics

METHODS = ("sap_admm", "sap_admm_halpern", "sap_admm_l1", "sdcam_l1", "sdcam_l2", "padmm_l0")


def generate_input(config: dict, probability_index: int, trial_index: int):
    seed = trial_index + 1 + (probability_index + 1) * 100
    rng = np.random.RandomState(seed)
    n = config["n"]
    truth = random_y(n, 50, 150, -5, 10, rng)
    observation = truth + config["gaussian_std"] * rng.randn(n)
    mask = rng.rand(n) < config["probabilities"][probability_index]
    observation[mask] += config["impulse_scale"] * (rng.rand(mask.sum()) - 0.5)
    return truth, observation, seed


def solver_parameters(config: dict, figure: bool = False) -> dict:
    return {
        "rho_scale": config.get("rho_scale", 2.0),
        "beta_factor": config.get("beta_factor", 6),
        "beta_eps": config.get("beta_eps", 1e-8),
        "lambda1_factor": config.get("lambda1_factor", 500),
        "lambda2_capped": config.get("lambda2_capped", 0.016),
        "lambda2_l1": config.get("lambda2_l1", 0.020),
        "max_iter": config["max_iter"],
        "tol_stop": config["stop_tol"],
        "alpha": config["alpha"],
        "t": config["t"],
        "max_fail": config["figure_max_fail"] if figure else config["max_fail"],
        "restart_iter": config["restart_iter"],
        "nu0": config["nu0"],
        "nu_switch_updates": config["nu_switch_updates"],
        "nu_decay_early": config["nu_decay_early"],
        "nu_decay_late": config["nu_decay_late"],
        "min_iterations_after_floor": config["min_iterations_after_floor"],
    }


def solve(observation: np.ndarray, method: str, config: dict, figure: bool = False):
    D = difference_matrix(observation.size)
    solvers = {"sap_admm": sap_admm, "sap_admm_halpern": sap_admm_halpern,
               "sap_admm_l1": sap_admm_l1}
    start = time.perf_counter()
    if method in solvers:
        x, jump, info = solvers[method](observation, D, solver_parameters(config, figure))
    elif method == "padmm_l0":
        x, jump, info = padmm_l0(observation, D, config["padmm"])
    elif method in {"sdcam_l1", "sdcam_l2"}:
        loss = "l1" if method == "sdcam_l1" else "l2"
        parameters = dict(config["sdcam"])
        parameters["lambda_half"] = parameters[f"lambda_half_{loss}"]
        x, jump, info = sdcam(observation, (lambda x: D@x, lambda y: D.T@y), parameters, loss=loss)
    else:
        raise ValueError(f"Unknown signal method: {method}")
    return x, jump, info, time.perf_counter() - start


def run(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_metadata(output, config, "one-dimensional signal recovery")
    R, P, M, n = config["trials"], len(config["probabilities"]), len(METHODS), config["n"]
    shape = (R, P, M)
    data = {key: np.full(shape, np.nan) for key in ("f1", "mse", "seconds", "iterations")}
    data.update(
        truth=np.zeros((R, P, n)), observations=np.zeros((R, P, n)),
        recovered=np.zeros((*shape, n)), jumps=np.zeros((*shape, n - 1)),
        seeds=np.zeros((R, P), dtype=int), probabilities=np.asarray(config["probabilities"]),
        methods=np.asarray(METHODS), nu_final=np.full(shape, np.nan),
        nu_floor=np.full(shape, np.nan), nu_floor_iteration=np.zeros(shape, dtype=int),
        nu_iterations_after_floor=np.zeros(shape, dtype=int),
        nu_applicable=np.zeros(shape, dtype=bool),
        nu_post_floor_satisfied=np.zeros(shape, dtype=bool), completed=np.zeros(shape, dtype=bool),
    )
    rows, diagnostics = [], []
    for j, probability in enumerate(config["probabilities"]):
        for r in range(R):
            truth, observation, seed = generate_input(config, j, r)
            data["truth"][r, j], data["observations"][r, j], data["seeds"][r, j] = truth, observation, seed
            support = np.flatnonzero(np.abs(np.diff(truth)) > config["support_tol"])
            for k, method in enumerate(METHODS):
                x, jump, info, seconds = solve(observation, method, config)
                idx = (r, j, k)
                f1 = compute_f1_from_support(support, np.flatnonzero(np.abs(jump) > config["support_tol"]))
                mse = float(np.mean((truth - x) ** 2))
                for key, value in (("f1", f1), ("mse", mse), ("seconds", seconds),
                                   ("iterations", info["iter"]),
                                   ("nu_applicable", info["nu_applicable"]),
                                   ("nu_final", info.get("nu_final", np.nan)),
                                   ("nu_floor", info.get("nu_floor", np.nan)),
                                   ("nu_floor_iteration", info.get("nu_floor_iteration", 0)),
                                   ("nu_iterations_after_floor", info.get("nu_iterations_after_floor", 0)),
                                   ("nu_post_floor_satisfied", info.get("nu_post_floor_satisfied", False))):
                    data[key][idx] = value
                data["recovered"][idx], data["jumps"][idx], data["completed"][idx] = x, jump, True
                rows.append({
                    "Trial": r + 1, "ImpulseProb": probability, "Method": method,
                    "F1": f1, "MSE": mse, "Time_s": seconds, "Iterations": info["iter"],
                    "NuApplicable": info["nu_applicable"],
                    "NuFinal": info.get("nu_final", ""), "NuFloor": info.get("nu_floor", ""),
                    "NuFloorIteration": info.get("nu_floor_iteration", ""),
                    "NuIterationsAfterFloor": info.get("nu_iterations_after_floor", ""),
                    "NuPostFloorSatisfied": info.get("nu_post_floor_satisfied", ""),
                    "StopReason": info["stop_reason"],
                })
                diagnostics.append({"Trial": r+1, "ImpulseProb": probability, "Method": method, "solver": info})
                print(f"pi={probability:.2f} trial={r+1}/{R} {method}: F1={f1:.4f} MSE={mse:.6g} iter={info['iter']} time={seconds:.3f}s", flush=True)
            save_npz(output / "results.npz", **data)
            write_csv(output / "trial_metrics.csv", rows)
            save_diagnostics(output, diagnostics)
    summary = []
    for j, probability in enumerate(config["probabilities"]):
        for k, method in enumerate(METHODS):
            summary.append({
                "ImpulseProb": probability, "Method": method, "Trials": R,
                "MeanF1": float(data["f1"][:, j, k].mean()),
                "ExactRecoveryRate": float((data["f1"][:, j, k] == 1).mean()),
                "MeanMSE": float(data["mse"][:, j, k].mean()),
                "MeanTime_s": float(data["seconds"][:, j, k].mean()),
                "MeanIterations": float(data["iterations"][:, j, k].mean()),
                "ExactF1Count": int((data["f1"][:, j, k] == 1).sum()),
                "Efficiency": float(data["f1"][:, j, k].mean()/data["seconds"][:, j, k].mean()),
            })
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paper_mse_time_table.csv", [
        {"Algorithm": method, **{f"pi_{p:.2f}":
         f"{data['mse'][:, j, k].mean():.6g}/{data['seconds'][:, j, k].mean():.6g}"
         for j,p in enumerate(config["probabilities"])}} for k,method in enumerate(METHODS)])

    rng = np.random.RandomState(config["recovery_seed"])
    truth = random_y(n, 50, 150, -5, 10, rng)
    examples = []
    for probability in (0.0, 0.2):
        observation = truth + config["gaussian_std"] * rng.randn(n)
        mask = rng.rand(n) < probability
        observation[mask] += config["impulse_scale"] * (rng.rand(mask.sum()) - 0.5)
        recovered = [solve(observation, method, config, figure=True)[0] for method in METHODS]
        examples.append((observation, *recovered))
    save_npz(
        output / "recovery_example.npz", truth=truth,
        observations=np.stack([x[0] for x in examples]),
        recovered=np.stack([x[1:] for x in examples]),
        probabilities=np.asarray([0.0, 0.2]), methods=np.asarray(METHODS),
        seed=np.asarray(config["recovery_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results" / "signal")
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    config = load_config("signal.json")
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be positive")
        config["trials"] = args.trials
    run(config, args.out)


if __name__ == "__main__":
    main()
