"""Run SAP-ADMM with a general rectangular matrix A and save its data."""

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
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

from sap_admm import sap_admm_generalA
from sap_admm.utils import compute_f1_from_support, difference_matrix, random_y

from ._shared import REPOSITORY_ROOT, load_config, save_metadata, save_npz, write_csv


def run(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_metadata(output, config, "general rectangular matrix A")
    n, m, R = config["n"], config["m"], config["trials"]
    values = config["Nmax_list"]
    shape = (R, len(values))
    data = {key: np.full(shape, np.nan) for key in ("iterations", "seconds", "f1", "mse", "fallback")}
    data.update(
        recovered=np.zeros((R, len(values), n)), jumps=np.zeros((R, len(values), n - 1)),
        seeds=10000 + np.arange(1, R + 1), Nmax_list=np.asarray(values),
        nu_final=np.full(shape, np.nan), nu_floor=np.full(shape, np.nan),
        nu_floor_iteration=np.zeros(shape, dtype=int),
        nu_iterations_after_floor=np.zeros(shape, dtype=int),
        nu_post_floor_satisfied=np.zeros(shape, dtype=bool), completed=np.zeros(shape, dtype=bool),
    )
    records = []
    D = difference_matrix(n)
    with threadpool_limits(limits=config["blas_threads"]):
        for r in range(R):
            rng = np.random.RandomState(10000 + r + 1)
            A = rng.randn(m * n).reshape(m, n, order="F")
            A /= svdvals(A, check_finite=False)[0]
            truth = random_y(n, 50, 150, -5, 10, rng)
            observation = A @ truth + config["gaussian_std"] * rng.randn(m)
            mask = rng.rand(m) < config["impulse_prob"]
            observation[mask] += config["impulse_scale"] * (rng.rand(mask.sum()) - 0.5)
            support = np.flatnonzero(np.abs(D @ truth) > config["support_tol"])
            base = {key: config[key] for key in (
                "alpha", "t", "max_iter", "tol_stop", "rho", "beta",
                "lambda_p", "lambda_0", "nu0", "min_iterations_after_floor",
            )}
            if base["beta"] is None:
                base["beta"] = 6 * base["rho"] + 1e-8
            if base["lambda_p"] is None:
                base["lambda_p"] = 500 / np.sqrt(m)
            base.update(x0=A.T @ observation, y0=np.zeros(n - 1))
            for j, nmax in enumerate(values):
                start = time.perf_counter()
                x, jump, info = sap_admm_generalA(observation, A, D, n, {**base, "Nmax": nmax})
                seconds = time.perf_counter() - start
                idx = (r, j)
                f1 = compute_f1_from_support(support, np.flatnonzero(np.abs(jump) > config["support_tol"]))
                mse = float(np.mean((truth - x) ** 2))
                for key, value in (("iterations", info["iter"]), ("seconds", seconds),
                                   ("f1", f1), ("mse", mse), ("fallback", info["fallback_count"]),
                                   ("nu_final", info["nu_final"]), ("nu_floor", info["nu_floor"]),
                                   ("nu_floor_iteration", info["nu_floor_iteration"]),
                                   ("nu_iterations_after_floor", info["nu_iterations_after_floor"]),
                                   ("nu_post_floor_satisfied", info["nu_post_floor_satisfied"])):
                    data[key][idx] = value
                data["recovered"][idx], data["jumps"][idx], data["completed"][idx] = x, jump, True
                records.append({
                    "Trial": r + 1, "Nmax": nmax, "F1": f1, "MSE": mse,
                    "Time_s": seconds, "Iterations": info["iter"],
                    "Fallback": info["fallback_count"], "NuFinal": info["nu_final"],
                    "NuFloor": info["nu_floor"], "NuFloorIteration": info["nu_floor_iteration"],
                    "NuIterationsAfterFloor": info["nu_iterations_after_floor"],
                    "NuPostFloorSatisfied": info["nu_post_floor_satisfied"],
                    "StopReason": info["stop_reason"],
                })
                print(f"trial={r+1}/{R} Nmax={nmax}: F1={f1:.5f} MSE={mse:.6g}")
                save_npz(output / "results.npz", **data)
                write_csv(output / "trial_metrics.csv", records)
    summary = []
    for j, nmax in enumerate(values):
        summary.append({
            "Nmax": nmax, "Trials": R,
            "MeanIterations": float(data["iterations"][:, j].mean()),
            "MeanF1": float(data["f1"][:, j].mean()),
            "MinF1": float(data["f1"][:, j].min()),
            "ExactF1Count": int((data["f1"][:, j] == 1).sum()),
            "MeanMSE": float(data["mse"][:, j].mean()),
            "MeanTime_s": float(data["seconds"][:, j].mean()),
        })
    write_csv(output / "summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results" / "general_a")
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    config = load_config("general_a.json")
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be positive")
        config["trials"] = args.trials
    run(config, args.out)


if __name__ == "__main__":
    main()
