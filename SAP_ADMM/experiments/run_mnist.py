"""Run the MNIST SAP-ADMM experiment and save its data."""

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

from sap_admm import sap_admm_image, sap_admm_l1_image, sdcam
from sap_admm.utils import compute_gmsd_value, compute_ssim_value, image_operators, psnr

from ._shared import REPOSITORY_ROOT, load_config, save_metadata, save_npz, write_csv, save_diagnostics

METHODS = ("sap_admm", "sap_admm_l1", "sdcam_l1", "sdcam_l2")


def solve(observation, method, config, operators=None):
    operators = operators or image_operators(*observation.shape)
    start = time.perf_counter()
    if method in {"sap_admm", "sap_admm_l1"}:
        solver = sap_admm_image if method == "sap_admm" else sap_admm_l1_image
        restored, _, info = solver(observation, operators, config["admm"])
    elif method in {"sdcam_l1", "sdcam_l2"}:
        loss = "l1" if method == "sdcam_l1" else "l2"
        parameters = dict(config["sdcam"])
        parameters["lambda_half"] = parameters[f"lambda_half_{loss}"]
        x, _, info = sdcam(observation, operators, parameters, loss=loss)
        restored = np.clip(x, 0, 1).reshape(observation.shape, order="F")
    else:
        raise ValueError(f"Unknown image method: {method}")
    return restored, info, time.perf_counter()-start


def load_clean_images(digits) -> np.ndarray:
    path = Path(__file__).resolve().parent / "inputs" / "mnist_clean_images.npz"
    with np.load(path, allow_pickle=False) as data:
        indices = [list(data["digits"]).index(digit) for digit in digits]
        return data["images"][indices]


def generate_observation(clean: np.ndarray, config: dict, digit_index: int, trial_index: int):
    seed = config["rng_base"] + 1000 * (digit_index + 1) + trial_index + 1
    rng = np.random.RandomState(seed)
    rows, cols = clean.shape
    mask = rng.rand(rows * cols).reshape(rows, cols, order="F") < config["impulse_prob"]
    replacement = rng.rand(rows * cols).reshape(rows, cols, order="F")
    observation = clean.copy()
    observation[mask] = replacement[mask]
    observation += config["gaussian_std"] * rng.randn(rows * cols).reshape(rows, cols, order="F")
    return np.clip(observation, 0, 1), seed


def run(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_metadata(output, config, "MNIST total variation denoising")
    clean = load_clean_images(config["digits"])
    R, D = config["trials"], len(config["digits"])
    rows, cols = clean.shape[1:]
    shape = (R, D, len(METHODS))
    data = {key: np.full(shape, np.nan) for key in ("psnr", "ssim", "gmsd", "mse", "seconds", "iterations")}
    data.update(
        clean=clean, observations=np.zeros((R, D, rows, cols)),
        recovered=np.zeros((*shape, rows, cols)), seeds=np.zeros((R, D), dtype=int),
        nu_final=np.full(shape, np.nan), nu_floor=np.full(shape, np.nan),
        nu_floor_iteration=np.zeros(shape, dtype=int),
        nu_iterations_after_floor=np.zeros(shape, dtype=int),
        nu_applicable=np.zeros(shape, dtype=bool),
        nu_post_floor_satisfied=np.zeros(shape, dtype=bool), completed=np.zeros(shape, dtype=bool),
        digits=np.asarray(config["digits"]), methods=np.asarray(METHODS),
    )
    records, diagnostics = [], []
    for d, digit in enumerate(config["digits"]):
        operators = image_operators(rows, cols)
        for r in range(R):
            observation, seed = generate_observation(clean[d], config, d, r)
            data["observations"][r, d], data["seeds"][r, d] = observation, seed
            for k, method in enumerate(METHODS):
                restored, info, seconds = solve(observation, method, config, operators)
                idx = (r, d, k)
                values = {
                    "psnr": psnr(restored, clean[d]),
                    "ssim": compute_ssim_value(restored, clean[d], config["ssim_mode"]),
                    "gmsd": compute_gmsd_value(restored, clean[d], config["gmsd_boundary"]),
                    "mse": float(np.mean((restored - clean[d]) ** 2)),
                    "seconds": seconds, "iterations": info["iter"],
                    "nu_applicable": info["nu_applicable"],
                    "nu_final": info.get("nu_final", np.nan), "nu_floor": info.get("nu_floor", np.nan),
                    "nu_floor_iteration": info.get("nu_floor_iteration", 0),
                    "nu_iterations_after_floor": info.get("nu_iterations_after_floor", 0),
                    "nu_post_floor_satisfied": info.get("nu_post_floor_satisfied", False),
                }
                for key, value in values.items():
                    data[key][idx] = value
                data["recovered"][idx], data["completed"][idx] = restored, True
                records.append({
                    "Trial": r + 1, "Digit": digit, "Method": method, "PSNR": values["psnr"],
                    "SSIM": values["ssim"], "GMSD": values["gmsd"], "MSE": values["mse"],
                    "Time_s": seconds, "Iterations": info["iter"],
                    "NuApplicable": info["nu_applicable"],
                    "NuFinal": info.get("nu_final", ""), "NuFloor": info.get("nu_floor", ""),
                    "NuFloorIteration": info.get("nu_floor_iteration", ""),
                    "NuIterationsAfterFloor": info.get("nu_iterations_after_floor", ""),
                    "NuPostFloorSatisfied": info.get("nu_post_floor_satisfied", ""),
                    "StopReason": info["stop_reason"],
                })
                diagnostics.append({"Trial": r+1, "Digit": digit, "Method": method, "solver": info})
                print(f"digit={digit} trial={r+1}/{R} {method}: PSNR={values['psnr']:.3f} SSIM={values['ssim']:.4f} GMSD={values['gmsd']:.4f} iter={info['iter']} time={seconds:.3f}s", flush=True)
            save_npz(output / "results.npz", **data)
            write_csv(output / "trial_metrics.csv", records)
            save_diagnostics(output, diagnostics)
    summary = []
    for d, digit in enumerate(config["digits"]):
        for k, method in enumerate(METHODS):
            summary.append({
                "Digit": digit, "Method": method, "Trials": R,
                "MeanPSNR": float(data["psnr"][:, d, k].mean()),
                "MeanSSIM": float(data["ssim"][:, d, k].mean()),
                "MeanGMSD": float(data["gmsd"][:, d, k].mean()),
                "MeanMSE": float(data["mse"][:, d, k].mean()),
                "MeanTime_s": float(data["seconds"][:, d, k].mean()),
                "MeanIterations": float(data["iterations"][:, d, k].mean()),
            })
    write_csv(output / "summary.csv", summary)
    table = []
    for d,digit in enumerate(config["digits"]):
        for metric,a,b in (("P/T","psnr","seconds"),("S/G","ssim","gmsd")):
            table.append({"Digit": digit, "Metric": metric, **{
                method: f"{data[a][:,d,k].mean():.6g}/{data[b][:,d,k].mean():.6g}"
                for k,method in enumerate(METHODS)}})
    write_csv(output / "paper_quality_time_table.csv", table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results" / "mnist")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--digits", type=int, nargs="+")
    args = parser.parse_args()
    config = load_config("mnist.json")
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be positive")
        config["trials"] = args.trials
    if args.digits is not None:
        if len(set(args.digits)) != len(args.digits) or any(d not in range(10) for d in args.digits):
            parser.error("--digits must be unique integers from 0 to 9")
        config["digits"] = args.digits
    run(config, args.out)


if __name__ == "__main__":
    main()
