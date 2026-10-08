"""Numerical and workflow checks for the paper's convex SAP-ADMM^1 model."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from sap_admm import sap_admm_l1, sap_admm_l1_image
from sap_admm.utils import difference_matrix
from experiments import run_mnist, run_signal
from experiments._shared import load_config


@pytest.mark.parametrize("image", [False, True])
def test_convex_solver_has_no_post_floor_wait(image):
    parameters = {"max_iter": 1, "min_iterations_after_floor": 500,
                  "nu0": -1, "nu_decay": -1, "max_fail": 0, "restart_iter": 0}
    if image:
        result, count, info = sap_admm_l1_image(np.zeros((3, 4)), parameters=parameters)
        assert count == 1
    else:
        result, jump, info = sap_admm_l1(np.zeros(12), difference_matrix(12), parameters)
        assert np.all(jump == 0)
    assert np.all(result == 0)
    assert info["iter"] == 1
    assert info["stop_reason"] == "converged"
    assert info["nu_applicable"] is False
    assert "nu_final" not in info
    assert not info["restarted"]


@pytest.mark.parametrize("image", [False, True])
def test_convex_stopping_profile_and_source_iteration_cap(image):
    observation = np.random.RandomState(31).randn(12)
    parameters = {"max_iter": 5, "stop_tol": 1e-20}
    if image:
        _, _, info = sap_admm_l1_image(observation.reshape(3, 4), parameters=parameters)
        assert info["iter"] == info["effective_max_iter"] == 5
        assert info["stopping_blocks"] == "x,y,p,q,eta,mu"
    else:
        _, _, info = sap_admm_l1(observation, difference_matrix(12), parameters)
        assert info["iter"] == info["effective_max_iter"] == 5
        assert info["stopping_blocks"] == "x,y,p,q,eta,mu"
    assert info["stop_reason"] == "max_iter"


@pytest.mark.parametrize("image", [False, True])
def test_convex_matches_saved_uploaded_source_outputs(image):
    folder = Path(__file__).parent / "fixtures"
    metadata = json.loads((folder / "l1_reference_outputs.json").read_text())
    with np.load(folder / "l1_reference_outputs.npz", allow_pickle=False) as data:
        if image:
            metadata["image_parameters"]["stopping_profile"] = "legacy_image"
            result, count, info = sap_admm_l1_image(data["image_input"], parameters=metadata["image_parameters"])
            np.testing.assert_allclose(result, data["image_output"], rtol=1e-12, atol=1e-12)
            assert count == metadata["image_updates"]
        else:
            observation = data["signal_input"]
            result, jump, info = sap_admm_l1(observation, difference_matrix(observation.size), metadata["signal_parameters"])
            np.testing.assert_allclose(result, data["signal_output"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(jump, data["signal_jump"], rtol=1e-12, atol=1e-12)
            assert info["iter"] == metadata["signal_updates"]


def test_signal_objective_matches_independent_linear_program():
    observation = np.array([0.0, 0.1, -0.1, 3.0, 2.9, 3.1, 0.0, 0.2])
    n = observation.size
    D = difference_matrix(n).toarray()
    penalty = 0.1
    # With lambda_p > lambda_l1*sqrt(n-1), minimizing over y gives y=Dx.
    # Thus the independent LP solves the same split convex objective at optimum.
    cost = np.r_[np.zeros(n), np.full(n, 1 / n), np.full(n - 1, penalty)]
    constraints = np.block([
        [np.eye(n), -np.eye(n), np.zeros((n, n - 1))],
        [-np.eye(n), -np.eye(n), np.zeros((n, n - 1))],
        [D, np.zeros((n - 1, n)), -np.eye(n - 1)],
        [-D, np.zeros((n - 1, n)), -np.eye(n - 1)],
    ])
    bounds = [(None, None)] * n + [(0, None)] * (2 * n - 1)
    exact = linprog(cost, A_ub=constraints, b_ub=np.r_[observation, -observation,
                    np.zeros(2 * (n - 1))], bounds=bounds, method="highs")
    assert exact.success
    x, y, info = sap_admm_l1(observation, difference_matrix(n),
                            {"lambda2_l1": penalty, "max_iter": 20000, "tol_stop": 1e-8})
    objective = np.abs(x - observation).sum() / n + info["lambda_p"] * np.linalg.norm(D @ x - y) + penalty * np.abs(y).sum()
    assert objective == pytest.approx(exact.fun, abs=5e-5)


def test_signal_workflow_saves_all_three_methods_and_applicability(tmp_path):
    config = load_config("signal.json")
    config.update(n=12, trials=1, probabilities=[0.0], max_iter=5)
    config["sdcam"]["max_inner"] = 5
    config["padmm"]["max_iter"] = 5
    run_signal.run(config, tmp_path)
    with np.load(tmp_path / "results.npz", allow_pickle=False) as data:
        assert list(data["methods"]) == list(run_signal.METHODS)
        assert data["completed"].shape == (1, 1, 6)
        assert data["completed"].all()
        np.testing.assert_array_equal(data["nu_applicable"][0, 0], [True, True, False, False, False, False])
        assert np.isnan(data["nu_final"][0, 0, 2])
        x, y, info = sap_admm_l1(data["observations"][0, 0], difference_matrix(12), run_signal.solver_parameters(config))
        np.testing.assert_allclose(data["recovered"][0, 0, 2], x)
        np.testing.assert_allclose(data["jumps"][0, 0, 2], y)
        assert data["iterations"][0, 0, 2] == info["iter"]
    with np.load(tmp_path / "recovery_example.npz", allow_pickle=False) as data:
        assert data["recovered"].shape == (2, 6, 12)


def test_mnist_workflow_saves_two_methods_on_shared_observation(tmp_path):
    config = load_config("mnist.json")
    config.update(trials=1, digits=[0])
    config["admm"]["max_iter"] = 5
    config["sdcam"]["max_inner"] = 5
    run_mnist.run(config, tmp_path)
    with np.load(tmp_path / "results.npz", allow_pickle=False) as data:
        assert list(data["methods"]) == list(run_mnist.METHODS)
        assert data["psnr"].shape == (1, 1, 4)
        assert data["recovered"].shape == (1, 1, 4, 28, 28)
        assert data["observations"].shape == (1, 1, 28, 28)
        assert data["completed"].all()
        np.testing.assert_array_equal(data["nu_applicable"][0, 0], [True, False, False, False])
        assert np.isnan(data["nu_final"][0, 0, 1])
        image, _, info = sap_admm_l1_image(data["observations"][0, 0], parameters=config["admm"])
        np.testing.assert_allclose(data["recovered"][0, 0, 1], image)
        assert data["iterations"][0, 0, 1] == info["iter"]
