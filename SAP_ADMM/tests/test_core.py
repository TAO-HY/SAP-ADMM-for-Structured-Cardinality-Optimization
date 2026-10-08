import numpy as np

from sap_admm import sap_admm, sap_admm_generalA, sap_admm_halpern, sap_admm_image
from sap_admm.utils import difference_matrix, image_operators
from experiments.run_signal import generate_input


def test_image_operator_adjoint_identity():
    rng = np.random.RandomState(7)
    D, DT = image_operators(5, 4)
    x = rng.randn(20)
    y = rng.randn(40)
    assert np.allclose(np.dot(D(x), y), np.dot(x, DT(y)), atol=1e-12)


def test_signal_variants_enforce_post_floor_updates():
    D = difference_matrix(12)
    parameters = {
        "max_iter": 10000, "tol_stop": 1.0, "nu_switch_updates": 10,
        "nu_decay_early": 0.9, "nu_decay_late": 0.8,
        "min_iterations_after_floor": 50,
    }
    for solver in (sap_admm, sap_admm_halpern):
        _, _, info = solver(np.zeros(12), D, parameters)
        assert info["nu_post_floor_satisfied"]
        assert info["nu_iterations_after_floor"] == 50
        assert info["iter"] == info["nu_floor_iteration"] + 50


def test_image_solver_enforces_post_floor_updates():
    _, _, info = sap_admm_image(
        np.zeros((4, 4)), parameters={"max_iter": 10000, "stop_tol": 1.0}
    )
    assert info["nu_post_floor_satisfied"]
    assert info["nu_iterations_after_floor"] == 50


def test_general_a_solver_enforces_post_floor_updates():
    A = np.zeros((3, 4))
    D = difference_matrix(4)
    _, _, info = sap_admm_generalA(
        np.zeros(3), A, D, 4, {"max_iter": 10000, "tol_stop": 1.0}
    )
    assert info["nu_post_floor_satisfied"]
    assert info["nu_iterations_after_floor"] == 50


def test_signal_input_generation_is_deterministic():
    config = {
        "n": 100, "gaussian_std": 0.5, "impulse_scale": 10.0,
        "probabilities": [0.0, 0.05],
    }
    first = generate_input(config, 1, 2)
    second = generate_input(config, 1, 2)
    assert first[2] == second[2]
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_iteration_cap_is_independent_of_post_floor_condition():
    D = difference_matrix(4)
    reports = []
    for solver in (sap_admm, sap_admm_halpern):
        reports.append(solver(np.zeros(4), D, {"max_iter": 1})[2])
    reports.append(sap_admm_image(np.zeros((2, 2)), parameters={"max_iter": 1})[2])
    reports.append(sap_admm_generalA(np.zeros(3), np.zeros((3, 4)), D, 4,
                                   {"max_iter": 1})[2])
    for info in reports:
        assert info["iter"] == 1
        assert info["stop_reason"] == "max_iter"
        assert not info["nu_post_floor_satisfied"]


def test_halpern_diagnostics_report_actual_parameters():
    _, _, info = sap_admm_halpern(np.zeros(4), difference_matrix(4),
                                 {"max_iter": 1, "alpha": 15, "t": 1.5})
    assert info["alpha"] == 2
    assert info["t"] == 1


def test_general_a_continuation_boundary_uses_zero_based_counter():
    from sap_admm.general_a import update_nu_by_bar_count
    assert update_nu_by_bar_count(1.0, 3999, 1e-9) == 0.9995
    assert update_nu_by_bar_count(1.0, 4000, 1e-9) == 0.95


def test_post_floor_counter_excludes_first_floor_update():
    from sap_admm.utils import NuMonitor
    monitor = NuMonitor(2.0, 1.0)
    monitor.record(1.0, 7)
    monitor.record(1.0, 56)
    assert monitor.after_floor == 49
    assert not monitor.can_stop()
    monitor.record(1.0, 57)
    assert monitor.can_stop()
