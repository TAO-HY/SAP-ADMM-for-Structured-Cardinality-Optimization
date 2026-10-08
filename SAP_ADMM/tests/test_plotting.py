"""Check paper figures against saved measurements without rerunning solvers."""
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from experiments import plot_general_a, plot_mnist, plot_signal

SIGNAL_ORDER = ("sap_admm", "sap_admm_halpern", "sdcam_l1",
                "sap_admm_l1", "sdcam_l2", "padmm_l0")
IMAGE_ORDER = ("sap_admm", "sdcam_l1", "sap_admm_l1", "sdcam_l2")
ALIASES = {"sap_admm": "acc_capl1", "sap_admm_halpern": "capl1",
           "sap_admm_l1": "l1", "padmm_l0": "l0",
           "sdcam_l1": "sdcam_l1", "sdcam_l2": "sdcam_l2"}


@pytest.fixture
def figures(monkeypatch):
    captured = {}

    def capture(fig, folder, basename, **kwargs):
        captured[basename] = fig

    for module in (plot_signal, plot_mnist):
        monkeypatch.setattr(module, "export", capture)
    monkeypatch.setattr(plot_general_a, "save_figure", capture)
    yield captured
    for fig in captured.values():
        plt.close(fig)


@pytest.mark.parametrize("trials", [1, 4])
@pytest.mark.parametrize("use_aliases", [False, True])
def test_signal_plot_preserves_means_categories_and_recovery(tmp_path, figures, trials, use_aliases):
    # Store a different method order from the display order.
    methods = ("sdcam_l2", "sap_admm_halpern", "sap_admm",
               "sap_admm_l1", "padmm_l0", "sdcam_l1")
    saved_methods = np.asarray([ALIASES[m] for m in methods], dtype="S") if use_aliases else np.asarray(methods)
    base = np.array([[1., .9999999, .9, .8, .7, 0.],
                     [.9, .8, .9999999, 1., .7, .85]])
    f1 = np.stack([np.roll(base, k, axis=1) for k in range(trials)])
    seconds = np.arange(1, trials * 2 * 6 + 1, dtype=float).reshape(trials, 2, 6)
    seconds *= np.array([1., .01, .001, .02, .1, .5])[None, None, :]
    probabilities = np.array([0., .2])
    np.savez(tmp_path / "results.npz", probabilities=probabilities, methods=saved_methods,
             f1=f1, seconds=seconds, completed=np.ones_like(f1, dtype=bool))
    truth = np.linspace(0, 1, 1000)
    observations = np.stack([truth + .1, truth + .2])
    recovered = np.stack([np.stack([truth + .01 * (k + j) for k in range(6)]) for j in range(2)])
    np.savez(tmp_path / "recovery_example.npz", truth=truth, observations=observations,
             recovered=recovered, probabilities=probabilities, methods=saved_methods)
    files = sorted(tmp_path.glob("*.npz"))
    before = [hashlib.sha256(path.read_bytes()).digest() for path in files]
    plot_signal.plot(tmp_path)

    mean_f1, mean_time = f1.mean(0), seconds.mean(0)
    metric = figures["F1_Time_Efficiency"]
    for ax, expected in zip(metric.axes, (mean_f1, mean_time, mean_f1 / mean_time)):
        assert len(ax.lines) == 6
        for method, line in zip(SIGNAL_ORDER, ax.lines):
            k = methods.index(method)
            np.testing.assert_array_equal(line.get_xdata(), probabilities)
            np.testing.assert_allclose(line.get_ydata(), expected[:, k])
    assert len(metric.axes[0].collections) == 6
    use_log = mean_time.max() / mean_time.min() > 100
    assert metric.axes[1].get_yscale() == ("log" if use_log else "linear")
    assert len(metric.legends) == 1 and metric.legends[0].get_frame_on()

    distribution = figures["F1_Distribution"]
    assert len(distribution.axes) == 4
    selected = ("sap_admm", "sdcam_l1", "sdcam_l2", "padmm_l0")
    for method, ax in zip(selected, distribution.axes):
        values = f1[:, :, methods.index(method)]
        tol = 1e-6
        expected = np.stack([(np.abs(values - 1) < tol).sum(0),
                             ((values >= .9-tol) & (values < 1-tol)).sum(0),
                             ((values >= .8-tol) & (values < .9-tol)).sum(0),
                             (values < .8-tol).sum(0)])
        heights = np.array([bar.get_height() for bar in ax.patches]).reshape(4, 2)
        np.testing.assert_array_equal(heights, expected)
        np.testing.assert_array_equal(heights.sum(0), np.full(2, trials))
        annotations = [text.get_text() for text in ax.texts if text.get_text().endswith("%")]
        assert annotations == [f"{100 * val / trials:.0f}%" for val in expected.ravel() if val > 0]

    recovery = figures["Figure2_Recovery_Comparison_FiveModels_acc_capl1_python"]
    assert len(recovery.axes) == 12
    panels = (None, "sap_admm", "sdcam_l1", "sap_admm_l1", "sdcam_l2", "padmm_l0")
    for j in range(2):
        for panel, method in enumerate(panels):
            ax = recovery.axes[j * 6 + panel]
            if method is None:
                assert len(ax.lines) == 1
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), observations[j])
            else:
                assert len(ax.lines) == 2
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), truth)
                np.testing.assert_array_equal(ax.lines[1].get_ydata(), recovered[j, methods.index(method)])
    assert before == [hashlib.sha256(path.read_bytes()).digest() for path in files]


@pytest.mark.parametrize("count", [1, 10])
@pytest.mark.parametrize("use_aliases", [False, True])
def test_mnist_square_tiles_preserve_images(tmp_path, figures, count, use_aliases):
    methods = ("sdcam_l2", "sap_admm_l1", "sap_admm", "sdcam_l1")
    saved_methods = np.asarray([ALIASES[m] for m in methods], dtype="S") if use_aliases else np.asarray(methods)
    clean = np.linspace(0, 1, count * 28 * 28).reshape(count, 28, 28)
    observations = np.stack((clean * .8, clean * .6))
    first = np.stack([clean * (.9 - k * .1) for k in range(4)], axis=1)
    recovered = np.stack((first, first * .5))
    path = tmp_path / "results.npz"
    np.savez(path, digits=np.arange(count), clean=clean, observations=observations,
             recovered=recovered, completed=np.ones((2, count, 4), dtype=bool), methods=saved_methods)
    before = hashlib.sha256(path.read_bytes()).digest()
    plot_mnist.plot(tmp_path)
    fig = figures["mnist_four_model_restoration_python"]
    assert len(fig.axes) == 6 * count
    arrays = (clean, observations[0]) + tuple(first[:, methods.index(m)] for m in IMAGE_ORDER)
    for col in range(count):
        for row, images in enumerate(arrays):
            ax = fig.axes[col * 6 + row]
            np.testing.assert_array_equal(ax.images[0].get_array(), images[col])
            bounds = ax.get_position()
            assert bounds.width * fig.get_figwidth() == pytest.approx(bounds.height * fig.get_figheight())
            assert ax.get_title() == ""
            if col == 0:
                assert ax.yaxis.label.get_rotation() == 90
                assert ax.get_ylabel()
    assert before == hashlib.sha256(path.read_bytes()).digest()


def test_general_a_panel_order_and_means(tmp_path, figures):
    x = np.array([200, 300, 400, 500, 600])
    arrays = {"f1": np.ones((2, 5)), "iterations": np.full((2, 5), 8239.95),
              "mse": np.array([[.003] * 5, [.005] * 5])}
    np.savez(tmp_path / "results.npz", Nmax_list=x, **arrays, completed=np.ones((2, 5), dtype=bool))
    plot_general_a.plot(tmp_path)
    for ax, key in zip(figures["general_A_sensitivity"].axes, ("f1", "iterations", "mse")):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), arrays[key].mean(0))
        lower, upper = ax.get_ylim()
        assert lower < arrays[key].mean() < upper


@pytest.mark.parametrize("module", [plot_signal, plot_mnist, plot_general_a])
def test_incomplete_results_are_not_plotted(tmp_path, figures, module):
    np.savez(tmp_path / "results.npz", completed=np.array([False]))
    with pytest.raises(ValueError, match="incomplete"):
        module.plot(tmp_path)
    assert not figures


@pytest.mark.parametrize("module", [plot_signal, plot_mnist])
def test_export_writes_png_and_pdf_at_600_dpi(tmp_path, module):
    # Real export on a small canvas, using the production export function.
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    module.export(fig, tmp_path, "export_check")
    with Image.open(tmp_path / "export_check.png") as image:
        assert image.size[0] > 0 and image.size[1] > 0
        assert image.info["dpi"] == pytest.approx((600, 600), abs=.1)
    assert (tmp_path / "export_check.pdf").read_bytes().startswith(b"%PDF-")
    assert not plt.fignum_exists(fig.number)
