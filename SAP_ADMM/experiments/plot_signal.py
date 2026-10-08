"""Plot saved signal results; no experiment computation is performed.

Windows command window, from the experiments folder:
    python plot_signal.py
    python plot_signal.py --data "E:/path/to/signal/results"

Reads results.npz and recovery_example.npz (or recovery.npz).
Outputs PNG, PDF and plot-data CSV files under the data folder's figures/.
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# Paths are relative to this script, independent of the command-window directory.
ROOT = Path(__file__).resolve().parents[1]
METHOD_ALIASES = {
    "sap_admm": "acc_capl1", "sap_admm_halpern": "capl1",
    "sap_admm_l1": "l1", "padmm_l0": "l0",
    "acc_capl1": "acc_capl1", "capl1": "capl1", "l1": "l1", "l0": "l0",
    "sdcam_l1": "sdcam_l1", "sdcam_l2": "sdcam_l2",
}


def method_names(values):
    names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]
    return [METHOD_ALIASES.get(name, name) for name in names]


def sample_std(x, axis=0):
    x = np.asarray(x)
    return np.std(x, axis=axis, ddof=1 if x.shape[axis] > 1 else 0)


def write_csv(path, rows):
    import csv
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


PAPER_LABELS = {
    "acc_capl1": r"$\mathbf{SAP}$-$\mathbf{ADMM}$",
    "capl1": r"$\mathbf{SAP}$-$\mathbf{ADMM}^{\mathbf{H}}$",
    "sdcam_l1": r"$\mathbf{SDCAM}^{\mathbf{1}}$",
    "l1": r"$\mathbf{SAP}$-$\mathbf{ADMM}^{\mathbf{1}}$",
    "sdcam_l2": r"$\mathbf{SDCAM}^{\mathbf{2}}$",
    "l0": r"$\mathbf{pADMM}$",
}
STYLES = {
    "acc_capl1": ("#0072B2", "-", "o"),
    "capl1": ("#4D4D4D", "--", "X"),
    "sdcam_l1": ("#DDAA33", "-.", "v"),
    "l1": ("#D55E00", "-.", "s"),
    "sdcam_l2": ("#7E2F8E", "-", "D"),
    "l0": ("#00806F", ":", "^"),
}
FILL_COLORS = {
    "acc_capl1": "#B3D9FF", "capl1": "#D9D9D9",
    "sdcam_l1": "#F6E3A0", "l1": "#FFD9B3",
    "sdcam_l2": "#CCB3D9", "l0": "#B3E6D9",
}


def style():
    """Use the original unified Times font, with an available serif fallback."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((name for name in ("Times New Roman", "Nimbus Roman",
                                    "Liberation Serif", "STIXGeneral")
                   if name in available), "DejaVu Serif")
    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": "STIXGeneral",
        "mathtext.fallback": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def export(fig, out, basename, *, pad_inches=0.1):
    """Keep the package's shared export API and the original 600 dpi output."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        with BytesIO() as buffer:
            fig.savefig(buffer, format=suffix, dpi=600,
                        bbox_inches="tight", pad_inches=pad_inches,
                        facecolor="white")
            (out / f"{basename}.{suffix}").write_bytes(buffer.getvalue())
    plt.close(fig)


def panel_label(ax, text, fontsize=30):
    ax.text(0.5, -0.25, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", clip_on=False)


def _probability_labels(probabilities):
    return ["0" if p == 0 else f"{p:.2f}" for p in probabilities]


def _format_comparison_axes(ax, probabilities):
    ax.set_xlim(-0.01, 0.21)
    ax.set_xticks(probabilities)
    ax.set_xticklabels(_probability_labels(probabilities))
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.tick_params(axis="both", labelsize=12, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _plot_comparison(probabilities, methods, f1, seconds, out):
    means, deviations = f1.mean(axis=0), sample_std(f1, axis=0)
    timing = seconds.mean(axis=0)
    efficiency = means / np.maximum(timing, np.finfo(float).eps)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    use_log_time = timing.max() / max(timing.min(), np.finfo(float).eps) > 100
    for method in STYLES:
        k = methods.index(method)
        color, linestyle, marker = STYLES[method]
        kwargs = dict(color=color, linestyle=linestyle, marker=marker,
                      linewidth=2.2, markersize=7, markerfacecolor="white",
                      markeredgecolor=color, label=PAPER_LABELS[method])
        axes[0].fill_between(
            probabilities, np.maximum(means[:, k] - deviations[:, k], 0),
            means[:, k] + deviations[:, k], color=FILL_COLORS[method],
            alpha=0.18 if method == "capl1" else 0.22, linewidth=0,
        )
        axes[0].plot(probabilities, means[:, k], **kwargs)
        time_plot = axes[1].semilogy if use_log_time else axes[1].plot
        time_plot(probabilities, timing[:, k], **kwargs)
        axes[2].plot(probabilities, efficiency[:, k], **kwargs)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks(np.arange(0, 1.11, 0.1))
    for i, (ax, ylabel) in enumerate(zip(
            axes, (r"$\mathbf{F_1}$ Score", "Time", r"$\mathbf{F_1}$ Score / Time"))):
        ax.set_xlabel(r"Impulse Noise Probability $\mathbf{\pi}$",
                      fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        _format_comparison_axes(ax, probabilities)
        panel_label(ax, f"({chr(97 + i)})")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5171, 1.05), ncol=3,
               prop={"size": 13, "weight": "bold"}, frameon=True,
               edgecolor="black", handlelength=2.5, columnspacing=1.6,
               handletextpad=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=1.15)
    export(fig, out, "F1_Time_Efficiency")
    rows = [{"ImpulseProb": float(p), "Method": str(method),
             "MeanF1": means[j, k], "StdF1": deviations[j, k],
             "MeanTime_s": timing[j, k], "Efficiency": efficiency[j, k]}
            for j, p in enumerate(probabilities) for k, method in enumerate(methods)]
    write_csv(out / "F1_Time_Efficiency_plot_data.csv", rows)


def _plot_distribution(probabilities, methods, f1, out):
    distribution_methods = ("acc_capl1", "sdcam_l1", "sdcam_l2", "l0")
    colors = np.array([[0.121, 0.343, 0.549], [0.196, 0.604, 0.569],
                       [0.894, 0.624, 0.235], [0.706, 0.247, 0.196]])
    labels = (r"$\mathbf{F_1 = 1}$", r"$\mathbf{0.9 \leq F_1 < 1}$",
              r"$\mathbf{0.8 \leq F_1 < 0.9}$", r"$\mathbf{F_1 < 0.8}$")
    csv_labels = (r"$F_1=1$", r"$0.9\leq F_1<1$", r"$0.8\leq F_1<0.9$", r"$F_1<0.8$")
    fig = plt.figure(figsize=(10.0, 3.8), facecolor="white")
    left, right, bottom, top, gap = 0.055, 0.015, 0.22, 0.22, 0.025
    width = (1 - left - right - 3 * gap) / 4
    height = 1 - bottom - top
    x = 1.0 + np.arange(len(probabilities)) * 0.72
    trials = f1.shape[0]
    count_rows = []
    tol = 1e-6
    for panel, method in enumerate(distribution_methods):
        ax = fig.add_axes([left + panel * (width + gap), bottom, width, height])
        values = f1[:, :, methods.index(method)]
        counts = np.stack((np.sum(np.abs(values - 1) < tol, axis=0),
                           np.sum((values >= 0.9 - tol) & (values < 1 - tol), axis=0),
                           np.sum((values >= 0.8 - tol) & (values < 0.9 - tol), axis=0),
                           np.sum(values < 0.8 - tol, axis=0)), axis=1)
        base = np.zeros(len(probabilities))
        for k, color in enumerate(colors):
            values = counts[:, k]
            ax.bar(x, values, width=0.54, bottom=base, color=color,
                   edgecolor=(0.18, 0.18, 0.18), linewidth=0.55)
            for j, val in enumerate(values):
                if val > 0:
                    small = val < max(2, 0.08 * trials)
                    ax.text(x[j], base[j] + val / 2, f"{100 * val / trials:.0f}%",
                            ha="center", va="center", fontsize=6.6 if small else 7.5,
                            fontweight="bold", color="black" if small else "white")
            base += values
            count_rows.extend({"ImpulseProb": float(p), "Method": method,
                               "Category": csv_labels[k], "Count": int(counts[j, k]),
                               "Trials": int(trials)}
                              for j, p in enumerate(probabilities))
        ax.set_xlabel(r"Impulse Noise Probability $\mathbf{\pi}$",
                      fontsize=11, fontweight="bold")
        if panel == 0:
            ax.set_ylabel("Number of Experiments", fontsize=10.2, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(_probability_labels(probabilities), fontsize=9.4)
        ax.set_xlim(x[0] - 0.34, x[-1] + 0.34)
        ax.set_ylim(0, trials * 1.06)
        ax.grid(True, axis="y", linestyle="--", alpha=0.22)
        ax.tick_params(axis="y", labelsize=9.4, width=0.95)
        ax.tick_params(axis="x", width=0.95)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.95)
        panel_label(ax, PAPER_LABELS[method], fontsize=16)
    handles = [Patch(facecolor=color, edgecolor=(0.18, 0.18, 0.18), linewidth=0.55)
               for color in colors]
    legend = fig.legend(handles, labels, loc="lower left",
                        bbox_to_anchor=(0.25, 0.85, 0.5, 0.1), mode="expand", ncol=4,
                        frameon=True, fontsize=14, borderaxespad=0,
                        handlelength=1.2, handletextpad=0.35, columnspacing=0.75)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_facecolor("white")
    export(fig, out, "F1_Distribution", pad_inches=0.03)
    write_csv(out / "F1_Distribution_counts.csv", count_rows)


def _plot_recovery(recovery_path, out):
    with np.load(recovery_path, allow_pickle=False) as data:
        clean = data["truth"] if "truth" in data else data["clean"]
        noisy, recovered = data["observations"], data["recovered"]
        methods, probabilities = method_names(data["methods"]), data["probabilities"]
    # Each noise setting occupies two rows: noisy + the five selected methods.
    panels = (None, "acc_capl1", "sdcam_l1", "l1", "sdcam_l2", "l0")
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, sharey=True)
    x = np.arange(1, len(clean) + 1)
    for probability_index, probability in enumerate(probabilities):
        for panel, method in enumerate(panels):
            row, col = 2 * probability_index + panel // 3, panel % 3
            ax = axes[row, col]
            if method is None:
                ax.plot(x, noisy[probability_index], color="#808080", linewidth=0.9)
                title = f"Noisy (Prob={probability:g})"
            else:
                ax.plot(x, clean, color="#D62728", linewidth=1.0, label="Original")
                ax.plot(x, recovered[probability_index, methods.index(method)],
                        color="#0072B2", linewidth=1.4, label="Recovery")
                ax.legend(loc="lower left", prop={"size": 7.2, "weight": "bold"},
                          frameon=True)
                title = f"{PAPER_LABELS[method]} (Prob={probability:g})"
            ax.set_title(title, fontsize=10.5, fontweight="bold")
            ax.set_xlim(0, 1000)
            ax.set_ylim(-15, 15)
            ax.set_xticks([0, 250, 500, 750, 1000])
            ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
            ax.grid(True, linestyle="--", alpha=0.30)
            ax.tick_params(axis="both", labelsize=8.5, direction="in")
            if row == 3:
                ax.set_xlabel("Index", fontsize=9.5, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Amplitude", fontsize=9.5, fontweight="bold")
    fig.tight_layout(pad=1.1)
    export(fig, out, "Figure2_Recovery_Comparison_FiveModels_acc_capl1_python")


def plot_results(data_dir, out):
    style()
    data_dir, out = Path(data_dir), Path(out)
    with np.load(data_dir / "results.npz", allow_pickle=False) as data:
        if not data["completed"].all():
            raise ValueError("Experiment is incomplete; finish/resume computation before making paper figures")
        probabilities, methods = data["probabilities"], method_names(data["methods"])
        f1, seconds = data["f1"], data["seconds"]
    _plot_comparison(probabilities, methods, f1, seconds, out)
    _plot_distribution(probabilities, methods, f1, out)
    recovery_path = data_dir / "recovery_example.npz"
    if not recovery_path.exists():
        recovery_path = data_dir / "recovery.npz"
    if recovery_path.exists():
        _plot_recovery(recovery_path, out)
    print(f"Saved paper figures to {out}")


def plot(data_dir):
    """Entry point used by the experiment package's plot-only runners."""
    data_dir = Path(data_dir)
    plot_results(data_dir, data_dir / "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "signal")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plot_results(args.data, args.out or args.data / "figures")


if __name__ == "__main__":
    main()
