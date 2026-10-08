"""Plot saved MNIST results; no experiment computation is performed.

Windows command window, from the experiments folder:
    python plot_mnist.py
    python plot_mnist.py --data "E:/path/to/mnist/results"

Reads results.npz and saves PNG/PDF under the data folder's figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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


MODEL_ORDER = ("acc_capl1", "sdcam_l1", "l1", "sdcam_l2")
FIG_WIDTH = 10.0
GAP_X = GAP_Y = 0.003
MARGIN_LEFT, MARGIN_RIGHT = 0.055, 0.015
MARGIN_TOP, MARGIN_BOTTOM = 0.025, 0.020
LABEL_FONTSIZE = 12


def plot_results(data_dir, out):
    style()
    with np.load(Path(data_dir) / "results.npz", allow_pickle=False) as data:
        if not data["completed"].all():
            raise ValueError("Experiment is incomplete; finish/resume computation first")
        digits, methods = data["digits"], method_names(data["methods"])
        clean, noisy, restored = data["clean"], data["observations"][0], data["recovered"][0]
    num_digits, num_rows = len(digits), 6
    row_labels = ["Original", "Noisy"] + [PAPER_LABELS[m] for m in MODEL_ORDER]
    usable_width = 1 - MARGIN_LEFT - MARGIN_RIGHT - (num_digits - 1) * GAP_X
    usable_height = 1 - MARGIN_TOP - MARGIN_BOTTOM - (num_rows - 1) * GAP_Y
    if num_digits == 0 or usable_width <= 0 or usable_height <= 0:
        raise ValueError("The image count and figure margins leave no usable area")
    subplot_width = usable_width / num_digits
    subplot_height = usable_height / num_rows
    fig_height = FIG_WIDTH * subplot_width / subplot_height
    fig = plt.figure(figsize=(FIG_WIDTH, fig_height), facecolor="white")
    for digit_index in range(num_digits):
        images = [clean[digit_index], noisy[digit_index]] + [
            restored[digit_index, methods.index(method)] for method in MODEL_ORDER
        ]
        left = MARGIN_LEFT + digit_index * (subplot_width + GAP_X)
        for row_index, image in enumerate(images):
            bottom = (1 - MARGIN_TOP - (row_index + 1) * subplot_height
                      - row_index * GAP_Y)
            ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
            ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if digit_index == 0:
                ax.set_ylabel(row_labels[row_index], fontsize=LABEL_FONTSIZE,
                              fontweight="bold", fontstyle="normal",
                              rotation=90, labelpad=10)
    export(fig, out, "mnist_four_model_restoration_python", pad_inches=0.01)
    print(f"Saved paper restoration figure to {out}")


def plot(data_dir):
    """Entry point used by the experiment package's plot-only runners."""
    data_dir = Path(data_dir)
    plot_results(data_dir, data_dir / "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "mnist")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plot_results(args.data, args.out or args.data / "figures")


if __name__ == "__main__":
    main()
