"""Draw the original SIAM general-A sensitivity figure from saved results."""
from __future__ import annotations
import argparse
from pathlib import Path
# Resolve imports when this file is run directly in Spyder.
if __package__ in (None, ""):
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    __package__ = "experiments"

import numpy as np
import matplotlib.pyplot as plt
from ._shared import REPOSITORY_ROOT, configure_plotting, save_figure


def plot(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    out = data_dir / "figures"
    with np.load(data_dir / "results.npz", allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    if not data["completed"].all():
        raise ValueError("Experiment checkpoint is incomplete; finish computation before plotting")
    configure_plotting()
    Nmax = data["Nmax_list"]
    fig,axes = plt.subplots(1,3,figsize=(10.8,3.8))
    for ax,key,label in zip(axes,("f1","iterations","mse"),(r"Average $F_1$","Average iterations","Average MSE")):
        mean = data[key].mean(axis=0)
        ax.plot(Nmax,mean,"o-",color="#0072B2",lw=1.6,markersize=5)
        ax.set(xlabel=r"$N_{\max}$",ylabel=label,xticks=Nmax)
        ax.grid(alpha=.2,ls=":")
        if np.ptp(mean)<1e-12:
            pad=max(abs(float(mean[0]))*.01,1e-4)
            ax.set_ylim(mean[0]-pad,mean[0]+pad)
    fig.tight_layout()
    save_figure(fig,out,"general_A_sensitivity")
    print(f"General-A figures saved to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "results" / "general_a")
    args = parser.parse_args()
    plot(args.data)


if __name__ == "__main__":
    main()
