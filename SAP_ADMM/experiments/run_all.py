"""Run, save, and plot all SAP-ADMM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

# Support running this file directly in Spyder or with python /path/to/file.py.
# Resolve imports from this checkout independently of the working directory.
if __package__ in (None, ""):
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    __package__ = "experiments"

from ._shared import REPOSITORY_ROOT, load_config
from .plot_general_a import plot as plot_general_a
from .plot_mnist import plot as plot_mnist
from .plot_signal import plot as plot_signal
from .run_general_a import run as run_general_a
from .run_mnist import run as run_mnist
from .run_signal import run as run_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compute", "plot", "all"), default="all")
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--quick", action="store_true", help="Use one trial per setting")
    args = parser.parse_args()
    paths = {name: args.out / name for name in ("signal", "mnist", "general_a")}
    if args.mode in ("compute", "all"):
        signal, mnist, general = (load_config(name) for name in ("signal.json", "mnist.json", "general_a.json"))
        if args.quick:
            for config in (signal, mnist, general):
                config["trials"] = 1
        run_signal(signal, paths["signal"])
        run_mnist(mnist, paths["mnist"])
        run_general_a(general, paths["general_a"])
    if args.mode in ("plot", "all"):
        plot_signal(paths["signal"])
        plot_mnist(paths["mnist"])
        plot_general_a(paths["general_a"])


if __name__ == "__main__":
    main()
