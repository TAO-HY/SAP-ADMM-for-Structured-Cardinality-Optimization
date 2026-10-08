"""Shared experiment helpers."""

from __future__ import annotations

import io
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy
import matplotlib
from threadpoolctl import threadpool_info

from sap_admm.utils import save_npz, write_csv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def save_metadata(folder: Path, config: dict, experiment: str) -> None:
    payload = {
        "experiment": experiment,
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "blas": threadpool_info(),
        "timer": "time.perf_counter (elapsed wall time)",
        "scope": "SAP-ADMM, SAP-ADMM^H, SAP-ADMM^1, SDCAM^1, SDCAM^2 and signal pADMM",
        "random_generator": "NumPy RandomState (MT19937)",
    }
    (folder / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def save_diagnostics(folder: Path, records: list) -> None:
    """Save solver-specific stops and SDCAM stages without losing applicability."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (folder / "solver_diagnostics.json").write_text(
        json.dumps(records, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def save_figure(fig, folder: Path, basename: str, *, pad_inches: float = 0.1) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix, bbox_inches="tight",
                    pad_inches=pad_inches, facecolor="white", **kwargs)
        (folder / f"{basename}.{suffix}").write_bytes(buffer.getvalue())
    plt.close(fig)


def configure_plotting() -> None:
    """Match SIAM_Experiments_Python's original Times/STIX figure defaults."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "STIXGeneral"
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "stix", "font.size": 11,
                         "axes.labelsize": 12, "axes.linewidth": 0.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300, "axes.unicode_minus": False})


__all__ = [
    "REPOSITORY_ROOT", "configure_plotting", "load_config", "save_figure",
    "save_metadata", "save_npz", "write_csv",
]
