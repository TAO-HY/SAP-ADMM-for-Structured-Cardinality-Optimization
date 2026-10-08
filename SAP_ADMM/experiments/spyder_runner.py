"""Run experiment workflows from the top-level Spyder entry scripts."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

from ._shared import REPOSITORY_ROOT, load_config


def run_experiments(names, *, quick=False, mode="all", output_root=None):
    """Compute and/or plot selected experiments using the supplied paper configs.

    Relative output paths are resolved against the SAP_ADMM project directory.
    Quick mode changes only the number of trials to one per setting.
    """
    names = tuple(names)
    allowed = {"signal", "general_a", "mnist"}
    if not names or any(name not in allowed for name in names):
        raise ValueError("Choose signal, general_a, and/or mnist")
    if mode not in {"compute", "plot", "all"}:
        raise ValueError("MODE must be 'compute', 'plot', or 'all'")
    base = Path(output_root).expanduser() if output_root is not None else Path(
        "quick_results" if quick else "results"
    )
    if not base.is_absolute():
        base = REPOSITORY_ROOT / base
    base = base.resolve()
    print(f"Output directory: {base}", flush=True)
    for name in names:
        folder = base / name
        if mode in {"compute", "all"}:
            config = load_config(f"{name}.json")
            if quick:
                config["trials"] = 1
            print(f"\nRunning {name}: {config['trials']} trial(s) per setting", flush=True)
            import_module(f"experiments.run_{name}").run(config, folder)
        if mode in {"plot", "all"}:
            print(f"Plotting {name}", flush=True)
            import_module(f"experiments.plot_{name}").plot(folder)
        print(f"Finished {name}: {folder}", flush=True)
    print("\nAll requested experiments finished.", flush=True)
    return base
