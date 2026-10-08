"""Open this file in Spyder and press F5 to run the general-A signal experiment.

The defaults reproduce the paper's trial counts and save both data and figures.
Set QUICK = True for one trial per setting; set MODE = "plot" to redraw saved data.
"""
from pathlib import Path
import sys

# These paths are based on this file; no package installation or cd is needed.
_PROJECT_ROOT = Path(__file__).resolve().parent / "SAP_ADMM"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from experiments.spyder_runner import run_experiments

# %% Settings (edit these before pressing F5)
QUICK = False          # True: one trial per setting; False: paper trial counts.
MODE = "all"           # "all": compute + plot; "compute": data only; "plot": figures only.
OUTPUT_ROOT = None     # None: SAP_ADMM/results (or SAP_ADMM/quick_results).
                       # Optional: use a different folder for a new parameter setting.

# %% Run the complete file with F5.
if __name__ == "__main__":
    run_experiments(("general_a",), quick=QUICK, mode=MODE, output_root=OUTPUT_ROOT)
