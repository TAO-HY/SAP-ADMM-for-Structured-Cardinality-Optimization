"""Run once in Spyder if a required third-party package is missing.

This explicit setup script installs dependencies into Spyder's current Python
interpreter. The experiment scripts never install packages automatically.
"""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    requirements = Path(__file__).resolve().parent / "SAP_ADMM" / "requirements.txt"
    print(f"Installing dependencies for: {sys.executable}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    print("Dependencies installed. Restart the Spyder console, then run an experiment.", flush=True)
