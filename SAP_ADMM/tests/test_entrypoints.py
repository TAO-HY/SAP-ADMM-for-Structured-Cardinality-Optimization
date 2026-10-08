"""Regression checks for running an extracted checkout without installation."""
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'run_all.py', 'run_signal.py', 'run_general_a.py', 'run_mnist.py',
    'plot_signal.py', 'plot_general_a.py', 'plot_mnist.py',
])
def test_direct_script_imports_from_an_unrelated_directory(tmp_path, name):
    result = subprocess.run(
        [sys.executable, '-I', str(PROJECT / 'experiments' / name), '--help'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


@pytest.mark.parametrize('name', ['run_all.py', 'run_signal.py', 'run_general_a.py', 'run_mnist.py'])
def test_spyder_entry_uses_this_checkout_without_installation(tmp_path, name):
    entry = PROJECT.parent / name
    if not entry.is_file():
        pytest.skip('Outer Spyder entry scripts are included in the full repository archive')
    code = (
        'import runpy; from pathlib import Path; '
        f'ns = runpy.run_path({str(entry)!r}); '
        'import sap_admm; '
        f'assert Path(sap_admm.__file__).resolve().is_relative_to(Path({str(PROJECT / "src")!r})); '
        'assert ns["QUICK"] is False; assert ns["MODE"] == "all"'
    )
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
