"""Run the module and input-interface test suites in separate processes."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for directory, script in (("main", "test_suite.py"), ("followup", "test_suite.py"),
                          ("prior", "test_prior.py"), ("tests", "test_input.py"),
                          ("transition_precision", "test_suite.py")):
    print(f"Running {directory}/{script}", flush=True)
    subprocess.run([sys.executable, script], cwd=root / directory, check=True)
