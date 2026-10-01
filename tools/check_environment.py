"""Check the interpreter and numerical dependencies."""
import platform
import sys

if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit("Use Python 3.11, 3.12, or 3.13 for this release.")
import numpy
import scipy
import openpyxl
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("Architecture:", platform.machine())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("openpyxl:", openpyxl.__version__)
if platform.system() == "Darwin" and platform.machine() != "arm64":
    print("Intel Python detected; on Apple Silicon select an arm64 interpreter.")
