#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="$HOME/.venvs/trajectory_basket_transition_precision"
valid_python() {
    "$1" -c 'import platform,sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) and (platform.system()!="Darwin" or platform.machine()=="arm64") else 1)' >/dev/null 2>&1
}
PY=""
for candidate in "${PYTHON:-}" \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
    "$(command -v python3.13 || true)" \
    "$(command -v python3.12 || true)" \
    "$(command -v python3.11 || true)" \
    "$(command -v python3 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ] && valid_python "$candidate"; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo 'Native Python 3.11-3.13 was not found. No environment was changed.' >&2
    echo 'Install a native Apple Silicon Python, or provide one explicitly:' >&2
    echo 'PYTHON=/absolute/path/to/python3.13 bash setup_mac.sh' >&2
    exit 1
fi
if [ -e "$TARGET" ]; then
    if [ ! -x "$TARGET/bin/python" ] || ! valid_python "$TARGET/bin/python"; then
        echo "Existing environment is incompatible: $TARGET" >&2
        echo 'It was not deleted or overwritten.' >&2
        exit 1
    fi
else
    mkdir -p "$HOME/.venvs"
    "$PY" -m venv "$TARGET"
fi
"$TARGET/bin/python" -m pip install --upgrade pip
"$TARGET/bin/python" -m pip install --only-binary=:all: -r "$ROOT/requirements.txt"
"$TARGET/bin/python" -m pip check
"$TARGET/bin/python" -c 'import platform,sys,numpy,openpyxl; print("Python:",platform.python_version()); print("Architecture:",platform.machine()); print("Executable:",sys.executable); print("NumPy:",numpy.__version__); print("openpyxl:",openpyxl.__version__)'
echo 'Setup complete. Run commands with:'
printf '"%s/bin/python" evaluate.py --plan-only\n' "$TARGET"
