#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.venvs/trajectory_basket_transition_precision/bin/python"
if [ ! -x "$PY" ]; then
    bash "$ROOT/setup_mac.sh"
fi
cd "$ROOT"
"$PY" test_suite.py
OUT="${OUT:-$HOME/basket_runs/results_transition_precision}"
mkdir -p "$(dirname "$OUT")"
ARGS=(evaluate.py --reps "${REPS:-10000}" --workers "${WORKERS:-20}" --out "$OUT")
if [ "${RESUME:-0}" = 1 ]; then ARGS+=(--resume); fi
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$PY" "${ARGS[@]}"
else
    "$PY" "${ARGS[@]}"
fi

"$PY" report.py --results "$OUT"
