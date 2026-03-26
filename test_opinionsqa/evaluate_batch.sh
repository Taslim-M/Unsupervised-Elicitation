#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate JSON result files with utils_calc_acc.py.
# Usage:
#   bash test_opinionsqa/evaluate_batch.sh
#   bash test_opinionsqa/evaluate_batch.sh test_opinionsqa/results/icm_few_shot
#   bash test_opinionsqa/evaluate_batch.sh test_opinionsqa/results/icm_few_shot test_opinionsqa/utils_calc_acc.py

RESULT_DIR="${1:-/root/Unsupervised-Elicitation/test_opinionsqa/results/gold_few_shot}"
UTILS_PY="${2:-/root/Unsupervised-Elicitation/test_opinionsqa/utils_calc_acc.py}"

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: result dir not found: $RESULT_DIR" >&2
  exit 2
fi
if [[ ! -f "$UTILS_PY" ]]; then
  echo "ERROR: evaluator not found: $UTILS_PY" >&2
  exit 2
fi

# Pick a python executable.
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3/python not found in PATH" >&2
  exit 2
fi

# Create a timestamped log in the result dir.
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULT_DIR/eval_${TS}.log"

shopt -s nullglob
files=("$RESULT_DIR"/*.json)
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "No .json files found in: $RESULT_DIR" >&2
  exit 3
fi

{
  echo "=== Batch evaluate ==="
  echo "Time: $(date -Is)"
  echo "Dir:  $RESULT_DIR"
  echo "Util: $UTILS_PY"
  echo "Py:   $PYTHON_BIN"
  echo "Files: ${#files[@]}"
  echo

  for f in "${files[@]}"; do
    echo "--- $(basename "$f") ---"
    "$PYTHON_BIN" "$UTILS_PY" "$f"
    echo
  done

  echo "=== Done ==="
} | tee "$LOG_FILE"

echo "Saved log to: $LOG_FILE"
