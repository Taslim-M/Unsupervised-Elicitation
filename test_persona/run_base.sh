#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/persona_eval_data}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p "${RESULT_DIR}"
shopt -s nullglob
test_files=("${DATA_DIR}"/*_fold*_test_persona.json)
shopt -u nullglob

if (( ${#test_files[@]} == 0 )); then
  echo "No persona test files found in ${DATA_DIR}" >&2
  exit 1
fi

for test_file in "${test_files[@]}"; do
  base_name="$(basename "${test_file}")"
  if [[ ! "${base_name}" =~ ^([A-Za-z0-9]+)_fold([0-9]+)_test_persona\.json$ ]]; then
    echo "Skipping unrecognized file: ${base_name}"
    continue
  fi

  persona="${BASH_REMATCH[1]}"
  fold="${BASH_REMATCH[2]}"
  output_file="${RESULT_DIR}/${persona}_results_zeroshot_base_fold${fold}.json"

  cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/gen_base.py"
    "${test_file}"
    "${output_file}"
    "${BASE_URL}"
    "${MODEL}"
  )

  echo "Running ${persona} fold ${fold}: ${cmd[*]}"
  "${cmd[@]}"
  echo "✓ Completed: ${output_file}"
  echo ""
done

echo "All zero-shot base persona evaluations completed."
