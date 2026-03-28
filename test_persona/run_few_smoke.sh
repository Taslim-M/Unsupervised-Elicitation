#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/persona_eval_data}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results_smoke_icm}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
LABEL_MODE="${LABEL_MODE:-icm}"
PERSONAS="${PERSONAS:-DSN DCF}"
FOLDS="${FOLDS:-1}"
SHOT_GRID="${SHOT_GRID:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ "${LABEL_MODE}" != "icm" && "${LABEL_MODE}" != "gold" ]]; then
  echo "LABEL_MODE must be 'icm' or 'gold', got: ${LABEL_MODE}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "Smoke few-shot run"
echo "  DATA_DIR=${DATA_DIR}"
echo "  RESULT_DIR=${RESULT_DIR}"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  LABEL_MODE=${LABEL_MODE}"
echo "  PERSONAS=${PERSONAS}"
echo "  FOLDS=${FOLDS}"
echo "  SHOT_GRID=${SHOT_GRID}"
echo ""

for persona in ${PERSONAS}; do
  for fold in ${FOLDS}; do
    test_file="${DATA_DIR}/${persona}_fold${fold}_test_persona.json"
    train_file="${DATA_DIR}/${persona}_fold${fold}_train_${LABEL_MODE}_persona.json"

    if [[ ! -f "${test_file}" ]]; then
      echo "Missing test file: ${test_file}" >&2
      exit 1
    fi
    if [[ ! -f "${train_file}" ]]; then
      echo "Missing train file: ${train_file}" >&2
      exit 1
    fi

    for max_shot in ${SHOT_GRID}; do
      output_file="${RESULT_DIR}/${persona}_results_${LABEL_MODE}_few${max_shot}_fold${fold}.json"
      cmd=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/gen_few_shot.py"
        "${test_file}"
        "${train_file}"
        "${output_file}"
        "${BASE_URL}"
        "${max_shot}"
        "${MODEL}"
      )

      echo "Running ${persona} fold ${fold} ${LABEL_MODE} few-shot=${max_shot}: ${cmd[*]}"
      "${cmd[@]}"
      echo "✓ Completed: ${output_file}"
      echo ""
    done
  done
done

echo "Smoke few-shot evaluation completed."
