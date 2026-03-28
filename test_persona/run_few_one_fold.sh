#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/persona_eval_data}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
LABEL_MODES="${LABEL_MODES:-icm gold}"
SHOT_GRID="${SHOT_GRID:-10 20 30 40}"
FOLD="${FOLD:-1}"
PERSONAS="${PERSONAS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p "${RESULT_DIR}"

resolve_personas() {
  if [[ -n "${PERSONAS}" ]]; then
    printf '%s\n' ${PERSONAS}
    return
  fi

  shopt -s nullglob
  local files=("${DATA_DIR}"/*_fold"${FOLD}"_test_persona.json)
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    echo "No persona test files found for fold ${FOLD} in ${DATA_DIR}" >&2
    exit 1
  fi

  for file in "${files[@]}"; do
    local base_name
    base_name="$(basename "${file}")"
    if [[ "${base_name}" =~ ^([A-Za-z0-9]+)_fold([0-9]+)_test_persona\.json$ ]]; then
      echo "${BASH_REMATCH[1]}"
    fi
  done | sort -u
}

echo "Persona few-shot single-fold run"
echo "  DATA_DIR=${DATA_DIR}"
echo "  RESULT_DIR=${RESULT_DIR}"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  LABEL_MODES=${LABEL_MODES}"
echo "  SHOT_GRID=${SHOT_GRID}"
echo "  FOLD=${FOLD}"
echo "  PERSONAS=${PERSONAS:-<auto>}"
echo ""

mapfile -t persona_list < <(resolve_personas)

for label_mode in ${LABEL_MODES}; do
  if [[ "${label_mode}" != "icm" && "${label_mode}" != "gold" ]]; then
    echo "LABEL_MODES entries must be 'icm' or 'gold', got: ${label_mode}" >&2
    exit 1
  fi

  for persona in "${persona_list[@]}"; do
    test_file="${DATA_DIR}/${persona}_fold${FOLD}_test_persona.json"
    train_file="${DATA_DIR}/${persona}_fold${FOLD}_train_${label_mode}_persona.json"

    if [[ ! -f "${test_file}" ]]; then
      echo "Missing test file: ${test_file}" >&2
      exit 1
    fi
    if [[ ! -f "${train_file}" ]]; then
      echo "Missing train file: ${train_file}" >&2
      exit 1
    fi

    for max_shot in ${SHOT_GRID}; do
      output_file="${RESULT_DIR}/${persona}_results_${label_mode}_few${max_shot}_fold${FOLD}.json"
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

      echo "Running ${persona} fold ${FOLD} ${label_mode} few-shot=${max_shot}: ${cmd[*]}"
      "${cmd[@]}"
      echo "✓ Completed: ${output_file}"
      echo ""
    done
  done
done

echo "All persona few-shot evaluations completed for fold ${FOLD}."
