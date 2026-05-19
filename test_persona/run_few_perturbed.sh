#!/usr/bin/env bash
# Run few-shot (40 demos) for all perturbed seed folders on LLaMA70b eval data.
# ICM / gold baselines are expected to already exist under test_persona/results/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PERSONA_TAILOR_DIR="${PERSONA_TAILOR_DIR:-${ROOT_DIR}/data/Persona_Tailor}"
PERTURBED_BASE="${PERTURBED_BASE:-${PERSONA_TAILOR_DIR}/persona_eval_data_perturbed_llama70b}"
PERTURBED_SEEDS="${PERTURBED_SEEDS:-42 101 202 303 404}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results_perturbed_llama70b}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
MAX_SHOT="${MAX_SHOT:-40}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p "${RESULT_ROOT}"

check_vllm() {
  if ! curl -sf "${BASE_URL}/v1/models" >/dev/null; then
    echo "ERROR: vLLM is not reachable at ${BASE_URL}" >&2
    echo "Start the API server before running this script." >&2
    exit 1
  fi
}

resolve_personas() {
  local data_dir="$1"
  shopt -s nullglob
  local files=("${data_dir}"/*_fold*_test_persona.json)
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    echo "No persona test files found in ${data_dir}" >&2
    return 1
  fi

  for file in "${files[@]}"; do
    local base_name
    base_name="$(basename "${file}")"
    if [[ "${base_name}" =~ ^([A-Za-z0-9]+)_fold([0-9]+)_test_persona\.json$ ]]; then
      echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    fi
  done | sort -u
}

run_one_job() {
  local data_dir="$1"
  local result_dir="$2"
  local persona="$3"
  local fold="$4"

  local test_file train_file output_file
  test_file="${data_dir}/${persona}_fold${fold}_test_persona.json"
  train_file="${data_dir}/${persona}_fold${fold}_train_perturbed_persona.json"
  output_file="${result_dir}/${persona}_results_perturbed_few${MAX_SHOT}_fold${fold}.json"

  if [[ ! -f "${test_file}" ]]; then
    echo "Missing test file: ${test_file}" >&2
    exit 1
  fi
  if [[ ! -f "${train_file}" ]]; then
    echo "Missing train file: ${train_file}" >&2
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_file}" ]]; then
    echo "[skip] ${output_file}"
    return 0
  fi

  mkdir -p "${result_dir}"
  local cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/gen_few_shot.py"
    "${test_file}"
    "${train_file}"
    "${output_file}"
    "${BASE_URL}"
    "${MAX_SHOT}"
    "${MODEL}"
  )

  echo "Running perturbed ${persona} fold ${fold} few=${MAX_SHOT} -> ${output_file}"
  check_vllm
  "${cmd[@]}"
  echo "✓ Completed: ${output_file}"
  echo ""
}

run_seed_suite() {
  local seed="$1"
  local data_dir="${PERTURBED_BASE}_seed${seed}"
  local result_dir="${RESULT_ROOT}/seed${seed}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "Missing perturbed data directory: ${data_dir}" >&2
    exit 1
  fi

  mkdir -p "${result_dir}"

  echo "================================================================"
  echo "Seed: ${seed}"
  echo "  DATA_DIR=${data_dir}"
  echo "  RESULT_DIR=${result_dir}"
  echo "================================================================"

  local persona fold
  while read -r persona fold; do
    [[ -z "${persona}" ]] && continue
    run_one_job "${data_dir}" "${result_dir}" "${persona}" "${fold}"
  done < <(resolve_personas "${data_dir}")
}

print_accuracy_summary() {
  echo ""
  echo "================================================================"
  echo "Perturbed accuracy summary (few=${MAX_SHOT})"
  echo "================================================================"
  for seed in ${PERTURBED_SEEDS}; do
    local dir="${RESULT_ROOT}/seed${seed}"
    if [[ ! -d "${dir}" ]]; then
      continue
    fi
    local files=("${dir}"/*.json)
    if (( ${#files[@]} == 0 )); then
      continue
    fi
    echo ""
    echo "--- seed${seed} ---"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/utils_calc_acc.py" "${dir}"/*.json
  done
}

echo "Persona few-shot perturbed batch run (perturbed only)"
echo "  PERTURBED_BASE=${PERTURBED_BASE}"
echo "  PERTURBED_SEEDS=${PERTURBED_SEEDS}"
echo "  RESULT_ROOT=${RESULT_ROOT}"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  MAX_SHOT=${MAX_SHOT}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo "  (ICM / gold: use existing results under test_persona/results/)"
echo ""

check_vllm

for seed in ${PERTURBED_SEEDS}; do
  run_seed_suite "${seed}"
done

print_accuracy_summary

echo "All perturbed few-shot evaluations completed."
echo "Results written under: ${RESULT_ROOT}/seed{${PERTURBED_SEEDS// /,}}/"
