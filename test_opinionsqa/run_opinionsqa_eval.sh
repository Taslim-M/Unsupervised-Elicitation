#!/usr/bin/env bash
# OpinionQA batch eval: ICM + Gold few-shot + Instruct (zero-shot chat).
# Reads fold JSON directly from data/ — no symlinks into test_opinionsqa/.
#
# Usage:
#   bash test_opinionsqa/run_opinionsqa_eval.sh qwen30
#   bash test_opinionsqa/run_opinionsqa_eval.sh /path/to/qwen30_shuffled_results
#
# Regenerate shuffled splits from raw ICM jsonl, then eval:
#   GEN_SHUFFLED=1 bash test_opinionsqa/run_opinionsqa_eval.sh qwen30
#
# Environment:
#   DATA_DIR          — explicit shuffled JSON directory (overrides dataset name)
#   JSONL_DIR         — raw .jsonl dir under data/ (default: data/<dataset>)
#   GEN_SHUFFLED=1    — run scripts/gen_icm_data.py before eval
#   RESULT_ROOT       — output root (default: test_opinionsqa/results_<name>)
#   BASE_URL          — vLLM base URL (default: http://localhost:8000)
#   MODEL             — served model name (default: llama70b-gpu0)
#   MAX_SHOTS         — few-shot counts (default: "30 50 70 80")
#   FOLDS             — fold indices (default: "1 2 3 4")
#   RUN_MODES         — subset: icm gold instruct (default: all three)
#   SKIP_EXISTING=1   — skip jobs whose output JSON already exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"

DATASET="${1:-}"
DATA_DIR="${DATA_DIR:-}"
JSONL_DIR="${JSONL_DIR:-}"
GEN_SHUFFLED="${GEN_SHUFFLED:-0}"
RESULT_ROOT="${RESULT_ROOT:-}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
MAX_SHOTS="${MAX_SHOTS:-30 50 70 80}"
FOLDS="${FOLDS:-1 2 3 4}"
RUN_MODES="${RUN_MODES:-icm gold instruct}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

usage() {
  echo "Usage: $0 <dataset|data_dir>" >&2
  echo "" >&2
  echo "  dataset   Shortcut for a shuffled folder under \${DATA_ROOT}:" >&2
  echo "              qwen30  -> qwen30_shuffled_results" >&2
  echo "              qwen14  -> qwen14_shuffled_results" >&2
  echo "              llama8b -> llama8b_shuffled_results (from llama8b_labels_jsonl)" >&2
  echo "  data_dir  Full path to folder with fold*_test/train_*_opinionsqa.json" >&2
  echo "" >&2
  echo "  GEN_SHUFFLED=1  Rebuild shuffled JSON from \${DATA_ROOT}/<dataset>/*.jsonl first." >&2
  echo "  RUN_MODES=\"icm gold instruct\"  (instruct = zero-shot chat)" >&2
}

resolve_dataset_name() {
  local arg="$1"
  case "${arg}" in
    qwen30|qwen30b) echo "qwen30" ;;
    qwen14|qwen14b) echo "qwen14" ;;
    llama8b) echo "llama8b" ;;
    *) basename "${arg}" ;;
  esac
}

resolve_shuffled_dir() {
  local name="$1"
  case "${name}" in
    qwen30) echo "${DATA_ROOT}/qwen30_shuffled_results" ;;
    qwen14) echo "${DATA_ROOT}/qwen14_shuffled_results" ;;
    llama8b) echo "${DATA_ROOT}/llama8b_shuffled_results" ;;
    *) echo "${DATA_ROOT}/${name}_shuffled_results" ;;
  esac
}

if [[ -z "${DATASET}" ]]; then
  usage
  exit 1
fi

if [[ -z "${DATA_DIR}" ]]; then
  if [[ -d "${DATASET}" ]]; then
    DATA_DIR="$(cd "${DATASET}" && pwd)"
    DATASET_NAME="$(basename "${DATA_DIR}")"
  else
    DATASET_NAME="$(resolve_dataset_name "${DATASET}")"
    DATA_DIR="$(resolve_shuffled_dir "${DATASET_NAME}")"
  fi
else
  DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
  DATASET_NAME="$(basename "${DATA_DIR}")"
fi

if [[ -z "${RESULT_ROOT}" ]]; then
  RESULT_ROOT="${SCRIPT_DIR}/results_${DATASET_NAME}"
fi

mkdir -p "${RESULT_ROOT}"

mode_enabled() {
  local want="$1"
  [[ " ${RUN_MODES} " == *" ${want} "* ]]
}

check_vllm() {
  if ! curl -sf "${BASE_URL}/v1/models" >/dev/null; then
    echo "ERROR: vLLM is not reachable at ${BASE_URL}" >&2
    exit 1
  fi
}

regen_shuffled_data() {
  local jsonl_name="$1"
  local out_name="$2"
  local jsonl_path="${DATA_ROOT}/${jsonl_name}"
  if [[ ! -d "${jsonl_path}" ]]; then
    echo "ERROR: JSONL input not found: ${jsonl_path}" >&2
    exit 1
  fi
  echo "Regenerating shuffled data: ${jsonl_path} -> ${DATA_ROOT}/${out_name}"
  mkdir -p "${DATA_ROOT}/${out_name}"
  (
    cd "${DATA_ROOT}"
    "${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${ROOT_DIR}/scripts")
import gen_icm_data as g

g.INPUT_DIR = "./${jsonl_name}"
g.OUTPUT_DIR = "./${out_name}"
g.process_folds()
PY
  )
}

default_jsonl_dir() {
  case "${1}" in
    llama8b) echo "${DATA_ROOT}/llama8b_labels_jsonl" ;;
    qwen30) echo "${DATA_ROOT}/qwen30" ;;
    qwen14) echo "${DATA_ROOT}/qwen14" ;;
    *) echo "${DATA_ROOT}/${1}" ;;
  esac
}

maybe_regen_shuffled() {
  if [[ "${GEN_SHUFFLED}" != "1" ]]; then
    return 0
  fi
  local jsonl_path
  jsonl_path="${JSONL_DIR:-$(default_jsonl_dir "${DATASET_NAME}")}"
  if [[ ! -d "${jsonl_path}" ]]; then
    echo "ERROR: JSONL dir not found: ${jsonl_path}" >&2
    exit 1
  fi
  regen_shuffled_data "$(basename "${jsonl_path}")" "$(basename "${DATA_DIR}")"
}

run_few_job() {
  local mode="$1"
  local fold="$2"
  local max_shot="$3"

  local test_file train_file output_file result_dir
  test_file="${DATA_DIR}/fold${fold}_test_opinionsqa.json"
  train_file="${DATA_DIR}/fold${fold}_train_${mode}_opinionsqa.json"
  result_dir="${RESULT_ROOT}/${mode}"
  output_file="${result_dir}/results_${mode}_few${max_shot}_fold${fold}.json"

  if [[ ! -f "${test_file}" ]]; then
    echo "Missing: ${test_file}" >&2
    exit 1
  fi
  if [[ ! -f "${train_file}" ]]; then
    echo "Missing: ${train_file}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_file}" ]]; then
    echo "[skip] ${output_file}"
    return 0
  fi

  mkdir -p "${result_dir}"
  echo "Running ${mode} fold=${fold} few=${max_shot} -> ${output_file}"
  check_vllm
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gen_few_shot.py" \
    "${test_file}" \
    "${train_file}" \
    "${output_file}" \
    "${BASE_URL}" \
    0 \
    "${max_shot}" \
    "${MODEL}"
  echo "✓ ${output_file}"
  echo ""
}

run_instruct_job() {
  local fold="$1"
  local test_file output_file result_dir
  test_file="${DATA_DIR}/fold${fold}_test_opinionsqa.json"
  result_dir="${RESULT_ROOT}/instruct"
  output_file="${result_dir}/results_zeroshot_chat_fold${fold}.json"

  if [[ ! -f "${test_file}" ]]; then
    echo "Missing: ${test_file}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_file}" ]]; then
    echo "[skip] ${output_file}"
    return 0
  fi

  mkdir -p "${result_dir}"
  echo "Running instruct (zero-shot chat) fold=${fold} -> ${output_file}"
  check_vllm
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gen_zero_shot_chat.py" \
    "${test_file}" \
    "${output_file}" \
    "${BASE_URL}" \
    "${MODEL}"
  echo "✓ ${output_file}"
  echo ""
}

print_accuracy_summary() {
  echo ""
  echo "================================================================"
  echo "Accuracy summary (${DATASET_NAME})"
  echo "================================================================"
  local mode
  for mode in icm gold instruct; do
    mode_enabled "${mode}" || continue
    local dir="${RESULT_ROOT}/${mode}"
    [[ -d "${dir}" ]] || continue
    shopt -s nullglob
    local files=("${dir}"/*.json)
    shopt -u nullglob
    if (( ${#files[@]} == 0 )); then
      continue
    fi
    echo ""
    echo "--- ${mode} ---"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/utils_calc_acc.py" "${files[@]}"
  done
}

echo "OpinionQA eval batch"
echo "  DATA_DIR=${DATA_DIR}"
echo "  RESULT_ROOT=${RESULT_ROOT}"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  MAX_SHOTS=${MAX_SHOTS}"
echo "  FOLDS=${FOLDS}"
echo "  RUN_MODES=${RUN_MODES}"
echo "  GEN_SHUFFLED=${GEN_SHUFFLED}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo ""

maybe_regen_shuffled

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: data directory not found: ${DATA_DIR}" >&2
  echo "Run with GEN_SHUFFLED=1 if you need to build it from data/<model>/*.jsonl" >&2
  exit 1
fi

check_vllm

if mode_enabled icm || mode_enabled gold; then
  for fold in ${FOLDS}; do
    for max_shot in ${MAX_SHOTS}; do
      mode_enabled icm && run_few_job icm "${fold}" "${max_shot}"
      mode_enabled gold && run_few_job gold "${fold}" "${max_shot}"
    done
  done
fi

if mode_enabled instruct; then
  for fold in ${FOLDS}; do
    run_instruct_job "${fold}"
  done
fi

print_accuracy_summary

echo "Done. Results under: ${RESULT_ROOT}/{icm,gold,instruct}/"
