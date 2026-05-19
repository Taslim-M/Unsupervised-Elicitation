#!/usr/bin/env bash
# Few-shot evaluation on OpinionQA perturbed data (one ICM source model per run).
#
# Data layout:
#   data/OQA_Perturbed/<model>/seed_<seed>/fold{1..4}_{test,train_icm,train_gold}_opinionsqa.json
#
# Usage (from repo root or test_opinionsqa/):
#   bash test_opinionsqa/run_few_perturbed.sh llama8b
#   PERTURBED_MODEL=qwen14b BASE_URL=http://localhost:8001 \
#     bash test_opinionsqa/run_few_perturbed.sh
#
# Evaluates all 5 seeds for the given model. ICM / gold baselines use existing
# results under test_opinionsqa/results/ if you have already run run_few.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OQA_PERTURBED_BASE="${OQA_PERTURBED_BASE:-${ROOT_DIR}/data/OQA_Perturbed}"
PERTURBED_MODEL="${PERTURBED_MODEL:-${1:-}}"
PERTURBED_SEEDS="${PERTURBED_SEEDS:-42 101 202}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results_perturbed_${PERTURBED_MODEL}}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-llama70b-gpu0}"
MAX_SHOTS="${MAX_SHOTS:-30 50 70 80}"
USE_ICM="${USE_ICM:-0}"
GEN_PERTURBED="${GEN_PERTURBED:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

usage() {
  echo "Usage: PERTURBED_MODEL=<model> [env vars...] $0 [model]" >&2
  echo "  model: subdirectory under ${OQA_PERTURBED_BASE} (e.g. llama8b, qwen14b, qwen30b)" >&2
  if [[ -d "${OQA_PERTURBED_BASE}" ]]; then
    echo "Available models:" >&2
    for d in "${OQA_PERTURBED_BASE}"/*/; do
      [[ -d "${d}" ]] || continue
      echo "  - $(basename "${d}")" >&2
    done
  fi
}

if [[ -z "${PERTURBED_MODEL}" ]]; then
  usage
  exit 1
fi

MODEL_DIR="${OQA_PERTURBED_BASE}/${PERTURBED_MODEL}"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: model data not found: ${MODEL_DIR}" >&2
  usage
  exit 1
fi

mkdir -p "${RESULT_ROOT}"

check_vllm() {
  if ! curl -sf "${BASE_URL}/v1/models" >/dev/null; then
    echo "ERROR: vLLM is not reachable at ${BASE_URL}" >&2
    echo "Start the API server before running this script." >&2
    exit 1
  fi
}

ensure_perturbed_train() {
  local data_dir="$1"
  local fold="$2"

  local icm gold pert
  icm="${data_dir}/fold${fold}_train_icm_opinionsqa.json"
  gold="${data_dir}/fold${fold}_train_gold_opinionsqa.json"
  pert="${data_dir}/fold${fold}_train_perturbed_opinionsqa.json"

  if [[ -f "${pert}" ]]; then
    return 0
  fi

  if [[ "${GEN_PERTURBED}" != "1" ]]; then
    echo "Missing perturbed train file: ${pert}" >&2
    echo "Set GEN_PERTURBED=1 to build from icm+gold, or pre-generate the file." >&2
    exit 1
  fi

  if [[ ! -f "${icm}" || ! -f "${gold}" ]]; then
    echo "Cannot build perturbed train: missing ${icm} or ${gold}" >&2
    exit 1
  fi

  echo "  [gen] ${pert} <- icm + gold"
  "${PYTHON_BIN}" - "${icm}" "${gold}" "${pert}" <<'PY'
import json
import random
import sys
from pathlib import Path

icm_path, gold_path, out_path = map(Path, sys.argv[1:4])

def invert_label(label_str):
    if label_str == "True":
        return "False"
    if label_str == "False":
        return "True"
    return label_str

icm_data = json.loads(icm_path.read_text(encoding="utf-8"))
gold_by_input = {
    row["input"]: row["output"]
    for row in json.loads(gold_path.read_text(encoding="utf-8"))
}

original_wrong = []
original_correct = []
for row in icm_data:
    inp = row["input"]
    gold_lbl = gold_by_input.get(inp)
    if gold_lbl is None:
        continue
    if row["output"] != gold_lbl:
        original_wrong.append(row)
    else:
        original_correct.append(row)

count_to_flip = min(len(original_wrong), len(original_correct))
perturbed = []

for row in original_wrong:
    perturbed.append(
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": gold_by_input[row["input"]],
        }
    )

random.shuffle(original_correct)
for row in original_correct[:count_to_flip]:
    perturbed.append(
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": invert_label(gold_by_input[row["input"]]),
        }
    )
for row in original_correct[count_to_flip:]:
    perturbed.append(
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }
    )

random.shuffle(perturbed)
out_path.write_text(json.dumps(perturbed, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"       wrote {len(perturbed)} rows (flipped {count_to_flip})")
PY
}

run_one_job() {
  local data_dir="$1"
  local result_dir="$2"
  local fold="$3"
  local max_shot="$4"

  local test_file train_file output_file
  test_file="${data_dir}/fold${fold}_test_opinionsqa.json"
  train_file="${data_dir}/fold${fold}_train_perturbed_opinionsqa.json"
  output_file="${result_dir}/results_perturbed_few${max_shot}_fold${fold}.json"

  if [[ ! -f "${test_file}" ]]; then
    echo "Missing test file: ${test_file}" >&2
    exit 1
  fi

  ensure_perturbed_train "${data_dir}" "${fold}"

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
    "${USE_ICM}"
    "${max_shot}"
    "${MODEL}"
  )

  echo "Running fold ${fold} few=${max_shot} -> ${output_file}"
  check_vllm
  "${cmd[@]}"
  echo "✓ Completed: ${output_file}"
  echo ""
}

run_seed_suite() {
  local seed="$1"
  local data_dir="${MODEL_DIR}/seed_${seed}"
  local result_dir="${RESULT_ROOT}/seed_${seed}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "Missing seed data directory: ${data_dir}" >&2
    exit 1
  fi

  echo "================================================================"
  echo "Model: ${PERTURBED_MODEL}  Seed: ${seed}"
  echo "  DATA_DIR=${data_dir}"
  echo "  RESULT_DIR=${result_dir}"
  echo "================================================================"

  local fold max_shot
  for fold in 1 2 3 4; do
    for max_shot in ${MAX_SHOTS}; do
      run_one_job "${data_dir}" "${result_dir}" "${fold}" "${max_shot}"
    done
  done
}

print_accuracy_summary() {
  echo ""
  echo "================================================================"
  echo "Perturbed accuracy summary (model=${PERTURBED_MODEL})"
  echo "================================================================"
  for seed in ${PERTURBED_SEEDS}; do
    local dir="${RESULT_ROOT}/seed_${seed}"
    if [[ ! -d "${dir}" ]]; then
      continue
    fi
    shopt -s nullglob
    local files=("${dir}"/results_perturbed_few*.json)
    shopt -u nullglob
    if (( ${#files[@]} == 0 )); then
      continue
    fi
    echo ""
    echo "--- seed_${seed} ---"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/utils_calc_acc.py" "${files[@]}"
  done
}

echo "OpinionQA few-shot perturbed batch run (single ICM source model)"
echo "  OQA_PERTURBED_BASE=${OQA_PERTURBED_BASE}"
echo "  PERTURBED_MODEL=${PERTURBED_MODEL}"
echo "  PERTURBED_SEEDS=${PERTURBED_SEEDS}"
echo "  RESULT_ROOT=${RESULT_ROOT}"
echo "  BASE_URL=${BASE_URL}"
echo "  MODEL=${MODEL}"
echo "  MAX_SHOTS=${MAX_SHOTS}"
echo "  USE_ICM=${USE_ICM}"
echo "  GEN_PERTURBED=${GEN_PERTURBED}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo ""

check_vllm

for seed in ${PERTURBED_SEEDS}; do
  run_seed_suite "${seed}"
done

print_accuracy_summary

echo "All perturbed few-shot evaluations completed for ${PERTURBED_MODEL}."
echo "Results: ${RESULT_ROOT}/seed_{${PERTURBED_SEEDS// /,}}/"
