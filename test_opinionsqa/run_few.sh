#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:8001"
MODEL="llama70b-gpu1"
USE_ICM=1
MODE="icm"

for fold in {1..4}; do
  for max_shot in 30 50 70 80; do
    cmd=(
      python gen_few_shot.py
      "fold${fold}_test_opinionsqa.json"
      "fold${fold}_train_${MODE}_opinionsqa.json"
      "results_${MODE}_few${max_shot}_fold${fold}.json"
      "${BASE_URL}"
      "${USE_ICM}"
      "${max_shot}"
      "${MODEL}"
    )
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
  done
done
