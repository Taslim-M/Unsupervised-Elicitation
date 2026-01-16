#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:8001"
MODEL="llama70b-gpu1"  # Specify your Instruct model

# Run zero-shot chat inference for all 4 folds
for fold in {1..4}; do
  test_file="fold${fold}_test_opinionsqa.json"
  output_file="results_zeroshot_chat_fold${fold}.json"
  
  cmd=(
    python gen_zero_shot_chat.py
    "${test_file}"
    "${output_file}"
    "${BASE_URL}"
    "${MODEL}"
  )
  
  echo "Running fold ${fold}: ${cmd[*]}"
  "${cmd[@]}"
  echo "✓ Completed: ${output_file}"
  echo ""
done

echo "All zero-shot chat inference completed (4 folds)!"
