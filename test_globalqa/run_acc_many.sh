#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 file1.json file2.json ..."
    exit 1
fi

total_acc=0
count=0

for f in "$@"; do
    output=$(python utils_calc_acc.py "$f")
    acc=$(echo "$output" | grep -oP 'Accuracy: \K[0-9.]+')
    echo "$f: $output"
    total_acc=$(echo "$total_acc + $acc" | bc)
    count=$((count + 1))
done

if [ "$count" -gt 0 ]; then
    mean=$(echo "scale=4; $total_acc / $count" | bc)
    echo ""
    echo "Mean accuracy over $count files: $mean"
fi