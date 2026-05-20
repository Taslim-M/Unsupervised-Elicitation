#!/bin/bash
for f in log_*.jsonl; do
    [ -e "$f" ] || continue
    acc=$(tail -n 1 "$f" | grep -oE '"acc":[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+')
    # Strip everything up to and including "-file", and drop the .jsonl extension
    short="${f##*-file}"
    short="${short%.jsonl}"
    printf "%s\t%s\n" "$short" "$acc"
done