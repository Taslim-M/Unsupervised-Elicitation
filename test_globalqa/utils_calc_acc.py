#!/usr/bin/env python3
import json, sys
from pathlib import Path

def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    # normalize booleans/strings like "true"/"False "
    if s.lower().startswith("true"):  return "True"
    if s.lower().startswith("false"): return "False"
    return s  # fallback, compares as-is

def main():
    if len(sys.argv) < 2:
        print("Usage: python accuracy.py results_alpaca_test.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Input must be a JSON list.")
        sys.exit(2)

    total = correct = skipped = 0
    for rec in data:
        gold = norm(rec.get("output"))
        pred = norm(rec.get("generated_output"))
        if gold is None or pred is None or gold == "":
            skipped += 1
            continue
        total += 1
        if pred == gold:
            correct += 1

    if total == 0:
        print(f"No comparable records (skipped={skipped}).")
        return

    acc = correct / total
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})  | skipped: {skipped}")

if __name__ == "__main__":
    main()
