#!/usr/bin/env python3
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PERSONA_RE = re.compile(r"^([A-Za-z0-9]+)_")


def norm(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower().startswith("true"):
        return "True"
    if text.lower().startswith("false"):
        return "False"
    return text


def resolve_paths(target):
    path = Path(target)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [Path(match) for match in sorted(glob.glob(target))]


def infer_persona(record, path):
    persona = record.get("persona")
    if persona:
        return str(persona)
    match = PERSONA_RE.match(path.name)
    if match:
        return match.group(1)
    return "UNKNOWN"


def summarize_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")

    total = 0
    correct = 0
    skipped = 0
    persona_counts = defaultdict(int)

    for record in data:
        gold = norm(record.get("output"))
        pred = norm(record.get("generated_output"))
        persona = infer_persona(record, path)
        persona_counts[persona] += 1

        if gold is None or pred is None or gold == "" or pred == "":
            skipped += 1
            continue
        total += 1
        if gold == pred:
            correct += 1

    persona_desc = ", ".join(
        f"{persona}:{count}" for persona, count in sorted(persona_counts.items())
    )
    if total == 0:
        print(f"{path.name}: No comparable records. skipped={skipped} personas=[{persona_desc}]")
        return

    accuracy = correct / total
    print(
        f"{path.name}: Accuracy={accuracy:.4f} ({correct}/{total}) "
        f"| skipped={skipped} | personas=[{persona_desc}]"
    )


def summarize_many(paths):
    persona_stats = defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0})
    overall_correct = 0
    overall_total = 0
    overall_skipped = 0

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list.")

        for record in data:
            persona = infer_persona(record, path)
            gold = norm(record.get("output"))
            pred = norm(record.get("generated_output"))

            if gold is None or pred is None or gold == "" or pred == "":
                persona_stats[persona]["skipped"] += 1
                overall_skipped += 1
                continue

            persona_stats[persona]["total"] += 1
            overall_total += 1
            if gold == pred:
                persona_stats[persona]["correct"] += 1
                overall_correct += 1

    if overall_total == 0:
        print(f"No comparable records across {len(paths)} files. skipped={overall_skipped}")
        return

    macro_values = []
    print(f"Files: {len(paths)}")
    for persona, stats in sorted(persona_stats.items()):
        total = stats["total"]
        skipped = stats["skipped"]
        if total == 0:
            print(f"{persona}: No comparable records | skipped={skipped}")
            continue
        accuracy = stats["correct"] / total
        macro_values.append(accuracy)
        print(
            f"{persona}: Accuracy={accuracy:.4f} "
            f"({stats['correct']}/{total}) | skipped={skipped}"
        )

    micro_accuracy = overall_correct / overall_total
    macro_accuracy = sum(macro_values) / len(macro_values) if macro_values else 0.0
    print(
        f"Micro Accuracy: {micro_accuracy:.4f} "
        f"({overall_correct}/{overall_total}) | skipped={overall_skipped}"
    )
    print(f"Macro Accuracy: {macro_accuracy:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils_calc_acc.py <file|directory|glob>")
        sys.exit(1)

    target = sys.argv[1]
    paths = resolve_paths(target)
    if not paths:
        print(f"No JSON files found for target: {target}")
        sys.exit(2)

    if len(paths) == 1 and paths[0].is_file():
        summarize_file(paths[0])
        return

    summarize_many(paths)


if __name__ == "__main__":
    main()
