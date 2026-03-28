#!/usr/bin/env python3
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PERSONA_RE = re.compile(r"^([A-Za-z0-9]+)_")
RESULT_RE = re.compile(r"^[A-Za-z0-9]+_results_(.+)_fold\d+\.json$")


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


def infer_setting(path):
    match = RESULT_RE.match(path.name)
    if not match:
        return "other", "other"

    setting = match.group(1)
    if setting == "zeroshot_base":
        return "base", setting
    if setting == "zeroshot_chat":
        return "instruct", setting
    if setting.startswith("gold_few"):
        return "gold", setting
    if setting.startswith("icm_few"):
        return "icm", setting
    return "other", setting


def new_bucket():
    return {
        "files": set(),
        "personas": defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0}),
    }


def update_bucket(bucket, path, record):
    bucket["files"].add(path.name)
    persona = infer_persona(record, path)
    gold = norm(record.get("output"))
    pred = norm(record.get("generated_output"))

    if gold is None or pred is None or gold == "" or pred == "":
        bucket["personas"][persona]["skipped"] += 1
        return

    bucket["personas"][persona]["total"] += 1
    if gold == pred:
        bucket["personas"][persona]["correct"] += 1


def print_bucket(title, bucket):
    print(f"{title}")
    print(f"Files: {len(bucket['files'])}")

    total_correct = 0
    total_count = 0
    total_skipped = 0
    macro_values = []

    for persona, stats in sorted(bucket["personas"].items()):
        total = stats["total"]
        skipped = stats["skipped"]
        total_correct += stats["correct"]
        total_count += total
        total_skipped += skipped

        if total == 0:
            print(f"{persona}: No comparable records | skipped={skipped}")
            continue

        accuracy = stats["correct"] / total
        macro_values.append(accuracy)
        print(
            f"{persona}: Accuracy={accuracy:.4f} "
            f"({stats['correct']}/{total}) | skipped={skipped}"
        )

    if total_count == 0:
        print(f"No comparable records | skipped={total_skipped}")
        print("")
        return

    micro_accuracy = total_correct / total_count
    macro_accuracy = sum(macro_values) / len(macro_values) if macro_values else 0.0
    print(
        f"Micro Accuracy: {micro_accuracy:.4f} "
        f"({total_correct}/{total_count}) | skipped={total_skipped}"
    )
    print(f"Macro Accuracy: {macro_accuracy:.4f}")
    print("")


def summarize_many(paths):
    family_buckets = defaultdict(dict)

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list.")

        family, setting = infer_setting(path)
        if setting not in family_buckets[family]:
            family_buckets[family][setting] = new_bucket()

        for record in data:
            update_bucket(family_buckets[family][setting], path, record)

    print(f"Files: {len(paths)}")
    print("")

    family_order = ["base", "instruct", "gold", "icm", "other"]
    for family in family_order:
        if family not in family_buckets:
            continue

        print(f"=== {family} ===")
        settings = family_buckets[family]
        for setting in sorted(settings):
            print_bucket(f"setting={setting}", settings[setting])


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
