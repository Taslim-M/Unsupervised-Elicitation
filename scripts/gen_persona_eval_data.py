#!/usr/bin/env python3
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "persona_results"
OUTPUT_DIR = ROOT_DIR / "data" / "persona_eval_data"
RANDOM_SEED = 42
FILENAME_RE = re.compile(r"^([A-Z]+)_(\d+)_fold(\d+)\.jsonl$")


def map_label(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if value in (1, "1", "True", "true"):
        return "True"
    if value in (0, "0", "False", "false"):
        return "False"
    raise ValueError(f"Unsupported label value: {value!r}")


def parse_filename(path):
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(
            f"Unexpected persona result filename: {path.name}. "
            "Expected <PERSONA>_<SIZE>_fold<K>.jsonl."
        )
    persona = match.group(1)
    fold = int(match.group(3))
    return persona, fold


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_record(row, persona, fold, label_key):
    label_value = row.get(label_key)
    if label_value is None:
        return None

    return {
        "persona": persona,
        "source_fold": fold,
        "uid": row.get("uid"),
        "consistency_id": row.get("consistency_id"),
        "prompt": row.get("prompt"),
        "output": map_label(label_value),
    }


def shuffled(records, seed_key):
    rng = random.Random(seed_key)
    copied = list(records)
    rng.shuffle(copied)
    return copied


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_rows = defaultdict(dict)
    for path in sorted(INPUT_DIR.glob("*.jsonl")):
        persona, fold = parse_filename(path)
        grouped_rows[persona][fold] = load_jsonl(path)

    if not grouped_rows:
        raise FileNotFoundError(f"No persona result files found in {INPUT_DIR}")

    for persona, fold_rows in sorted(grouped_rows.items()):
        folds = sorted(fold_rows)
        print(f"[persona] {persona}: folds={folds}")

        for held_out_fold in folds:
            test_records = []
            train_icm_records = []
            train_gold_records = []

            for fold, rows in sorted(fold_rows.items()):
                for row in rows:
                    if fold == held_out_fold:
                        record = make_record(row, persona, fold, "vanilla_label")
                        if record is not None:
                            test_records.append(record)
                        continue

                    icm_record = make_record(row, persona, fold, "label")
                    if icm_record is not None:
                        train_icm_records.append(icm_record)

                    gold_record = make_record(row, persona, fold, "vanilla_label")
                    if gold_record is not None:
                        train_gold_records.append(gold_record)

            test_records = shuffled(
                test_records,
                f"{RANDOM_SEED}:{persona}:fold{held_out_fold}:test",
            )
            train_icm_records = shuffled(
                train_icm_records,
                f"{RANDOM_SEED}:{persona}:fold{held_out_fold}:train_icm",
            )
            train_gold_records = shuffled(
                train_gold_records,
                f"{RANDOM_SEED}:{persona}:fold{held_out_fold}:train_gold",
            )

            test_path = OUTPUT_DIR / f"{persona}_fold{held_out_fold}_test_persona.json"
            train_icm_path = (
                OUTPUT_DIR / f"{persona}_fold{held_out_fold}_train_icm_persona.json"
            )
            train_gold_path = (
                OUTPUT_DIR / f"{persona}_fold{held_out_fold}_train_gold_persona.json"
            )

            write_json(test_path, test_records)
            write_json(train_icm_path, train_icm_records)
            write_json(train_gold_path, train_gold_records)

            print(
                f"  fold {held_out_fold}: "
                f"test={len(test_records)} "
                f"train_icm={len(train_icm_records)} "
                f"train_gold={len(train_gold_records)}"
            )


if __name__ == "__main__":
    main()
