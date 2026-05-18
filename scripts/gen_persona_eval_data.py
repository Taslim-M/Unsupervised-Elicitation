#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PERSONA_TAILOR_DIR = ROOT_DIR / "data" / "Persona_Tailor"
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "persona_results"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "persona_eval_data"
# Keep output folder names aligned with existing shuffled-results dirs.
OUTPUT_SUFFIX_ALIASES = {"llama3_70b": "llama70b"}
RANDOM_SEED = 42
FILENAME_RE = re.compile(r"^([A-Z]+)_(\d+)_fold(\d+)\.jsonl$")


def label_dir_suffix(label_dir: Path) -> str:
    return label_dir.name.removeprefix("persona_labels_")


def eval_output_dir_for_labels(label_dir: Path) -> Path:
    suffix = OUTPUT_SUFFIX_ALIASES.get(label_dir_suffix(label_dir), label_dir_suffix(label_dir))
    return PERSONA_TAILOR_DIR / f"persona_eval_data_{suffix}"


def discover_model_label_dirs() -> list[Path]:
    label_dirs = []
    for path in sorted(PERSONA_TAILOR_DIR.glob("persona_labels_*")):
        if path.is_dir() and list(path.glob("*.jsonl")):
            label_dirs.append(path)
    return label_dirs


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build persona_eval_data train/test splits from ICM-labeled jsonl files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing <PERSONA>_<SIZE>_fold<K>.jsonl label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write processed persona eval JSON files.",
    )
    parser.add_argument(
        "--all-persona-tailor",
        action="store_true",
        help=(
            "Process every Persona_Tailor/persona_labels_* folder that contains "
            "*.jsonl files, writing to matching persona_eval_data_* folders."
        ),
    )
    return parser.parse_args()


def main(input_dir=DEFAULT_INPUT_DIR, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is None:
        if input_dir.parent == PERSONA_TAILOR_DIR and input_dir.name.startswith("persona_labels_"):
            output_dir = eval_output_dir_for_labels(input_dir)
        else:
            output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] {input_dir.name} -> {output_dir}")

    grouped_rows = defaultdict(dict)
    for path in sorted(input_dir.glob("*.jsonl")):
        persona, fold = parse_filename(path)
        grouped_rows[persona][fold] = load_jsonl(path)

    if not grouped_rows:
        raise FileNotFoundError(f"No persona result files found in {input_dir}")

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

            test_path = output_dir / f"{persona}_fold{held_out_fold}_test_persona.json"
            train_icm_path = (
                output_dir / f"{persona}_fold{held_out_fold}_train_icm_persona.json"
            )
            train_gold_path = (
                output_dir / f"{persona}_fold{held_out_fold}_train_gold_persona.json"
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


def main_all_persona_tailor():
    label_dirs = discover_model_label_dirs()
    if not label_dirs:
        raise FileNotFoundError(
            f"No persona_labels_* folders with .jsonl files under {PERSONA_TAILOR_DIR}"
        )
    for label_dir in label_dirs:
        main(label_dir, eval_output_dir_for_labels(label_dir))


if __name__ == "__main__":
    args = parse_args()
    if args.all_persona_tailor:
        main_all_persona_tailor()
    else:
        main(args.input_dir, args.output_dir)
