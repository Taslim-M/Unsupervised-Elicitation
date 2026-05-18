#!/usr/bin/env python3
"""Build fold-level shuffled train/test splits from Persona_Tailor ICM jsonl labels."""

import argparse
import glob
import json
import os
import random
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PERSONA_TAILOR_DIR = ROOT_DIR / "data" / "Persona_Tailor"
DEFAULT_INPUT_DIR = PERSONA_TAILOR_DIR / "persona_labels_llama8b"
DEFAULT_OUTPUT_DIR = PERSONA_TAILOR_DIR / "persona_shuffled_results_llama8b"
OUTPUT_SUFFIX_ALIASES = {"llama3_70b": "llama70b"}
RANDOM_SEED = 42


def label_dir_suffix(label_dir: Path) -> str:
    return label_dir.name.removeprefix("persona_labels_")


def shuffled_output_dir_for_labels(label_dir: Path) -> Path:
    suffix = OUTPUT_SUFFIX_ALIASES.get(label_dir_suffix(label_dir), label_dir_suffix(label_dir))
    return PERSONA_TAILOR_DIR / f"persona_shuffled_results_{suffix}"


def discover_model_label_dirs() -> list[Path]:
    label_dirs = []
    for path in sorted(PERSONA_TAILOR_DIR.glob("persona_labels_*")):
        if path.is_dir() and list(path.glob("*.jsonl")):
            label_dirs.append(path)
    return label_dirs


def map_label(val):
    if val is True or val == 1:
        return "True"
    if val is False or val == 0:
        return "False"
    return str(val)


def get_fold_index(filename):
    match = re.search(r"fold(\d)\.jsonl$", filename, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def process_folds(input_dir, output_dir):
    random.seed(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    if not all_files:
        raise FileNotFoundError(f"No .jsonl label files found in {input_dir}")

    print(f"Found {len(all_files)} label files in {input_dir}")

    for fold_idx in range(1, 5):
        test_data = []
        train_icm_data = []
        train_gold_data = []

        for file_path in all_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("label") is None:
                        continue

                    instruction = "Label the input claim as True or False"
                    input_text = row.get("prompt", "")
                    item_test = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": map_label(row["vanilla_label"]),
                    }
                    item_train_icm = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": map_label(row["label"]),
                    }
                    item_train_gold = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": map_label(row["vanilla_label"]),
                    }

                    row_fold_idx = get_fold_index(filename)
                    if row_fold_idx == fold_idx:
                        test_data.append(item_test)
                    else:
                        train_icm_data.append(item_train_icm)
                        train_gold_data.append(item_train_gold)

        random.shuffle(test_data)
        random.shuffle(train_icm_data)
        random.shuffle(train_gold_data)

        outputs = [
            (f"fold{fold_idx}_test_opinionsqa.json", test_data),
            (f"fold{fold_idx}_train_icm_opinionsqa.json", train_icm_data),
            (f"fold{fold_idx}_train_gold_opinionsqa.json", train_gold_data),
        ]
        for name, records in outputs:
            path = os.path.join(output_dir, name)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=4, ensure_ascii=False)
            print(f"  wrote {name}: {len(records)} samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build fold-level shuffled persona OpinionQA-style eval files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--all-persona-tailor",
        action="store_true",
        help=(
            "Process every Persona_Tailor/persona_labels_* folder that contains "
            "*.jsonl files, writing to matching persona_shuffled_results_* folders."
        ),
    )
    return parser.parse_args()


def main(input_dir=DEFAULT_INPUT_DIR, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is None:
        if input_dir.parent == PERSONA_TAILOR_DIR and input_dir.name.startswith("persona_labels_"):
            output_dir = shuffled_output_dir_for_labels(input_dir)
        else:
            output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    print(f"[gen] {input_dir.name} -> {output_dir}")
    process_folds(str(input_dir), str(output_dir))
    print(f"Done. Output written to {output_dir}")


def main_all_persona_tailor():
    label_dirs = discover_model_label_dirs()
    if not label_dirs:
        raise FileNotFoundError(
            f"No persona_labels_* folders with .jsonl files under {PERSONA_TAILOR_DIR}"
        )
    for label_dir in label_dirs:
        main(label_dir, shuffled_output_dir_for_labels(label_dir))
        print()


if __name__ == "__main__":
    args = parse_args()
    if args.all_persona_tailor:
        main_all_persona_tailor()
    else:
        main(args.input_dir, args.output_dir)
