#!/usr/bin/env python3
"""Build persona eval splits with randomly permuted train labels at ICM accuracy."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from gen_persona_eval_data import (
    load_jsonl,
    make_record,
    map_label,
    parse_filename,
    shuffled,
    write_json,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PERSONA_TAILOR_DIR = ROOT_DIR / "data" / "Persona_Tailor"
DEFAULT_INPUT_DIR = PERSONA_TAILOR_DIR / "persona_labels_llama3_70b"
DEFAULT_OUTPUT_BASE = PERSONA_TAILOR_DIR / "persona_eval_data_perturbed_llama70b"
DEFAULT_SEEDS = (42, 101, 202, 303, 404)


def invert_label(label_str):
    if label_str == "True":
        return "False"
    if label_str == "False":
        return "True"
    raise ValueError(f"Cannot invert label: {label_str!r}")


def make_record_with_output(row, persona, fold, output):
    return {
        "persona": persona,
        "source_fold": fold,
        "uid": row.get("uid"),
        "consistency_id": row.get("consistency_id"),
        "prompt": row.get("prompt"),
        "output": output,
    }


def record_key(row, fold):
    return (row.get("uid"), row.get("consistency_id"), fold)


def build_perturbed_train(train_candidates, seed_key):
    """Flip a random subset of train rows so perturbed accuracy matches ICM vs gold."""
    rng = random.Random(seed_key)
    num_errors = sum(1 for candidate in train_candidates if candidate["icm"] != candidate["gold"])

    indices = list(range(len(train_candidates)))
    rng.shuffle(indices)
    flip_indices = set(indices[:num_errors])

    perturbed_records = []
    for index, candidate in enumerate(train_candidates):
        if index in flip_indices:
            output = invert_label(candidate["gold"])
        else:
            output = candidate["gold"]
        perturbed_records.append(
            make_record_with_output(
                candidate["row"], candidate["persona"], candidate["fold"], output
            )
        )

    rng.shuffle(perturbed_records)
    return perturbed_records, num_errors


def output_dir_for_seed(output_base: Path, seed: int) -> Path:
    return Path(f"{output_base}_seed{seed}")


def main(input_dir=DEFAULT_INPUT_DIR, output_dir=None, random_seed=42):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = output_dir_for_seed(DEFAULT_OUTPUT_BASE, random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows = defaultdict(dict)
    for path in sorted(input_dir.glob("*.jsonl")):
        persona, fold = parse_filename(path)
        grouped_rows[persona][fold] = load_jsonl(path)

    if not grouped_rows:
        raise FileNotFoundError(f"No persona label files found in {input_dir}")

    print(f"[gen seed={random_seed}] {input_dir.name} -> {output_dir}")

    for persona, fold_rows in sorted(grouped_rows.items()):
        folds = sorted(fold_rows)
        print(f"[persona] {persona}: folds={folds}")

        for held_out_fold in folds:
            test_records = []
            train_candidates = []

            for fold, rows in sorted(fold_rows.items()):
                for row in rows:
                    if fold == held_out_fold:
                        record = make_record(row, persona, fold, "vanilla_label")
                        if record is not None:
                            test_records.append(record)
                        continue

                    if row.get("label") is None or row.get("vanilla_label") is None:
                        continue

                    gold = map_label(row["vanilla_label"])
                    icm = map_label(row["label"])

                    train_candidates.append(
                        {
                            "row": row,
                            "persona": persona,
                            "fold": fold,
                            "gold": gold,
                            "icm": icm,
                        }
                    )

            test_records = shuffled(
                test_records,
                f"{random_seed}:{persona}:fold{held_out_fold}:test",
            )
            perturbed_records, num_flipped = build_perturbed_train(
                train_candidates,
                f"{random_seed}:{persona}:fold{held_out_fold}:perturbed",
            )

            perturbed_by_key = {
                record_key(record, record["source_fold"]): record["output"]
                for record in perturbed_records
            }

            icm_correct = sum(1 for c in train_candidates if c["icm"] == c["gold"])
            pert_correct = sum(
                1
                for c in train_candidates
                if perturbed_by_key[record_key(c["row"], c["fold"])] == c["gold"]
            )
            overlap_with_icm = sum(
                1
                for c in train_candidates
                if perturbed_by_key[record_key(c["row"], c["fold"])] == c["icm"]
            )
            icm_total = len(train_candidates)

            test_path = output_dir / f"{persona}_fold{held_out_fold}_test_persona.json"
            perturbed_path = (
                output_dir / f"{persona}_fold{held_out_fold}_train_perturbed_persona.json"
            )

            write_json(test_path, test_records)
            write_json(perturbed_path, perturbed_records)

            icm_acc = icm_correct / icm_total if icm_total else 0.0
            pert_acc = pert_correct / icm_total if icm_total else 0.0
            print(
                f"  fold {held_out_fold}: test={len(test_records)} "
                f"train_perturbed={len(perturbed_records)} flipped={num_flipped} "
                f"icm_acc={icm_acc:.4f} pert_acc={pert_acc:.4f} "
                f"overlap_with_icm={overlap_with_icm}/{icm_total}"
            )

            if icm_correct != pert_correct:
                raise RuntimeError(
                    f"Accuracy mismatch for {persona} fold {held_out_fold}: "
                    f"icm={icm_correct}/{icm_total} pert={pert_correct}/{icm_total}"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate persona eval data with perturbed train labels that preserve "
            "ICM accuracy relative to gold labels."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with persona ICM label jsonl files.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base output directory; each seed writes to <base>_seed<seed>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single output directory (only used with one seed).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds; one output folder is created per seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.seeds) == 1 and args.output_dir is not None:
        main(args.input_dir, args.output_dir, args.seeds[0])
    elif len(args.seeds) == 1:
        main(args.input_dir, None, args.seeds[0])
    else:
        if args.output_dir is not None:
            raise SystemExit("--output-dir cannot be used with multiple --seeds.")
        for seed in args.seeds:
            print("")
            main(args.input_dir, output_dir_for_seed(args.output_base, seed), seed)
