#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path

from completion_utils import (
    complete,
    extract_pred_confidence_from_completion_logprobs,
    normalize_bool,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_URL = "http://localhost:8000"
IN_PATH = ROOT_DIR / "data" / "persona_eval_data" / "DSN_fold1_test_persona.json"
TRAIN_PATH = ROOT_DIR / "data" / "persona_eval_data" / "DSN_fold1_train_icm_persona.json"
OUT_PATH = SCRIPT_DIR / "results" / "DSN_results_icm_few10_fold1.json"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LABEL_MODE_RE = re.compile(r"_train_(icm|gold)_persona$")


def infer_label_mode(train_path):
    match = LABEL_MODE_RE.search(Path(train_path).stem)
    if match:
        return match.group(1)
    return "unknown"


def build_prompt(demonstrations, example):
    demo_prefix = "".join(
        f"{demo['prompt']}{demo['output']}\n\n" for demo in demonstrations
    )
    return f"{demo_prefix}{example['prompt']}"


def main(
    in_path=IN_PATH,
    train_path=TRAIN_PATH,
    out_path=OUT_PATH,
    base_url=BASE_URL,
    max_shots=None,
    model=MODEL,
):
    test_data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    train_data = json.loads(Path(train_path).read_text(encoding="utf-8"))
    if not isinstance(test_data, list):
        raise ValueError("Test data must be a JSON list.")
    if not isinstance(train_data, list):
        raise ValueError("Train data must be a JSON list.")

    train_size = len(train_data)
    requested_shots = train_size if max_shots is None else int(max_shots)
    used_shots = min(requested_shots, train_size)
    demonstrations = train_data[:used_shots]
    label_mode = infer_label_mode(train_path)

    print(
        f"[info] train_size={train_size} requested_shots={requested_shots} "
        f"used_shots={used_shots} label_mode={label_mode}"
    )

    results = []
    for idx, item in enumerate(test_data, 1):
        raw = ""
        logprobs = None
        prompt = build_prompt(demonstrations, item)
        try:
            raw, logprobs = complete(base_url, model, prompt, max_tokens=8, logprobs_k=5)
            prediction = normalize_bool(raw)
            if not prediction:
                raw, logprobs = complete(
                    base_url, model, prompt, max_tokens=16, logprobs_k=5
                )
                prediction = normalize_bool(raw)
        except Exception as exc:
            print(f"[error] item {idx} failed: {exc}")
            prediction = ""

        if not prediction:
            prediction = normalize_bool(raw)

        result = dict(item)
        result["generated_output"] = prediction
        result["pred_confidence"] = (
            extract_pred_confidence_from_completion_logprobs(logprobs, prediction)
            if prediction in ("True", "False")
            else None
        )
        result["_requested_shots"] = requested_shots
        result["_used_shots"] = used_shots
        result["_train_size"] = train_size
        result["_train_label_mode"] = label_mode
        results.append(result)
        time.sleep(0.005)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] Wrote {out_path} ({len(results)} items).")


if __name__ == "__main__":
    in_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    train_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_PATH
    out_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_PATH
    url_arg = sys.argv[4] if len(sys.argv) > 4 else BASE_URL
    max_shots_arg = int(sys.argv[5]) if len(sys.argv) > 5 else None
    model_arg = sys.argv[6] if len(sys.argv) > 6 else MODEL
    main(in_arg, train_arg, out_arg, url_arg, max_shots_arg, model_arg)
